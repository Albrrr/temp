import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchhd import embeddings
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from modules.resnet import ResNet34
from modules.losses import LovaszSoftmax, BoundaryLoss
from modules.ioueval import iouEval
from modules.trainer import AverageMeter, WarmupExpDecayLR, save_checkpoint

class STEQuantise(torch.autograd.Function):
    """
    Straight-Through Estimator for hard_quantize (sign function).
    Forward : q = sign(x)   (±1)
    Backward: dL/dx = dL/dq  (identity pass-through)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def ste_quantise(x: torch.Tensor) -> torch.Tensor:
    return STEQuantise.apply(x)


def hd_separation_loss(feats_flat: torch.Tensor, proj_weight: torch.Tensor, class_protos: torch.Tensor, labels_flat: torch.Tensor, ignore_index: int = 0, temperature: float = 0.1, max_pixels: int = 4096, hard_quantize: bool = True) -> torch.Tensor:
    """
    Per-pixel InfoNCE contrastive loss in HD space.

    Memory note: N can be ~32 768 per image and hd_dim = 10 000.  The full
    similarity matrix would be NxC which is fine (32768x17), but the
    intermediate (N, hd_dim) projection is ~1.3 GB in float32 for N=32768.
    We therefore subsample `max_pixels` pixels per call.
    """
    valid_mask = labels_flat != ignore_index
    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return feats_flat.new_zeros(1).squeeze()

    if valid_idx.numel() > max_pixels:
        perm = torch.randperm(valid_idx.numel(), device=feats_flat.device)
        valid_idx = valid_idx[perm[:max_pixels]]

    feats_sub = feats_flat[valid_idx]
    labels_sub = labels_flat[valid_idx]

    s = feats_sub @ proj_weight.t()

    if hard_quantize:
        q = ste_quantise(s)
    else:
        q = s

    q_norm = F.normalize(q, dim=1)
    sims = q_norm @ class_protos.t()
    sims = sims / temperature

    loss = F.cross_entropy(sims, labels_sub)
    return loss

class _BaseHDTrainer:
    """
    Shared infrastructure for TeZO and DFA trainers.
    """
    def __init__(
        self,
        num_classes: int,
        loss_weights: torch.Tensor,
        hd_dim: int,
        feat_dim: int,
        log_dir: str,
        device: torch.device,
        lr: float = 0.01,
        momentum: float = 0.9,
        w_decay: float = 1e-4,
        wup_epochs: float = 1.0,
        lr_decay: float = 0.99,
        steps_per_epoch: int = 1,
        aux_loss: bool = True,
        aux_lambda: float = 0.4,
        sep_lambda: float = 0.5,
        sep_temperature: float = 0.1,
        sep_max_pixels: int = 4096,
        ignore_index: int = 0,
    ):
        self.num_classes = num_classes
        self.hd_dim = hd_dim
        self.feat_dim = feat_dim
        self.log_dir = log_dir
        self.device = device
        self.aux_loss = aux_loss
        self.aux_lambda = aux_lambda
        self.sep_lambda = sep_lambda
        self.sep_temperature = sep_temperature
        self.sep_max_pixels = sep_max_pixels
        self.ignore_index = ignore_index

        ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

        if hasattr(self, 'model_override'):
            self.model = self.model_override
        else:
            self.model = ResNet34(num_classes, aux=aux_loss).to(device)

        _proj_emb = embeddings.Projection(feat_dim, hd_dim)
        self.rp_weight = _proj_emb.weight.detach().to(device)
        self.class_protos = torch.zeros(num_classes, hd_dim, device=device)

        self.criterion = nn.NLLLoss(weight=loss_weights.to(device)).to(device)
        self.ls = LovaszSoftmax(ignore=0).to(device)
        self.bd = BoundaryLoss().to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        warmup_steps = int(wup_epochs * steps_per_epoch)
        final_decay = lr_decay ** (1.0 / steps_per_epoch)
        self.scheduler = WarmupExpDecayLR(self.optimizer, lr, warmup_steps, final_decay)

        self.evaluator = iouEval(num_classes, device, ignore_idx)

    def set_class_protos(self, protos: torch.Tensor):
        """
        Update class prototype HVs from the HDCModel's classify.weight.
        Call this before each epoch so the separation loss uses current protos.
        protos: (C, hd_dim) normalised float tensor.
        """
        self.class_protos = protos.detach().to(self.device)

    def train(self, train_loader, num_epochs: int):
        seg_losses = []
        sep_losses = []
        for epoch in range(num_epochs):
            acc, iou, l_seg, l_sep = self._train_epoch(
                train_loader, epoch, num_epochs)
            seg_losses.append(l_seg)
            sep_losses.append(l_sep)
            print(f"[Epoch {epoch+1}/{num_epochs}]  seg={l_seg:.4f}  sep={l_sep:.4f}  acc={acc:.4f}  iou={iou:.4f}")
            net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            save_checkpoint({"epoch": epoch, "state_dict": net.state_dict(), "optimizer": self.optimizer.state_dict()},self.log_dir)

        os.makedirs("logs/graphs", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), seg_losses, marker='o', linestyle='-', color='b', label='Segmentation Loss')
        plt.plot(range(1, num_epochs + 1), sep_losses, marker='x', linestyle='--', color='r', label='Separation Loss')
        plt.title(f'{self.__class__.__name__} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        graph_name = f"{self.__class__.__name__}_{os.path.basename(os.path.normpath(self.log_dir))}_loss.png"
        plt.savefig(os.path.join("logs/graphs", graph_name))
        plt.close()

    def validate(self, val_loader):
        self.model.eval()
        self.evaluator.reset()
        with torch.no_grad():
            for in_vol, _, proj_labels, *_ in tqdm(val_loader, desc="Validate"):
                in_vol = in_vol.to(self.device)
                proj_labels = proj_labels.to(self.device).long()
                out  = self.model(in_vol)
                pred = out[0] if isinstance(out, tuple) else out
                self.evaluator.addBatch(pred.argmax(1), proj_labels)
        acc = self.evaluator.getacc()
        iou, _ = self.evaluator.getIoU()
        print(f"[Validation] acc={acc:.4f}  iou={iou:.4f}")
        return acc.item(), iou.item()

    def _seg_loss(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (self.criterion(torch.log(pred.clamp(min=1e-8)), labels) + 1.5 * self.ls(pred, labels.long()) + self.bd(pred, labels))

    def _full_seg_loss(self, in_vol, proj_labels):
        """Standard forward + seg loss. Returns (loss, pred)."""
        out = self.model(in_vol)
        if self.aux_loss:
            pred, aux = out
            z2, z4, z8 = aux
            lam = self.aux_lambda
            loss = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
        else:
            pred = out
            loss = self._seg_loss(pred, proj_labels)
        return loss, pred

    def _sep_loss_from_feats(self, feats: torch.Tensor, proj_labels: torch.Tensor) -> torch.Tensor:
        """Project features via RP, apply STE, compute InfoNCE separation loss."""
        B, C_f, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C_f)
        labels_flat = proj_labels.reshape(-1)
        return hd_separation_loss(
            feats_flat, self.rp_weight, self.class_protos,
            labels_flat, self.ignore_index,
            self.sep_temperature, self.sep_max_pixels,
        )

    def _eval_metrics(self, pred, proj_labels):
        self.evaluator.reset()
        self.evaluator.addBatch(pred.argmax(1), proj_labels)
        return self.evaluator.getacc(), self.evaluator.getIoU()[0]

    def _train_epoch(self, loader, epoch, total_epochs):
        raise NotImplementedError

class TeZOTrainer(_BaseHDTrainer):
    """
    Temporal Zero-Order Optimisation trainer.
    """
    DEFAULT_HEAD_NAMES = ("conv_1", "conv_2", "semantic_output", "aux_head")

    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), zo_epsilon: float = 1e-3, zo_ema_beta: float = 0.9, zo_n_samples: int = 1, head_param_names: Tuple[str, ...] = DEFAULT_HEAD_NAMES, **base_kwargs):
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        self.zo_epsilon = zo_epsilon
        self.zo_ema_beta = zo_ema_beta
        self.zo_n_samples = zo_n_samples
        self.head_param_names = head_param_names

        self._ema_grads: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
        }

        head_params = [p for n, p in self.model.named_parameters() if any(h in n for h in self.head_param_names)]
        backbone_params = [p for n, p in self.model.named_parameters() if not any(h in n for h in self.head_param_names)]

        self.head_optimizer = optim.SGD(
            head_params,
            lr=self.optimizer.param_groups[0]["lr"],
            momentum=self.optimizer.param_groups[0]["momentum"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        self.optimizer = self.head_optimizer
        self._backbone_params: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in self.model.named_parameters()
            if not any(h in n for h in self.head_param_names)
        ]

    @torch.no_grad()
    def _zo_loss(self, in_vol: torch.Tensor, proj_labels: torch.Tensor) -> torch.Tensor:
        """
        Scalar loss for ZO perturbation.
        """
        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        out = net(in_vol, return_feat=True)
        
        if self.aux_loss:
            pred, _, feats = out
        else:
            pred, feats = out

        l_seg = F.nll_loss(torch.log(pred.clamp(min=1e-8)), proj_labels, weight=self.criterion.weight, ignore_index=self.ignore_index,)
        l_sep = self._sep_loss_from_feats(feats, proj_labels)

        return l_seg + (self.sep_lambda * l_sep)

    @torch.no_grad()
    def _zo_step(self, in_vol: torch.Tensor, proj_labels: torch.Tensor):
        """
        Estimate ZO gradient for backbone params, EMA-smooth, apply update.
        """
        eps = self.zo_epsilon
        accumulated = {n: torch.zeros_like(p) for n, p in self._backbone_params}

        for _ in range(self.zo_n_samples):
            zs = {n: torch.randn_like(p) for n, p in self._backbone_params}

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=eps)
            loss_pos = self._zo_loss(in_vol, proj_labels)

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=-2 * eps)
            loss_neg = self._zo_loss(in_vol, proj_labels)

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=eps)

            g_scalar = (loss_pos - loss_neg) / (2 * eps)
            for n, _ in self._backbone_params:
                accumulated[n].add_(zs[n], alpha=g_scalar.item())

        for n in accumulated:
            accumulated[n].div_(self.zo_n_samples)

        β = self.zo_ema_beta
        for n, p in self._backbone_params:
            self._ema_grads[n].mul_(β).add_(accumulated[n], alpha=1 - β)

        lr = self.head_optimizer.param_groups[0]["lr"]
        for n, p in self._backbone_params:
            p.data.add_(self._ema_grads[n], alpha=-lr)

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        sep_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[TeZO] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self._zo_step(in_vol, proj_labels)

            self.head_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                seg_loss, pred = self._full_seg_loss(in_vol, proj_labels)

            seg_loss.backward()
            self.head_optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                net = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                feats = net(in_vol, only_feat=True).detach()

            with torch.no_grad():
                sep_loss = self.sep_lambda * self._sep_loss_from_feats(feats, proj_labels)
            sep_loss_val = sep_loss.item()

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            sep_m.update(sep_loss_val, N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, sep_m.avg

class DFATrainer(_BaseHDTrainer):
    """
    Direct Feedback Alignment trainer.
    """

    DEFAULT_DFA_LAYERS = ("layer1", "layer2", "layer3", "layer4")
    HEAD_PARAM_NAMES = ("conv_1", "conv_2", "semantic_output", "aux_head")

    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), dfa_layer_names: Tuple[str, ...] = DEFAULT_DFA_LAYERS, dfa_lr: float = 1e-4, **base_kwargs):
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        self.dfa_layer_names = dfa_layer_names
        self.dfa_lr = dfa_lr

        head_params     = [p for n, p in self.model.named_parameters() if any(h in n for h in self.HEAD_PARAM_NAMES)]
        backbone_params = [p for n, p in self.model.named_parameters() if not any(h in n for h in self.HEAD_PARAM_NAMES)]

        self.head_optimizer = optim.SGD(
            head_params,
            lr=self.optimizer.param_groups[0]["lr"],
            momentum=self.optimizer.param_groups[0]["momentum"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        self.backbone_optimizer = optim.SGD(
            backbone_params, lr=dfa_lr,
            momentum=self.optimizer.param_groups[0]["momentum"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        self.optimizer = self.head_optimizer

        self._feedback: Dict[str, torch.Tensor] = {}
        self._layer_out_shape: Dict[str, torch.Size] = {}
        self._delta: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        self._register_feedback_matrices()

    def _register_feedback_matrices(self):
        """
        Registers matrices and attaches a forward hook that applies a Tensor-level 
        backward hook. This prevents gradient masking and combines DFA with L_seg.
        """
        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        dummy = torch.zeros(1, 5, 64, 512, device=self.device)
        shapes: Dict[str, torch.Size] = {}

        def _shape_hook(name):
            def hook(module, inp, out):
                shapes[name] = out.shape
            return hook

        tmp_hooks = []
        for name, module in net.named_modules():
            if name in self.dfa_layer_names:
                tmp_hooks.append(module.register_forward_hook(_shape_hook(name)))

        with torch.no_grad():
            net(dummy)

        for h in tmp_hooks:
            h.remove()

        for name, shape in shapes.items():
            C_l = shape[1] 
            B_l = torch.randn(self.hd_dim, C_l, device=self.device) / math.sqrt(self.hd_dim)
            self._feedback[name] = B_l.half() 

            module = dict(net.named_modules())[name]
            self._hooks.append(module.register_forward_hook(self._make_tensor_injection_hook(name)))

    def _make_tensor_injection_hook(self, name: str):
        def forward_hook(module, inp, out):
            if not out.requires_grad:
                return
            
            def tensor_backward_hook(grad):
                if name not in self._delta:
                    return grad
                return grad + self._delta[name].to(grad.dtype) 
            
            out.register_hook(tensor_backward_hook)
        return forward_hook

    def _compute_dfa_deltas(self, feats: torch.Tensor, proj_labels: torch.Tensor):
        """
        Compute δ_l = (e · B_l) for each DFA layer l, where
        e = hv_soft - proto_{y_i}  is the HD-space error.

        Stores results in self._delta[name] for the backward hook to pick up.
        Memory: uses sep_max_pixels subsampling before projecting to HD space.
        """
        B, C_f, H, W = feats.shape
        feats_flat   = feats.detach().permute(0, 2, 3, 1).reshape(-1, C_f)
        labels_flat  = proj_labels.reshape(-1)

        valid_mask = labels_flat != self.ignore_index
        valid_idx  = valid_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            self._delta.clear()
            return

        if valid_idx.numel() > self.sep_max_pixels:
            perm = torch.randperm(valid_idx.numel(), device=self.device)
            valid_idx = valid_idx[perm[:self.sep_max_pixels]]

        feats_sub  = feats_flat[valid_idx]
        labels_sub = labels_flat[valid_idx]

        with torch.no_grad():
            s = feats_sub @ self.rp_weight.t()
            proto_true = self.class_protos[labels_sub]
            e = F.normalize(s, dim=1) - proto_true

        for name, B_l in self._feedback.items():
            B_l_f = B_l.float()
            delta_flat = e @ B_l_f
            C_l = delta_flat.shape[1]

            delta_full = torch.zeros(B * H * W, C_l, device=self.device)
            delta_full[valid_idx] = delta_flat
            delta_spatial = (delta_full.reshape(B, H, W, C_l).permute(0, 3, 1, 2).contiguous())

            self._delta[name] = delta_spatial / (self.sep_max_pixels ** 0.5)

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        sep_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        net = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[DFA] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            with torch.no_grad():
                feats = net(in_vol, only_feat=True)

            self._compute_dfa_deltas(feats, proj_labels)

            with torch.enable_grad():
                sep_loss = self.sep_lambda * self._sep_loss_from_feats(feats, proj_labels)
            sep_loss_val = sep_loss.item()

            self.head_optimizer.zero_grad()
            self.backbone_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                seg_loss, pred = self._full_seg_loss(in_vol, proj_labels)

            seg_loss.backward()

            self.head_optimizer.step()
            self.backbone_optimizer.step()
            self.scheduler.step()

            self._delta.clear()

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            sep_m.update(sep_loss_val, N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, sep_m.avg

    def __del__(self):
        for h in self._hooks:
            h.remove()