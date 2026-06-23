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
from modules.losses import LovaszSoftmax, BoundaryLoss, ArcFaceLoss, DeCovLoss
from modules.ioueval import iouEval
from modules.trainer import AverageMeter, WarmupExpDecayLR, save_checkpoint

class STEQuantise(torch.autograd.Function):
    """
    Straight-Through Estimator for hard_quantize (sign function).
    Forward : q = sign(x) (±1)
    Backward: dL/dx = dL/dq (identity pass-through)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

def ste_quantise(x: torch.Tensor) -> torch.Tensor:
    return STEQuantise.apply(x)

def hd_margin_loss(feats_flat: torch.Tensor, proj_weight: torch.Tensor, class_protos: torch.Tensor, labels_flat: torch.Tensor, ignore_index: int = 0, temperature: float = 0.1, max_pixels: int = 4096, hard_quantize: bool = True, margin_criterion=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-pixel InfoNCE/ArcFace contrastive loss in HD space.
    """
    valid_mask = labels_flat != ignore_index
    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return feats_flat.new_zeros(1).squeeze(), valid_idx

    if valid_idx.numel() > max_pixels:
        perm = torch.randperm(valid_idx.numel(), device=feats_flat.device)
        valid_idx = valid_idx[perm[:max_pixels]]

    feats_sub = feats_flat[valid_idx]
    labels_sub = labels_flat[valid_idx]

    s = feats_sub @ proj_weight.t()

    # TODO: We currently compute the margin loss (ArcFace) using the unquantized, 
    # continuous hypervectors (when hard_quantize=False). We need to eventually 
    # add a penalty term or regularizer that incentivizes the features to have a 
    # close quantized representation (i.e. pushing values toward ±1).
    if hard_quantize:
        q = ste_quantise(s)
    else:
        q = s

    q = q + 1e-8 # Prevent dividing by zero
    q_norm = F.normalize(q, dim=1)
    sims = q_norm @ class_protos.t()
    
    if margin_criterion is not None:
        if isinstance(margin_criterion, ArcFaceLoss):
            loss = margin_criterion(sims, labels_sub)
        else:
            sims = sims / temperature
            loss = margin_criterion(sims, labels_sub)
    else:
        sims = sims / temperature
        loss = F.cross_entropy(sims, labels_sub)

    return loss, valid_idx

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
        model: nn.Module = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        w_decay: float = 1e-4,
        wup_epochs: float = 1.0,
        lr_decay: float = 0.99,
        steps_per_epoch: int = 1,
        aux_loss: bool = True,
        aux_lambda: float = 0.4,
        margin_lambda: float = 0.5,
        margin_temperature: float = 0.1,
        margin_max_pixels: int = 4096,
        ignore_index: int = 0,
        decov_weight: float = 0.1,
        use_arcface: bool = True,
    ):
        self.num_classes = num_classes
        self.hd_dim = hd_dim
        self.feat_dim = feat_dim
        self.log_dir = log_dir
        self.device = device
        self.aux_loss = aux_loss
        self.aux_lambda = aux_lambda
        self.margin_lambda = margin_lambda
        self.margin_temperature = margin_temperature
        self.margin_max_pixels = margin_max_pixels
        self.ignore_index = ignore_index
        self.decov_weight = decov_weight

        ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

        if model is not None:
            self.model = model.to(device)
        elif hasattr(self, 'model_override'):
            self.model = self.model_override
        else:
            self.model = ResNet34(num_classes, aux=aux_loss, use_mlp_proj=True, use_l2_norm=True).to(device)

        self.arcface = ArcFaceLoss().to(device) if use_arcface else None
        self.decov = DeCovLoss().to(device)

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
        Call this before each epoch so the margin loss uses current protos.
        protos: (C, hd_dim) normalised float tensor.
        """
        self.class_protos = protos.detach().to(self.device)

    def train(self, train_loader, num_epochs: int):
        seg_losses = []
        margin_losses = []
        for epoch in range(num_epochs):
            acc, iou, l_seg, l_margin = self._train_epoch(
                train_loader, epoch, num_epochs)
            seg_losses.append(l_seg)
            margin_losses.append(l_margin)
            print(f"[Epoch {epoch+1}/{num_epochs}]  seg={l_seg:.4f}  sep={l_margin:.4f}  acc={acc:.4f}  iou={iou:.4f}")
            net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            save_checkpoint({"epoch": epoch, "state_dict": net.state_dict(), "optimizer": self.optimizer.state_dict()},self.log_dir)

        os.makedirs("logs/graphs", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), seg_losses, marker='o', linestyle='-', color='b', label='Segmentation Loss')
        plt.plot(range(1, num_epochs + 1), margin_losses, marker='x', linestyle='--', color='r', label='Margin Loss')
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
        pred_safe = torch.nan_to_num(pred, nan=1e-8, posinf=1.0, neginf=1e-8).clamp(min=1e-8, max=1.0)
        return (self.criterion(torch.log(pred_safe), labels) + 1.5 * self.ls(pred_safe, labels.long()) + self.bd(pred_safe, labels))

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

    def _margin_loss_from_feats(self, feats: torch.Tensor, pre_norm_feats: torch.Tensor, proj_labels: torch.Tensor, hard_quantize: bool = False) -> torch.Tensor:
        """Project features via RP, apply STE, compute ArcFace and DeCov margin loss."""
        B, C_f, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C_f)
        pre_feats_flat = pre_norm_feats.permute(0, 2, 3, 1).reshape(-1, C_f)
        labels_flat = proj_labels.reshape(-1)
        
        margin_loss, valid_idx = hd_margin_loss(
            feats_flat, self.rp_weight, self.class_protos,
            labels_flat, self.ignore_index,
            self.margin_temperature, self.margin_max_pixels,
            hard_quantize=hard_quantize,
            margin_criterion=self.arcface
        )
        
        if valid_idx.numel() > 0:
            decov_loss = self.decov(pre_feats_flat[valid_idx])
            loss = margin_loss + self.decov_weight * decov_loss
        else:
            loss = margin_loss
            
        return loss

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

    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), zo_epsilon: float = 1e-3, zo_ema_beta: float = 0.99, zo_n_samples: int = 1, zo_lr: float = 1e-5, head_param_names: Tuple[str, ...] = DEFAULT_HEAD_NAMES, **base_kwargs):
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        self.zo_epsilon = zo_epsilon
        self.zo_ema_beta = zo_ema_beta
        self.zo_n_samples = zo_n_samples
        self.zo_lr = zo_lr
        self.head_param_names = head_param_names

        os.makedirs(self.log_dir, exist_ok=True)
        self.stats_file = open(os.path.join(self.log_dir, "tezo_stats.csv"), "w")
        self.stats_file.write("step,g_scalar,accum_norm,ema_grad_norm,loss_pos,loss_neg\n")
        self.global_step = 0

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
        self.scheduler = WarmupExpDecayLR(self.optimizer, self.optimizer.param_groups[0]["lr"], self.scheduler.warmup_steps, self.scheduler.decay_rate)
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

        out = net(in_vol, return_feat=True, return_pre_feat=True)
        
        if self.aux_loss:
            pred, aux, feats, pre_feats = out
            z2, z4, z8 = aux
            lam = self.aux_lambda
            l_seg = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
        else:
            pred, feats, pre_feats = out
            l_seg = self._seg_loss(pred, proj_labels)

        l_margin = self._margin_loss_from_feats(feats, pre_feats, proj_labels)

        return l_seg + (self.margin_lambda * l_margin)

    @torch.no_grad()
    def _zo_step(self, in_vol: torch.Tensor, proj_labels: torch.Tensor):
        """
        Estimate ZO gradient for backbone params, EMA-smooth, apply update.
        """
        eps = self.zo_epsilon
        accumulated = {n: torch.zeros_like(p) for n, p in self._backbone_params}

        was_training = self.model.training
        self.model.eval()

        for _ in range(self.zo_n_samples):
            zs = {n: torch.randn_like(p) for n, p in self._backbone_params}

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=eps)
            with torch.amp.autocast('cuda'):
                loss_pos = self._zo_loss(in_vol, proj_labels)

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=-2 * eps)
            loss_neg = self._zo_loss(in_vol, proj_labels)

            for n, p in self._backbone_params:
                p.data.add_(zs[n], alpha=eps)

            g_scalar = (loss_pos - loss_neg) / (2 * eps)

            # Divide by D for regularization (MeZO) to prevent massive variance scaling
            D = sum(p.numel() for _, p in self._backbone_params)
            g_scalar = g_scalar / D

            g_scalar = torch.nan_to_num(g_scalar, nan=0.0, posinf=1000.0, neginf=-1000.0)

            for n, _ in self._backbone_params:
                accumulated[n].add_(zs[n], alpha=g_scalar.item())

        if was_training:
            self.model.train()

        for n in accumulated:
            accumulated[n].div_(self.zo_n_samples)

        accum_norm_val = sum(accumulated[n].norm().item()**2 for n in accumulated)**0.5 # Clip accumulated ZO gradients
        if accum_norm_val > 100.0:
            clip_coef = 100.0 / (accum_norm_val + 1e-6)
            for n in accumulated:
                accumulated[n].mul_(clip_coef)

        β = self.zo_ema_beta
        for n, p in self._backbone_params:
            self._ema_grads[n].mul_(β).add_(accumulated[n], alpha=1 - β)

        accum_norm = sum(accumulated[n].norm().item()**2 for n in accumulated)**0.5
        ema_norm = sum(self._ema_grads[n].norm().item()**2 for n in self._ema_grads)**0.5
        if hasattr(self, 'stats_file'):
            self.global_step += 1
            self.stats_file.write(f"{self.global_step},{g_scalar.item():.4f},{accum_norm:.4f},{ema_norm:.4f},{loss_pos.item():.4f},{loss_neg.item():.4f}\n")
            self.stats_file.flush()

        lr = self.zo_lr
        wd = self.head_optimizer.param_groups[0]["weight_decay"]
        for n, p in self._backbone_params:
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)
            p.data.add_(self._ema_grads[n], alpha=-lr)

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        margin_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[TeZO] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self._zo_step(in_vol, proj_labels)

            self.head_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                seg_loss, pred = self._full_seg_loss(in_vol, proj_labels)

            self.scaler.scale(seg_loss).backward()
            self.scaler.step(self.head_optimizer)
            self.scaler.update()
            self.scheduler.step()

            with torch.no_grad():
                net = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                feats, pre_feats = net(in_vol, only_feat=True, return_pre_feat=True)
                feats = feats.detach()
                pre_feats = pre_feats.detach()

            with torch.no_grad():
                margin_loss = self.margin_lambda * self._margin_loss_from_feats(feats, pre_feats, proj_labels)
            margin_loss_val = margin_loss.item()

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            margin_m.update(margin_loss_val, N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, margin_m.avg

    def __del__(self):
        if hasattr(self, 'stats_file') and not self.stats_file.closed:
            self.stats_file.close()

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

        os.makedirs(self.log_dir, exist_ok=True)
        self.stats_file = open(os.path.join(self.log_dir, "dfa_stats.csv"), "w")
        self.stats_file.write("step,delta_norm,grad_norm_backbone,grad_norm_head\n")
        self.global_step = 0

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
        self.scheduler = WarmupExpDecayLR(self.optimizer, self.optimizer.param_groups[0]["lr"], self.scheduler.warmup_steps, self.scheduler.decay_rate)
        self.backbone_scheduler = WarmupExpDecayLR(
            self.backbone_optimizer,
            self.dfa_lr,
            self.scheduler.warmup_steps,
            self.scheduler.decay_rate
        )

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
                delta = self._delta[name].to(grad.dtype)
                if delta.shape[2:] != grad.shape[2:]:
                    delta = F.interpolate(delta, size=grad.shape[2:], mode='bilinear', align_corners=False)
                scale = self.scaler.get_scale() if hasattr(self, 'scaler') else 1.0
                return grad + (delta.to(grad.dtype) * scale)
            
            out.register_hook(tensor_backward_hook)
        return forward_hook

    def _compute_dfa_deltas(self, feats: torch.Tensor, proj_labels: torch.Tensor):
        """
        Compute δ_l = (e · B_l) for each DFA layer l, where
        e = hv_soft - proto_{y_i}  is the HD-space error.

        Stores results in self._delta[name] for the backward hook to pick up.
        Memory: uses margin_max_pixels subsampling before projecting to HD space.
        """
        B, C_f, H, W = feats.shape
        feats_flat   = feats.detach().permute(0, 2, 3, 1).reshape(-1, C_f)
        labels_flat  = proj_labels.reshape(-1)

        valid_mask = labels_flat != self.ignore_index
        valid_idx  = valid_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            self._delta.clear()
            return

        if valid_idx.numel() > self.margin_max_pixels:
            perm = torch.randperm(valid_idx.numel(), device=self.device)
            valid_idx = valid_idx[perm[:self.margin_max_pixels]]

        feats_sub  = feats_flat[valid_idx]
        labels_sub = labels_flat[valid_idx]

        with torch.no_grad():
            s = feats_sub @ self.rp_weight.t()
            proto_true = self.class_protos[labels_sub]
            e = F.normalize(s, dim=1) - proto_true

        for name, B_l in self._feedback.items():
            delta_flat = e @ B_l.float()
            C_l = delta_flat.shape[1]

            delta_full = torch.zeros(B * H * W, C_l, device=self.device, dtype=delta_flat.dtype)
            delta_full[valid_idx] = delta_flat
            delta_spatial = (delta_full.reshape(B, H, W, C_l).permute(0, 3, 1, 2).contiguous())

            self._delta[name] = delta_spatial / max(1, valid_idx.numel())

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        margin_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        net = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[DFA] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self.head_optimizer.zero_grad()
            self.backbone_optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = net(in_vol, return_feat=True, return_pre_feat=True)
                if self.aux_loss:
                    pred, aux, feats, pre_feats = out
                    z2, z4, z8 = aux
                    lam = self.aux_lambda
                    seg_loss = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
                else:
                    pred, feats, pre_feats = out
                    seg_loss = self._seg_loss(pred, proj_labels)

                margin_loss = self.margin_lambda * self._margin_loss_from_feats(feats, pre_feats, proj_labels)
                margin_loss_val = margin_loss.item()

                self._compute_dfa_deltas(feats.detach(), proj_labels)

            self.scaler.scale(seg_loss).backward()

            self.scaler.unscale_(self.head_optimizer)
            self.scaler.unscale_(self.backbone_optimizer)

            delta_norm = sum(d.norm().item()**2 for d in self._delta.values())**0.5
            grad_norm_backbone = sum(p.grad.norm().item()**2 for p in self.backbone_optimizer.param_groups[0]['params'] if p.grad is not None)**0.5
            grad_norm_head = sum(p.grad.norm().item()**2 for p in self.head_optimizer.param_groups[0]['params'] if p.grad is not None)**0.5

            if hasattr(self, 'stats_file'):
                self.global_step += 1
                self.stats_file.write(f"{self.global_step},{delta_norm:.4f},{grad_norm_backbone:.4f},{grad_norm_head:.4f}\n")
                self.stats_file.flush()

            self.scaler.step(self.head_optimizer)
            self.scaler.step(self.backbone_optimizer)
            self.scaler.update()

            self.scheduler.step()
            self.backbone_scheduler.step()

            self._delta.clear()

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            margin_m.update(margin_loss_val, N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, margin_m.avg

    def __del__(self):
        if hasattr(self, 'stats_file') and not self.stats_file.closed:
            self.stats_file.close()
        for h in getattr(self, '_hooks', []):
            h.remove()

class EndToEndHDTrainer(_BaseHDTrainer):
    """
    Trains the backbone and HD class prototypes jointly using exact backprop.
    """
    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), **base_kwargs):
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        
        self.class_protos = nn.Parameter(torch.randn(num_classes, hd_dim, device=device))
        
        self.optimizer = optim.SGD(
            [
                {"params": self.model.parameters()},
                {"params": self.class_protos, "lr": 0.05}
            ],
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.steps_per_epoch * 80)

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        margin_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[E2E] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = net(in_vol, return_feat=True, return_pre_feat=True)
                if self.aux_loss:
                    pred, aux, feats, pre_feats = out
                    z2, z4, z8 = aux
                    lam = self.aux_lambda
                    seg_loss = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
                else:
                    pred, feats, pre_feats = out
                    seg_loss = self._seg_loss(pred, proj_labels)

                margin_loss = self.margin_lambda * self._margin_loss_from_feats(feats, pre_feats, proj_labels)
                total_loss = seg_loss + margin_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            margin_m.update(margin_loss.item(), N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, margin_m.avg