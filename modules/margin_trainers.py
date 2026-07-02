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
from modules.losses import LovaszSoftmax, BoundaryLoss, CircleLoss, VICRegLoss
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
        self.steps_per_epoch = steps_per_epoch

        ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

        if model is not None:
            self.model = model.to(device)
        elif hasattr(self, 'model_override'):
            self.model = self.model_override
        else:
            self.model = ResNet34(num_classes, aux=aux_loss, use_mlp_proj=True, use_l2_norm=True).to(device)

        self.circle_loss = CircleLoss(m=0.25, gamma=256).to(device)
        self.decov = VICRegLoss().to(device)

        _proj_emb = embeddings.Projection(feat_dim, hd_dim)
        self.rp_weight = _proj_emb.weight.detach().to(device)
        

        self.criterion = nn.NLLLoss(weight=loss_weights.to(device)).to(device)
        self.ls = LovaszSoftmax(ignore=0).to(device)
        self.bd = BoundaryLoss().to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        warmup_steps = int(wup_epochs * steps_per_epoch)
        final_decay = lr_decay ** (1.0 / steps_per_epoch)
        self.scheduler = WarmupExpDecayLR(self.optimizer, lr, warmup_steps, final_decay)

        self.evaluator = iouEval(num_classes, device, ignore_idx)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        



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
        B, C_f, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C_f)
        pre_feats_flat = pre_norm_feats.permute(0, 2, 3, 1).reshape(-1, C_f)
        labels_flat = proj_labels.reshape(-1)

        valid_idx = (labels_flat != self.ignore_index).nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if valid_idx.numel() > self.margin_max_pixels:
            perm = torch.randperm(valid_idx.numel(), device=self.device)
            valid_idx = valid_idx[perm[:self.margin_max_pixels]]

        feats_sub = feats_flat[valid_idx]
        pre_feats_sub = pre_feats_flat[valid_idx]
        labels_sub = labels_flat[valid_idx]

        s = feats_sub @ self.rp_weight.t()
        
        if hard_quantize:
            q = ste_quantise(s)
        else:
            q = s

        q = q + 1e-8
        q_norm = F.normalize(q, dim=1)

        circle_loss = self.circle_loss(q_norm, labels_sub)
        decov_loss = self.decov(pre_feats_sub)

        return circle_loss + self.decov_weight * decov_loss

    def _eval_metrics(self, pred, proj_labels):
        self.evaluator.reset()
        self.evaluator.addBatch(pred.argmax(1), proj_labels)
        return self.evaluator.getacc(), self.evaluator.getIoU()[0]

    def _train_epoch(self, loader, epoch, total_epochs):
        raise NotImplementedError

class MeZOTrainer(_BaseHDTrainer):
    """
    Memory-Efficient Zeroth-Order (MeZO) Optimisation trainer.
    Replaces naive TeZO.
    """
    DEFAULT_HEAD_NAMES = ("conv_1", "conv_2", "semantic_output", "aux_head")

    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), zo_epsilon: float = 1e-3, zo_lr: float = 1e-5, head_param_names: Tuple[str, ...] = DEFAULT_HEAD_NAMES, use_pcgrad: bool = True, **base_kwargs):
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        self.zo_epsilon = zo_epsilon
        self.zo_lr = zo_lr
        self.head_param_names = head_param_names
        self.use_pcgrad = use_pcgrad

        os.makedirs(self.log_dir, exist_ok=True)
        self.stats_file = open(os.path.join(self.log_dir, "mezo_stats.csv"), "w")
        self.stats_file.write("step,g_scalar,loss_pos,loss_neg\n")
        self.global_step = 0

        head_params = [p for n, p in self.model.named_parameters() if any(h in n for h in self.head_param_names)]
        self._backbone_params = [
            (n, p) for n, p in self.model.named_parameters()
            if not any(h in n for h in self.head_param_names)
        ]

        self.head_optimizer = optim.SGD(
            head_params,
            lr=self.optimizer.param_groups[0]["lr"],
            momentum=self.optimizer.param_groups[0]["momentum"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        self.optimizer = self.head_optimizer
        self.scheduler = WarmupExpDecayLR(self.optimizer, self.optimizer.param_groups[0]["lr"], self.scheduler.warmup_steps, self.scheduler.decay_rate)

    @torch.no_grad()
    def _zo_loss(self, in_vol: torch.Tensor, proj_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return l_seg, l_margin

    @torch.no_grad()
    def _mezo_step(self, in_vol: torch.Tensor, proj_labels: torch.Tensor):
        eps = self.zo_epsilon
        was_training = self.model.training
        self.model.eval()

        seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Step 1: Forward +
        torch.manual_seed(seed)
        for _, p in self._backbone_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=eps)
            
        loss_pos_seg, loss_pos_margin = self._zo_loss(in_vol, proj_labels)

        # Step 2: Forward -
        torch.manual_seed(seed)
        for _, p in self._backbone_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=-2 * eps)
            
        loss_neg_seg, loss_neg_margin = self._zo_loss(in_vol, proj_labels)

        # Step 3: Compute scalar and update
        g_scalar_seg = (loss_pos_seg - loss_neg_seg) / (2 * eps)
        g_scalar_margin = (loss_pos_margin - loss_neg_margin) / (2 * eps)

        if self.use_pcgrad and (g_scalar_seg * g_scalar_margin < 0):
            g_scalar_margin = torch.tensor(0.0, device=self.device)

        g_scalar = g_scalar_seg + self.margin_lambda * g_scalar_margin
        g_scalar = torch.nan_to_num(g_scalar, nan=0.0, posinf=10.0, neginf=-10.0)

        lr = self.zo_lr
        wd = self.head_optimizer.param_groups[0]["weight_decay"]
        
        torch.manual_seed(seed)
        for _, p in self._backbone_params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=eps - lr * g_scalar.item())
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)

        if was_training:
            self.model.train()

        if hasattr(self, 'stats_file'):
            self.global_step += 1
            loss_pos = loss_pos_seg + self.margin_lambda * loss_pos_margin
            loss_neg = loss_neg_seg + self.margin_lambda * loss_neg_margin
            self.stats_file.write(f"{self.global_step},{g_scalar.item():.4f},{loss_pos.item():.4f},{loss_neg.item():.4f}\n")
            self.stats_file.flush()

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        margin_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[MeZO] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self._mezo_step(in_vol, proj_labels)

            # Freeze backbone to avoid storing activations and wasting VRAM
            for _, p in self._backbone_params:
                p.requires_grad = False

            self.head_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                out = net(in_vol, return_feat=True, return_pre_feat=True)
                if self.aux_loss:
                    pred, aux, feats, pre_feats = out
                    z2, z4, z8 = aux
                    lam = self.aux_lambda
                    seg_loss = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
                else:
                    pred, feats, pre_feats = out
                    seg_loss = self._seg_loss(pred, proj_labels)

            self.scaler.scale(seg_loss).backward()
            
            # Unfreeze backbone
            for _, p in self._backbone_params:
                p.requires_grad = True

            self.scaler.step(self.head_optimizer)
            self.scaler.update()
            self.scheduler.step()

            with torch.no_grad():
                margin_loss = self.margin_lambda * self._margin_loss_from_feats(feats.detach(), pre_feats.detach(), proj_labels)
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

class LocalGradTrainer(_BaseHDTrainer):
    """
    Gradient-Isolated Learning trainer (Local Exact Gradients) with PCGrad.
    Replaces DFA.
    """
    DEFAULT_LAYERS = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, num_classes: int, loss_weights: torch.Tensor, hd_dim: int, feat_dim: int = 128, log_dir: str = "logs", device: torch.device = torch.device("cpu"), local_layer_names: Tuple[str, ...] = DEFAULT_LAYERS, use_pcgrad: bool = True, **base_kwargs):
        base_kwargs.pop("num_epochs", None)
        super().__init__(num_classes, loss_weights, hd_dim, feat_dim, log_dir, device, **base_kwargs)
        self.local_layer_names = local_layer_names
        self.use_pcgrad = use_pcgrad

        os.makedirs(self.log_dir, exist_ok=True)
        self.stats_file = open(os.path.join(self.log_dir, "local_grad_stats.csv"), "w")
        self.stats_file.write("step,loss_layer1,loss_layer2,loss_layer3,loss_layer4\n")
        self.global_step = 0

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
            momentum=self.optimizer.param_groups[0]["momentum"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )
        self.scheduler = WarmupExpDecayLR(self.optimizer, self.optimizer.param_groups[0]["lr"], self.scheduler.warmup_steps, self.scheduler.decay_rate)

        self._local_rp: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._current_labels = None
        self._local_losses = {}

        self._register_local_hooks()

    def _register_local_hooks(self):
        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        dummy = torch.zeros(1, 5, 64, 512, device=self.device)
        shapes: Dict[str, torch.Size] = {}

        def _shape_hook(name):
            def hook(module, inp, out):
                shapes[name] = out.shape
            return hook

        tmp_hooks = []
        for name, module in net.named_modules():
            if name in self.local_layer_names:
                tmp_hooks.append(module.register_forward_hook(_shape_hook(name)))

        with torch.no_grad():
            net(dummy)

        for h in tmp_hooks:
            h.remove()

        for name, shape in shapes.items():
            C_l = shape[1] 
            B_l = torch.randn(self.hd_dim, C_l, device=self.device) / math.sqrt(self.hd_dim)
            self._local_rp[name] = B_l

            module = dict(net.named_modules())[name]
            self._hooks.append(module.register_forward_hook(self._make_local_grad_hook(name)))

    def _make_local_grad_hook(self, name: str):
        def forward_hook(module, inp, out):
            if not out.requires_grad or not self.model.training or self._current_labels is None:
                return out
                
            B, C_f, H, W = out.shape
            
            if self.use_pcgrad:
                out_target = out.detach().clone()
                out_target.requires_grad_(True)
            else:
                out_target = out
            
            feats_flat = out_target.permute(0, 2, 3, 1).reshape(-1, C_f)
            labels_flat = self._current_labels.reshape(-1)

            valid_mask = labels_flat != self.ignore_index
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            
            if valid_idx.numel() > 0:
                if valid_idx.numel() > self.margin_max_pixels:
                    perm = torch.randperm(valid_idx.numel(), device=self.device)
                    valid_idx = valid_idx[perm[:self.margin_max_pixels]]

                feats_sub = feats_flat[valid_idx].float()
                labels_sub = labels_flat[valid_idx]

                rp_weight = self._local_rp[name].to(feats_sub.dtype)
                s = feats_sub @ rp_weight.t()
                q_norm = F.normalize(s, dim=1)
                
                local_loss = self.circle_loss(q_norm, labels_sub)
                self._local_losses[name] = local_loss.item()
                
                scaled_loss = self.scaler.scale(local_loss) if hasattr(self, 'scaler') else local_loss
                
                if self.use_pcgrad:
                    scaled_loss.backward()
                    g_margin = out_target.grad
                    if g_margin is not None:
                        def tensor_backward_hook(g_seg):
                            g_m = g_margin.to(g_seg.dtype)
                            g_seg_flat = g_seg.reshape(-1)
                            g_m_flat = g_m.reshape(-1)
                            
                            dot_product = torch.dot(g_seg_flat, g_m_flat)
                            if dot_product < 0:
                                seg_norm_sq = torch.dot(g_seg_flat, g_seg_flat) + 1e-8
                                g_m = g_m - (dot_product / seg_norm_sq) * g_seg
                            return g_seg + g_m
                        out.register_hook(tensor_backward_hook)
                    return out
                else:
                    scaled_loss.backward()
                    return out.detach()

            return out if self.use_pcgrad else out.detach()
            
        return forward_hook

    def _train_epoch(self, loader, epoch, total_epochs):
        seg_m = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()

        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for in_vol, _, proj_labels, *_ in tqdm(loader, desc=f"[LocalGrad] Train {epoch+1}/{total_epochs}"):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()
            N = in_vol.size(0)

            self.optimizer.zero_grad()
            self._current_labels = proj_labels
            self._local_losses.clear()

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

            self.scaler.scale(seg_loss).backward()

            if hasattr(self, 'stats_file'):
                self.global_step += 1
                l1 = self._local_losses.get("layer1", 0.0)
                l2 = self._local_losses.get("layer2", 0.0)
                l3 = self._local_losses.get("layer3", 0.0)
                l4 = self._local_losses.get("layer4", 0.0)
                self.stats_file.write(f"{self.global_step},{l1:.4f},{l2:.4f},{l3:.4f},{l4:.4f}\n")
                self.stats_file.flush()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self._current_labels = None

            with torch.no_grad():
                acc, jac = self._eval_metrics(pred, proj_labels)

            seg_m.update(seg_loss.item(), N)
            acc_m.update(acc.item(), N)
            iou_m.update(jac.item(), N)

        return acc_m.avg, iou_m.avg, seg_m.avg, 0.0

    def __del__(self):
        if hasattr(self, 'stats_file') and not self.stats_file.closed:
            self.stats_file.close()
        for h in getattr(self, '_hooks', []):
            h.remove()
