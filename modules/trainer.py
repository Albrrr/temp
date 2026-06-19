import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from modules.resnet import ResNet34
from modules.losses import LovaszSoftmax, BoundaryLoss
from modules.ioueval import iouEval

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, path, suffix=""):
    torch.save(state, f"{path}/SENet{suffix}")


class WarmupExpDecayLR:
    """Exponential warmup followed by per-batch exponential decay."""

    def __init__(self, optimizer, base_lr: float, warmup_steps: int, step_decay: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.step_decay = step_decay
        self.start_lr = base_lr / self.warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            progress = self.step_count / self.warmup_steps
            lr = self.start_lr * (self.base_lr / self.start_lr) ** progress
        else:
            decay_steps = self.step_count - self.warmup_steps
            lr = self.base_lr * (self.step_decay ** decay_steps)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def build_warmup_exp_decay_scheduler(optimizer, lr: float, steps_per_epoch: int, wup_epochs: float = 1.0, lr_decay: float = 0.99):
    warmup_steps = int(wup_epochs * steps_per_epoch)
    step_decay = lr_decay ** (1.0 / steps_per_epoch)
    return WarmupExpDecayLR(optimizer, lr, warmup_steps, step_decay)


class CNNTrainer:
    def __init__(self, num_classes: int, loss_weights: torch.Tensor, log_dir: str, device: torch.device, lr: float = 1e-3, momentum: float = 0.9, w_decay: float = 5e-4, aux_loss: bool = True, aux_lambda: float = 0.4, model=None):
        self.num_classes = num_classes
        self.log_dir = log_dir
        self.device = device
        self.aux_loss = aux_loss
        self.aux_lambda = aux_lambda
        self.lr = lr

        if model is None:
            self.model = ResNet34(num_classes, aux=aux_loss)
        else:
            self.model = model
        self.model.to(device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

        self.criterion = nn.NLLLoss(weight=loss_weights.to(device)).to(device)
        self.ls = LovaszSoftmax(ignore=0).to(device)
        self.bd = BoundaryLoss().to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        self.evaluator = iouEval(num_classes, device, ignore_idx)

    def train(self, train_loader, num_epochs: int):
        scaler = GradScaler()
        for epoch in range(num_epochs):
            if epoch == 0:
                self._warmup_lr(train_loader)
            
            acc, iou, loss = self._train_epoch(train_loader, scaler, epoch)
            print(f"[Epoch {epoch+1}/{num_epochs}] loss={loss:.4f} acc={acc:.4f} iou={iou:.4f}")
            
            state = {
                "epoch": epoch,
                "state_dict": (self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(state, self.log_dir)

    def _warmup_lr(self, loader):
        """Linear warmup for the first epoch."""
        for i, _ in enumerate(loader):
            lr = self.lr * (i + 1) / len(loader)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            if i == 0: break

    def _train_epoch(self, loader, scaler, epoch):
        losses = AverageMeter()
        acc_m = AverageMeter()
        iou_m = AverageMeter()
        self.model.train()
        self.evaluator.reset()

        for i, (in_vol, _, proj_labels, *_) in enumerate(tqdm(loader, desc=f"Train epoch {epoch+1}")):
            in_vol = in_vol.to(self.device)
            proj_labels = proj_labels.to(self.device).long()

            if epoch == 0:
                lr = self.lr * (i + 1) / len(loader)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            with torch.cuda.amp.autocast():
                out = self.model(in_vol)
                if self.aux_loss:
                    pred, (z2, z4, z8) = out
                    lam = self.aux_lambda
                    loss = (self._seg_loss(pred, proj_labels) + lam * self._seg_loss(z2, proj_labels) + lam * self._seg_loss(z4, proj_labels) + lam * self._seg_loss(z8, proj_labels))
                else:
                    pred = out
                    loss = self._seg_loss(pred, proj_labels)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            with torch.no_grad():
                self.evaluator.addBatch(pred.argmax(1), proj_labels)

            losses.update(loss.item(), in_vol.size(0))

        accuracy = self.evaluator.getacc()
        jaccard, _ = self.evaluator.getIoU()
        return accuracy.item(), jaccard.item(), losses.avg

    def _seg_loss(self, pred, labels):
        bd_loss  = self.bd(pred, labels)
        nll_loss = self.criterion(torch.log(pred.clamp(min=1e-8)), labels)
        lov_loss = self.ls(pred, labels.long())
        return nll_loss + 1.5 * lov_loss + bd_loss

    def validate(self, val_loader):
        self.model.eval()
        self.evaluator.reset()
        with torch.no_grad():
            for in_vol, _, proj_labels, *_ in tqdm(val_loader, desc="Validate"):
                in_vol = in_vol.to(self.device)
                proj_labels = proj_labels.to(self.device).long()
                out = self.model(in_vol)
                pred = out[0] if isinstance(out, tuple) else out
                self.evaluator.addBatch(pred.argmax(1), proj_labels)

        acc = self.evaluator.getacc()
        iou, _ = self.evaluator.getIoU()
        print(f"[Validation] acc={acc:.4f} iou={iou:.4f}")
        return acc.item(), iou.item()
