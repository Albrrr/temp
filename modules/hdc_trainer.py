import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.hdc_model import HDCModel
from modules.ioueval import iouEval

class HDCTrainer:
    def __init__(self, model: HDCModel, num_classes: int, device: torch.device, retrain_epochs: int = 20):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.retrain_epochs = retrain_epochs

        self.is_wrong_list: list[torch.Tensor | None] = []

    def train(self, train_loader):
        print("HDC Phase 1: initial training pass")
        self.model.eval()
        self.is_wrong_list = [None] * len(train_loader)
        times = []

        with torch.no_grad():
            for i, (proj_in, _, proj_labels, *_) in enumerate(tqdm(train_loader, desc="HDC train")):
                proj_in = proj_in.to(self.device)
                proj_labels = proj_labels.view(-1).to(self.device)

                t0 = time.time()
                hv, _ = self.model.encode(proj_in)
                hv = hv.to(self.model.classify_weights.dtype)

                self.model.classify_weights.index_add_(0, proj_labels, hv)
                times.append(time.time() - t0)

                self.model.sync_class_weights()
                preds   = self.model.get_predictions(hv).argmax(dim=1)
                is_wrong = proj_labels != preds
                self.is_wrong_list[i] = is_wrong

            self.model.sync_class_weights()

        print(f"  wrong pixels: {sum(x.sum().item() for x in self.is_wrong_list if x is not None)}")
        print(f"  time  mean={np.mean(times):.3f}s  std={np.std(times):.3f}s")

    def retrain(self, train_loader, epoch: int):
        print(f"HDC Phase 2: retrain epoch {epoch}")
        self.model.eval()
        total_miss = 0
        times = []

        with torch.no_grad():
            for i, (proj_in, _, proj_labels, *_) in enumerate(tqdm(train_loader, desc=f"HDC retrain {epoch}")):
                proj_in = proj_in.to(self.device)
                proj_labels = proj_labels.view(-1).to(self.device)

                self.model.sync_class_weights()
                t0 = time.time()

                hv, _  = self.model.encode(proj_in)
                logits = self.model.get_predictions(hv)
                argmax = logits.argmax(dim=1)

                is_wrong = proj_labels != argmax
                if self.is_wrong_list[i] is not None:
                    is_wrong = is_wrong & self.is_wrong_list[i]

                if is_wrong.sum() == 0:
                    times.append(time.time() - t0)
                    continue

                total_miss += is_wrong.sum().item()
                w_labels = proj_labels[is_wrong]
                w_argmax = argmax[is_wrong]
                w_hv = hv[is_wrong].to(self.model.classify_weights.dtype)

                self.model.classify_weights.index_add_(0, w_labels, w_hv)
                # self.model.classify_weights.index_add_(0, w_labels, w_hv)
                self.model.classify_weights.index_add_(0, w_argmax, -w_hv)
                # self.model.classify_weights.index_add_(0, w_argmax, -w_hv)

                self.model.sync_class_weights()
                new_preds = self.model.get_predictions(hv).argmax(dim=1)
                self.is_wrong_list[i] = proj_labels != new_preds

                times.append(time.time() - t0)

        print(f"  corrected: {total_miss}")
        print(f"  wrong remaining: {sum(x.sum().item() for x in self.is_wrong_list if x is not None)}")
        print(f"  time  mean={np.mean(times):.3f}s  std={np.std(times):.3f}s")

    def validate(self, val_loader, evaluator: iouEval) -> float:
        self.model.eval()
        evaluator.reset()
        times = []

        with torch.no_grad():
            for proj_in, _, proj_labels, *_ in tqdm(val_loader, desc="HDC val"):
                B, _, H, W = proj_in.shape
                proj_in = proj_in.to(self.device)
                proj_labels = proj_labels.to(self.device)

                self.model.sync_class_weights()
                t0 = time.time()
                logits, _ = self.model(proj_in)
                times.append(time.time() - t0)

                preds = (logits.view(B, H, W, self.num_classes).permute(0, 3, 1, 2).argmax(dim=1))
                evaluator.addBatch(preds, proj_labels)

        acc = evaluator.getacc()
        iou, class_iou = evaluator.getIoU()
        print(f"[Validation] acc={acc:.4f}  mIoU={iou:.4f}")
        print(f"  per-class IoU: {class_iou.cpu().numpy()}")
        print(f"  time  mean={np.mean(times):.3f}s  std={np.std(times):.3f}s")
        return iou.item()

    def run(self, train_loader, val_loader, evaluator: iouEval):
        self.train(train_loader)
        self.validate(val_loader, evaluator)

        for epoch in range(1, self.retrain_epochs + 1):
            self.retrain(train_loader, epoch)
            self.validate(val_loader, evaluator)