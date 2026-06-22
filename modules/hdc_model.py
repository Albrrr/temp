import torch
import torch.nn as nn
import torch.nn.functional as F
from torchhd import embeddings, functional

from modules.resnet import get_model

class HDCModel(nn.Module):
    def __init__(self, num_classes: int, model_path: str, device: torch.device, hd_dim: int = 10000, model_type: str = "resnet34"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.hd_dim = hd_dim
        self.feat_dim = 128 

        self.net = get_model(model_type, num_classes, aux=True)

        ckpt = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(ckpt["state_dict"], strict=False)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.projection = embeddings.Projection(self.feat_dim, self.hd_dim)

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone(), requires_grad=False)
        self.to(device)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=True):
            feat = self.net(x, only_feat=True)
        feat = feat.permute(0, 2, 3, 1)
        return feat.reshape(-1, self.feat_dim)

    def _encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dtype != self.projection.weight.dtype:
            self.projection = self.projection.to(x_flat.dtype).to(self.device)
        hv = self.projection(x_flat)
        return functional.hard_quantize(hv)

    def _logits(self, hv: torch.Tensor) -> torch.Tensor:
        if hv.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(hv.dtype)
        return self.classify(F.normalize(hv))

    def encode(self, x: torch.Tensor):
        feats = self._extract_features(x)
        hv = self._encode(feats)
        return hv, feats

    def forward(self, x: torch.Tensor):
        hv, _ = self.encode(x)
        logits = self._logits(hv)
        return logits, F.normalize(hv)

    def get_predictions(self, hv: torch.Tensor) -> torch.Tensor:
        return self._logits(hv)

    def sync_class_weights(self):
        """Normalise accumulator → classify layer (call before inference)."""
        self.classify.weight.data[:] = F.normalize(self.classify_weights)