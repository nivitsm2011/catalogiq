"""Multi-output attribute network: a shared MobileNetV3-Large backbone with one head per target.

Why MobileNetV3-Large: ~5.4M parameters and ~0.22 GFLOPs at 224x224 (about 20x cheaper than
ResNet-50), which is what makes CPU-only fine-tuning feasible. Compared with EfficientNet-B0 it
uses hardware-friendly operations (no squeeze-excite swish chains at every stage), so it is
typically faster per image on CPU at similar ImageNet accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from catalogiq.utils import configure_cache_dirs


class MultiHeadNet(nn.Module):
    """Shared backbone + per-target classification heads.

    Args:
        n_classes: Mapping target name -> number of classes.
        feature_dim: Backbone output channels (960 for MobileNetV3-Large).
        dropout: Dropout before every head.
        pretrained: Load ImageNet weights (downloads once into ``models/cache``).
        weights: torchvision weights enum name.
    """

    def __init__(
        self,
        n_classes: dict[str, int],
        feature_dim: int = 960,
        dropout: float = 0.2,
        pretrained: bool = True,
        weights: str = "IMAGENET1K_V2",
    ) -> None:
        super().__init__()
        configure_cache_dirs()
        w = models.MobileNet_V3_Large_Weights[weights] if pretrained else None
        base = models.mobilenet_v3_large(weights=w)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.targets = list(n_classes)
        self.heads = nn.ModuleDict(
            {
                t: nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, n))
                for t, n in n_classes.items()
            }
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features, shape (N, feature_dim)."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits per target for a normalised (N, 3, H, W) batch."""
        z = self.embed(x)
        return {t: head(z) for t, head in self.heads.items()}

    def set_backbone_trainable(self, from_block: int | None) -> None:
        """Freeze the whole backbone (``None``) or unfreeze ``features[from_block:]``."""
        for p in self.features.parameters():
            p.requires_grad = False
        if from_block is not None:
            for block in self.features[from_block:]:
                for p in block.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True) -> MultiHeadNet:
        """Train mode, but keep BatchNorm of frozen blocks in eval mode (stable statistics)."""
        super().train(mode)
        if mode:
            for module in self.features.modules():
                if isinstance(module, nn.BatchNorm2d) and not any(
                    p.requires_grad for p in module.parameters()
                ):
                    module.eval()
        return self


class ExportWrapper(nn.Module):
    """Takes a normalised batch, returns a tuple of softmax-free logits in ``targets`` order."""

    def __init__(self, net: MultiHeadNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Logits per target as a tuple (TorchScript-friendly)."""
        z = self.net.embed(x)
        return tuple(self.net.heads[t](z) for t in self.net.targets)
