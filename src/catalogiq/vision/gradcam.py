"""Grad-CAM heatmaps: which image regions pushed the model toward a predicted class."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from catalogiq.vision.model import MultiHeadNet


def grad_cam(
    net: MultiHeadNet,
    x: torch.Tensor,
    target: str,
    class_index: int | None = None,
    layer_index: int = 12,
) -> tuple[np.ndarray, int]:
    """Grad-CAM for one normalised image ``x`` of shape (1, 3, H, W).

    Args:
        net: The trained network (switched to eval mode).
        x: Normalised input batch with a single image.
        target: Attribute whose head to explain (e.g. ``"articleType"``).
        class_index: Class to explain; defaults to the predicted class.
        layer_index: ``net.features`` block whose activations are used. Block 12 has stride 16
            (an 8x6 map for a 128x96 input); the final block (16) would give only 4x3 cells, too
            coarse for 60x80 px photos.

    Returns:
        Heatmap in [0, 1] resized to the input size, and the explained class index.
    """
    net.eval()
    store: dict[str, torch.Tensor] = {}
    layer = net.features[layer_index]
    handle = layer.register_forward_hook(lambda _m, _i, out: store.__setitem__("act", out))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = net(x)[target]
            idx = int(logits.argmax(1)) if class_index is None else class_index
            act = store["act"]
            grad = torch.autograd.grad(logits[0, idx], act)[0]
    finally:
        handle.remove()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)
    return cam.numpy(), idx
