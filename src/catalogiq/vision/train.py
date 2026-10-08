"""Two-stage training of the multi-output attribute network.

Stage 1 trains only the heads on top of the frozen pretrained backbone. Stage 2 unfreezes the last
backbone blocks and fine-tunes them with a smaller learning rate than the heads. Both stages use
class-weighted cross-entropy summed over targets, photo-safe augmentation, early stopping on the
validation macro-F1 (mean over targets) and checkpoint the best epoch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from catalogiq.logging_utils import get_logger
from catalogiq.vision.data import (
    IGNORE_INDEX,
    augment,
    class_weights,
    normalise,
    resize,
    to_tensor,
)
from catalogiq.vision.model import MultiHeadNet

logger = get_logger(__name__)


def make_batch(
    images_u8: np.ndarray,
    positions: np.ndarray,
    hw: tuple[int, int],
    aug_cfg: dict[str, Any] | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Cached uint8 images -> (augmented) normalised float batch at network input size."""
    x = to_tensor(images_u8[positions])
    if aug_cfg is not None and generator is not None:
        x = augment(x, aug_cfg, generator)
    return normalise(resize(x, hw))


@torch.no_grad()
def predict_logits(
    net: MultiHeadNet,
    images_u8: np.ndarray,
    positions: np.ndarray,
    hw: tuple[int, int],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Logits per target for the given image positions (eval mode, no augmentation)."""
    net.eval()
    out: dict[str, list[torch.Tensor]] = {t: [] for t in net.targets}
    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        logits = net(make_batch(images_u8, batch, hw, None, None))
        for t, v in logits.items():
            out[t].append(v)
    return {t: torch.cat(v) for t, v in out.items()}


def mean_macro_f1(logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict:
    """Macro-F1 per target and their mean."""
    scores = {}
    for t, lg in logits.items():
        y = labels[t]
        mask = y != IGNORE_INDEX
        scores[t] = float(
            f1_score(y[mask].numpy(), lg[mask].argmax(1).numpy(), average="macro", zero_division=0)
        )
    scores["mean"] = float(np.mean([v for k, v in scores.items()]))
    return scores


def _loss(
    out: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    smoothing: float,
) -> torch.Tensor:
    total = torch.zeros(())
    for t, logits in out.items():
        if (y[t] != IGNORE_INDEX).any():
            total = total + F.cross_entropy(
                logits,
                y[t],
                weight=weights[t],
                ignore_index=IGNORE_INDEX,
                label_smoothing=smoothing,
            )
    return total


def run_stage(
    net: MultiHeadNet,
    stage: str,
    images_u8: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    labels: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    seed: int,
    checkpoint: Path,
    epochs_override: int | None = None,
) -> list[dict[str, Any]]:
    """Train one stage with early stopping; leaves the best epoch's weights loaded in ``net``.

    Args:
        stage: ``"stage1"`` (heads only) or ``"stage2"`` (fine-tune last blocks).
        cfg: The ``vision`` config section.
        checkpoint: Where the best weights are saved.
        epochs_override: Use fewer epochs (speed trials).

    Returns:
        Per-epoch history (losses, validation macro-F1, seconds).
    """
    tcfg, hw = cfg["training"], tuple(cfg["input_hw"])
    scfg = tcfg[stage]
    epochs = epochs_override or scfg["epochs"]
    if stage == "stage1":
        net.set_backbone_trainable(None)
        groups = [{"params": net.heads.parameters(), "lr": scfg["lr"]}]
    else:
        net.set_backbone_trainable(cfg["backbone"]["unfreeze_from_block"])
        backbone_params = [p for p in net.features.parameters() if p.requires_grad]
        groups = [
            {"params": backbone_params, "lr": scfg["lr_backbone"]},
            {"params": net.heads.parameters(), "lr": scfg["lr_heads"]},
        ]
    opt = torch.optim.AdamW(groups, weight_decay=scfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    val_labels = {t: v[val_pos] for t, v in labels.items()}
    best, bad_epochs, history = -1.0, 0, []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        gen = torch.Generator().manual_seed(seed + epoch + (1000 if stage == "stage2" else 0))
        order = train_pos[torch.randperm(len(train_pos), generator=gen).numpy()]
        net.train()
        running, n_batches = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            pos = order[start : start + cfg["batch_size"]]
            if len(pos) < 2:
                continue
            x = make_batch(images_u8, pos, hw, cfg["augmentation"], gen)
            y = {t: v[pos] for t, v in labels.items()}
            if stage == "stage1":
                with torch.no_grad():
                    z = net.embed(x)
                out = {t: head(z) for t, head in net.heads.items()}
            else:
                out = net(x)
            loss = _loss(out, y, weights, tcfg["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        sched.step()
        val_f1 = mean_macro_f1(
            predict_logits(net, images_u8, val_pos, hw, cfg["batch_size"]), val_labels
        )
        seconds = time.perf_counter() - t0
        record = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_macro_f1": val_f1["mean"],
            "seconds": seconds,
            **{f"val_f1_{t}": v for t, v in val_f1.items() if t != "mean"},
        }
        history.append(record)
        step = epoch + (epochs if stage == "stage2" else 0)
        mlflow.log_metrics(
            {
                f"{stage}_train_loss": record["train_loss"],
                f"{stage}_val_macro_f1": record["val_macro_f1"],
                f"{stage}_seconds": seconds,
            },
            step=step,
        )
        logger.info(
            "%s epoch %d/%d | loss %.4f | val macro-F1 %.4f | %.0fs",
            stage,
            epoch,
            epochs,
            record["train_loss"],
            record["val_macro_f1"],
            seconds,
        )
        if val_f1["mean"] > best:
            best, bad_epochs = val_f1["mean"], 0
            torch.save(net.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                logger.info("early stopping: no improvement for %d epochs", bad_epochs)
                break
    net.load_state_dict(torch.load(checkpoint))
    return history


def build_class_weights(
    labels: dict[str, torch.Tensor], train_pos: np.ndarray, n_classes: dict[str, int], power: float
) -> dict[str, torch.Tensor]:
    """Class weights per target computed on the training rows only."""
    return {t: class_weights(labels[t][train_pos], n_classes[t], power) for t in labels}
