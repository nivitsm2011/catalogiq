"""Build blind contact sheets of 120 validation photos for labelling image defects.

60 photos are random; 60 are drawn (blind to the person labelling) from the extreme tails of each
measure so that rare defects appear. The sheets show no scores. Labels are written by hand to
``reports/phase6/image_labels.csv`` (columns: id, blurry, too_dark, overexposed,
cluttered_background,
text_or_watermark; 1 = defect present).
Usage: python scripts/make_quality_label_sheet.py
Outputs: reports/phase6/label_sheets/sheet_N.png, reports/phase6/label_sample.csv
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.quality.image_checks import compute_metrics

logger = get_logger("make_quality_label_sheet")
COLS, ROWS, SCALE = 6, 4, 3


def main() -> None:
    """Pick the sample and draw the numbered contact sheets."""
    cfg = get_quality_config()
    rng = np.random.default_rng(cfg["seed"])
    images = np.load(resolve_path(cfg["paths"]["image_cache"]), mmap_mode="r")
    index = pd.read_parquet(resolve_path(cfg["paths"]["image_index"]))
    val = np.where(index["split"].eq("val"))[0]
    metrics = pd.DataFrame([compute_metrics(np.asarray(images[i])) for i in val], index=val)
    random_pick = list(rng.choice(val, 60, replace=False))
    pool = metrics.drop(index=random_pick)
    tails = [
        pool.nsmallest(40, "blur").index,
        pool.nsmallest(40, "mean_luma").index,
        pool.nlargest(40, "p5_luma").index,
        pool.nlargest(40, "border_edge").index,
        pool.nlargest(40, "text_chars").index,
    ]
    tail_pick: list[int] = []
    for t in tails:
        tail_pick += [i for i in rng.permutation(t.to_numpy()) if i not in tail_pick][:12]
    chosen = list(rng.permutation(random_pick + tail_pick))[:120]
    sample = pd.DataFrame(
        {"n": range(1, len(chosen) + 1), "row": chosen, "id": index["id"].to_numpy()[chosen]}
    )
    sample["source"] = ["random" if r in random_pick else "tail" for r in chosen]
    out = resolve_path(cfg["paths"]["label_sheets_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(resolve_path(cfg["paths"]["eval_dir"]) / "label_sample.csv", index=False)
    per = COLS * ROWS
    for s in range(0, len(chosen), per):
        tile_h, tile_w = 80 * SCALE, 60 * SCALE
        sheet = np.full((ROWS * (tile_h + 6), COLS * (tile_w + 6), 3), 255, np.uint8)
        for k, row in enumerate(chosen[s : s + per]):
            tile = cv2.resize(
                np.asarray(images[row]), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST
            )
            y, x = (k // COLS) * (tile_h + 6), (k % COLS) * (tile_w + 6)
            sheet[y : y + tile_h, x : x + tile_w] = tile
            cv2.rectangle(sheet, (x, y), (x + 34, y + 18), (0, 0, 0), -1)
            cv2.putText(
                sheet,
                str(s + k + 1),
                (x + 3, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.imwrite(str(out / f"sheet_{s // per + 1}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    logger.info("wrote %d photos on %d sheets", len(chosen), -(-len(chosen) // per))


if __name__ == "__main__":
    main()
