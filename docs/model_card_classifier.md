# Model card: product attribute classifier (Snap-to-Listing)

All numbers were computed on the held-out test split (6,339 photos) unless stated otherwise. Source tables:
`reports/model_vs_baselines.csv`, `reports/classifier_metrics.csv`, `reports/calibration.csv`,
`reports/latency.json`; narrative and charts in `notebooks/02_attribute_classifier.ipynb`.

## Summary
One MobileNetV3-Large backbone (ImageNet pre-trained) with seven linear heads predicts, from one product
photo: masterCategory, subCategory, articleType, baseColour, gender, season, usage. Each prediction comes with
a calibrated confidence and a `needs_review` flag.

## Intended use
- Pre-fill the attribute form for a seller who uploads a product photo, asking for confirmation when unsure.
- Not intended for: automatic publishing without a human check, categories outside fashion, photos with
  several products, or decisions about people.

## Data
Kaggle *Fashion Product Images (Small)*, cleaned in Phase 1 (`docs/data_card.md`): 42,257 photos, split 70/15/15
by near-duplicate group (29,579 / 6,339 / 6,339), no group spans two splits. Class lists are fitted on the
train split only: 4 masterCategory, 25 subCategory, 55 articleType, 15 baseColour, 5 gender, 4 season, 6 usage.

## Model and training
| | |
|---|---|
| Backbone | torchvision MobileNetV3-Large, ImageNet weights `IMAGENET1K_V2`, 960-d pooled features |
| Heads | one `Dropout(0.2) + Linear` per target |
| Input | photo resized to 80x60 (cache), then to 128x96, ImageNet normalisation |
| Loss | sum over targets of class-weighted cross-entropy; weight ~ (1/class count)^0.5; missing labels ignored |
| Augmentation | horizontal flip, rotation +-10 deg, scale 0.9-1.1, translation 5%, brightness/contrast/saturation +-20%; **hue shift disabled** (it would corrupt colour labels) |
| Stage 1 | heads only, AdamW lr 1e-3, 4 epochs (about 189 s/epoch) |
| Stage 2 | unfreeze `features[13:]`, AdamW lr 1e-4 backbone / 3e-4 heads, cosine schedule, 6 epochs (about 291 s/epoch) |
| Selection | best epoch by mean validation macro-F1 (fine-tune epoch 5, 0.7713); early stopping patience 3 |
| Seed | 42; logged in MLflow (local file store `mlruns/`) |
| Compute | CPU only (x64 Python emulated on a Windows ARM64 laptop); about 42 min total |

The first training run was interrupted during fine-tuning (process killed, cause not logged). Fine-tuning was
re-run from the saved heads-only checkpoint (`--skip-stage1`); all reported numbers come from the resumed run.
The planned 6 + 10 epochs was reduced to 4 + 6 to keep CPU time under an hour (decision by the project owner).

## Results (test split)
| Target | Classes | Accuracy | Macro-F1 | Weighted-F1 | Top-3 acc | Majority acc | Logreg macro-F1 | CLIP zero-shot macro-F1 |
|---|---|---|---|---|---|---|---|---|
| masterCategory | 4 | 0.9915 | 0.9899 | 0.9915 | 0.9994 | 0.490 | 0.976 | 0.883 |
| subCategory | 25 | 0.9590 | 0.9216 | 0.9601 | 0.9950 | 0.362 | 0.896 | 0.668 |
| articleType | 55 | 0.8566 | 0.8060 | 0.8545 | 0.9751 | 0.167 | 0.775 | 0.683 |
| baseColour | 15 | 0.6750 | 0.5821 | 0.6761 | 0.9047 | 0.229 | 0.469 | 0.561 |
| gender | 5 | 0.8781 | 0.7474 | 0.8806 | 0.9926 | 0.507 | 0.661 | 0.570 |
| season | 4 | 0.6990 | 0.7141 | 0.6976 | 0.9834 | 0.489 | 0.679 | 0.232 |
| usage | 6 | 0.8763 | 0.6391 | 0.8800 | 0.9972 | 0.780 | 0.622 | 0.387 |

Mean macro-F1 over the 7 targets: fine-tuned model 0.771, frozen-embedding logistic regression 0.725,
zero-shot CLIP 0.569, majority class 0.094. Baselines: majority class; frozen MobileNetV3 embeddings +
logistic regression (C=1.0); zero-shot CLIP ViT-B/32 (laion2b) with one prompt template per target
(`configs/vision.yaml`, templates not tuned).
Per-class precision/recall tables: `reports/per_class_<target>.csv`.

Weakest product types (classes with >= 30 test photos): Flats (F1 0.218), Tunics (0.222), Kurtis (0.347),
Sweaters (0.458), Jackets (0.500), Sweatshirts (0.533). Best: sunglasses, watches, bras, lipstick, ties,
socks, belts, sarees (F1 >= 0.977). 15 of 55 product types have fewer than 30 test photos, so their scores are
noisy.

## Calibration and thresholds
Temperature scaling fitted per target on the validation split; threshold = lowest confidence at which
accepted validation predictions are at least 90% correct (`vision.calibration.target_precision`).

| Target | Temp. | ECE before | ECE after | Threshold | Test coverage | Test precision of accepted |
|---|---|---|---|---|---|---|
| masterCategory | 0.914 | 0.0018 | 0.0026 | 0.341 | 1.000 | 0.9915 |
| subCategory | 0.887 | 0.0132 | 0.0074 | 0.187 | 1.000 | 0.9590 |
| articleType | 0.905 | 0.0158 | 0.0139 | 0.495 | 0.929 | 0.8904 |
| baseColour | 0.995 | 0.0170 | 0.0160 | 0.765 | 0.398 | 0.8848 |
| gender | 0.950 | 0.0104 | 0.0119 | 0.490 | 0.966 | 0.8940 |
| season | 0.976 | 0.0239 | 0.0203 | 0.791 | 0.291 | 0.9030 |
| usage | 0.913 | 0.0211 | 0.0178 | 0.601 | 0.922 | 0.9008 |

Thresholds tuned on validation photos land slightly under the 90% target on test for articleType, baseColour
and gender (88.5-89.4%). Precision vs coverage is a product decision: a higher bar means fewer wrong
auto-fills but more confirmation clicks.

Recommended UI behaviour (rule: auto-fill if coverage >= 90% and macro-F1 >= 0.70): auto-fill with low-confidence
flag for masterCategory, subCategory, articleType and gender; suggest-and-confirm for baseColour, season and usage.

## Error analysis (articleType, 909 of 6,339 test photos wrong)
Manual review by one reviewer (the AI assistant) of 24 most confident errors and 40 random errors
(`reports/error_review.csv`); indicative only.
- 24 most confident: 10 wrong catalog label (model right), 13 ambiguous class boundary, 1 corrupted image.
- 40 random: 4 wrong label (10%), 27 ambiguous boundary (67.5%), 9 model error (22.5%).
- Extrapolated to all errors: about 91 plain label errors (1.4% of the test set), 614 boundary cases (9.7%),
  205 real model errors (3.2%). Counting plain label errors as correct gives roughly 87.1% accuracy.
- No clear multi-item-photo failures were seen in the reviewed samples.
Largest confusions: sports shoes -> casual shoes 16%, casual -> sports shoes 12%, tops -> T-shirts 14%,
sandals -> flip flops 9%.

## Explainability
Grad-CAM on `features[12]` (8x6 map for a 128x96 input), `scripts/gradcam_figure.py`. It highlights the product
for correct predictions but part of the T-shirt example's attention falls on the model's face and background.
Coarse resolution; use as a qualitative aid.

## Inference
`catalogiq.vision.predict.predict(image) -> {attribute: (label, confidence, needs_review)}`; TorchScript export
`models/attribute_classifier/model_ts.pt` (12.9 MB; output identical to the eager model, max difference 0.0).
CPU latency per image including preprocessing: mean 18.7 ms, median 18.6 ms, p95 22.2 ms (100 runs, 8 threads, emulated
x64 Python on ARM64). Model files are in `models/` (gitignored); regenerate with
`scripts/train_classifier.py`, `scripts/evaluate_classifier.py`, `scripts/export_model.py`.

## Limitations and risks
- **Domain gap:** trained on clean studio photos from one retailer. Small-seller phone photos with clutter,
  poor light or several items will likely perform worse. This is unmeasured; the quality checker (planned) and
  augmentation do not remove the risk.
- **Label noise and fuzzy taxonomy:** about 10% of sampled errors are plain wrong labels and two-thirds sit on
  ambiguous boundaries, so the reported accuracy understates agreement with a clearer taxonomy, and the
  test labels themselves contain errors.
- **Colour:** 15-colour palette; look-alike photo groups disagree on colour 44% of the time in the source labels.
- **Season and usage** are not reliably visible in a photo; usage adds only 9.6 accuracy points over always
  guessing "Casual" (macro-F1 0.639).
- **Rare product types:** 88 article types were excluded for having fewer than 100 images and cannot be predicted.
- **Small input:** 60x80 px source photos hide fabric and detail; Grad-CAM resolution is coarse.
- **Demographics/market:** Indian retailer catalogue, mostly men's and women's fashion; kids and other markets
  are under-represented. Gender here is a product-audience label, not a statement about people.
- Single seed and single train/val/test split; no confidence intervals were computed for the test metrics.
