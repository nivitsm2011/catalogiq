# Model card: catalog quality checker (Phase 6)

## What it does
For each listing (photo + stated colour, product type, gender, and other fields) it
1. measures the photo: blur, brightness, washed-out exposure, resolution, aspect ratio, background clutter, overlaid text or watermark;
2. checks that the photo matches the stated colour, product type and gender;
3. computes completeness against the attributes its category requires;
4. combines these into a 0-100 score with a plain-English fix for every flag.

Code: `src/catalogiq/quality/` (`image_checks`, `degrade`, `clip_checks`, `consistency`, `completeness`, `score`, `evaluate`), settings in
`configs/quality.yaml`, scripts `scripts/{make_quality_label_sheet,tune_quality_thresholds,evaluate_image_checks,evaluate_consistency,run_quality_report}.py`.

## Components
| Part | Choice |
|---|---|
| Blur / exposure / resolution / aspect / clutter | OpenCV measures, one threshold each, tuned on validation: highest recall subject to flagging at most 3% of clean photos |
| Text or watermark | CLIP ViT-B/32 zero-shot ("text overlaid" vs "clean product photo"); the blob heuristic is kept only as the baseline |
| Consistency | Phase 2 classifier (confidence >= 0.9 that a different value is shown) AND CLIP zero-shot (best alternative beats the stated value by >= 0.4) |
| Completeness | required attributes per category from `configs/rag.yaml` + product name |
| Score | 40% image + 35% consistency + 25% completeness; fixed penalties per flag (`quality.score`) |

## Intended use
A review queue for catalog and operations teams and a hint list for sellers ("retake the photo in better light"). **Not** for automatically rejecting listings.

## Evaluation
**Image checks** (test split: 500 clean photos + 8,500 degraded copies made by us; precision counts a flag from another defect type as wrong)
| Check | Recall | Precision | Clean photos flagged |
|---|---|---|---|
| Blurry | 0.999 | 0.42 | 2.4% |
| Too dark | 1.000 | 0.90 | 0.2% |
| Overexposed | 0.115 | 0.65 | 1.2% |
| Low resolution | 1.000 | 0.67 | 0.0% |
| Bad aspect ratio | 1.000 | 1.00 | 0.0% |
| Cluttered background | 0.452 | 0.75 | 1.0% |
| Text or watermark | 0.604 | 0.39 | 2.6% |

Low precision for blurry, text and low resolution mostly comes from flags on other defect types: washed-out and dark copies have low contrast (so also look "blurry"),
cluttered and dark copies look like text to CLIP, and narrow stretched copies really do have a short side under 48 px. Mild defects (level 0) are deliberately not counted as defects.

**Consistency** (300 injected mismatches per split: 100 colour, 100 product type, 100 gender; thresholds tuned on validation, reported on test)
| | Precision | Recall | F1 |
|---|---|---|---|
| Colour | 0.25 | 0.21 | 0.23 |
| Product type | 0.51 | 0.56 | 0.53 |
| Gender | 0.60 | 0.66 | 0.63 |
| Overall (classifier AND CLIP) | 0.47 | 0.48 | 0.47 |
| Classifier alone / CLIP alone / either | 0.53, 0.42, F1 0.47 / 0.20, 0.43, F1 0.27 / 0.30, 0.55, F1 0.39 | | |

Precision counts untouched listings as false alarms. A manual review of 24 such flags (AI reviewer, tiny thumbnails): 12 looked like real catalog errors, 12 were ambiguous boundaries, none was clearly correct.

**Completeness:** the test catalog is 100% complete; blanking one required attribute in 300 listings was found 300 of 300 times (a code test, not a measure of real sellers).
**Score:** clean test photos average 99.1; AUC 0.925 for separating real-defect copies from clean photos (image component only).

**Whole test catalog (6,339 listings):** 8.3% carry a flag, 4.1% score below 85, mean score 98.7. Most common flags: text 171, blurry 104, overexposed 84, cluttered 64; mismatches 162.

## Error analysis
- **Content, not quality, drives many flags.** Jewellery (146 listings, tiny items on a white canvas) has 46.6% flagged: 38% "overexposed" and 23% "blurry". Watches get the text flag 14% of the time (dials and logos). One global threshold per check does not fit all categories.
- **Overexposure is nearly undetectable** with a simple measure: a white product on a white background looks washed out. Recall 0.115 overall, 0.17 at the strongest level.
- **Blur depends on contrast**, so dark and washed-out photos also trigger it.
- **Colour is the weakest consistency attribute** (classifier 67.5% accurate on colour; shades blend between families; Phase 1 found 44% disagreement between twin photos).

## Limitations
- All photos are 60x80 studio shots; the gap to seller phone photos is unmeasured. The 48 px resolution rule suits this catalog; a live marketplace should use 500 px or more.
- Detection rates come from copies we made, which are more uniform than real defects. Only 2 of 120 natural validation photos had a visible defect.
- All labels (120 natural photos, the 24-listing review) are the AI assistant's, not a person's. The injected mismatches are exact.
- Score weights and penalties are judgement, not fitted to returns or sales (no such data).
- Thresholds were tuned on the validation split and the 3% false-alarm cap is a policy choice.

## Next steps
Per-category thresholds; contrast-normalised blur; a sample of real seller photos; owner-made labels; surface scores and fixes in the Streamlit app and the API.
