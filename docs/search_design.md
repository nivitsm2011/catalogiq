# Visual search: design notes

Code: `src/catalogiq/search/`. Settings: `configs/search.yaml`. Numbers and charts: `notebooks/03_visual_search.ipynb`,
`reports/search_*.csv`, `reports/index_comparison.csv`, `reports/duplicate_threshold_eval.csv`.

## Embeddings
- Model: OpenCLIP ViT-B-32, checkpoint `laion2b_s34b_b79k` (605.2 MB file, 512-d embeddings), cached in `models/cache`.
- Catalog photos are about 60x80 px with white backgrounds. OpenCLIP's default preprocessing resizes the short side
  and centre-crops, which would cut off heads and shoes, so images are **padded to a square with white** and
  upsampled bicubically to 224x224 (done with torch ops on whole batches).
- Every image and an **attribute sentence** ("a product photo of a black shirts for men, casual style, summer
  season") are embedded once and cached (`data/processed/clip_*.npy`, id map `clip_ids.npy`). The free-text product
  name is deliberately not used. Missing attributes are dropped from the sentence.
- CPU speed measured on this laptop: about 5-7 images/s (batch size and thread count made no difference), so the full
  catalog needs about 2 hours; embedding is chunked and resumable.

## Index
Cosine similarity via L2-normalised vectors and inner product. `IndexFlatIP` (exact) is the default; HNSW
(`M=32`, `efConstruction=200`, `efSearch=64`) is available. Filtered queries restrict the candidate set first and
search it exactly, so a filter can never return fewer results than exist.

## Shop the look: rules
Given one item (catalog id, or a photo plus the Phase 2 attribute predictor) recommend items from **other roles**.

Roles (`article_roles` in the config): top (T-shirts, shirts, kurtas, tops, sweatshirts, sweaters, jackets, kurtis,
tunics), bottom (jeans, shorts, trousers, track pants, leggings, capris, skirts), dress (dresses, sarees), footwear
(all shoe types), bag (handbags, backpacks, clutches), accessory (watches, wallets, sunglasses, belts, jewellery, caps,
ties, scarves, dupatta, cufflinks). Underwear, beauty, perfume, socks and gift sets are never recommended.
Complements: top -> bottom, footwear, accessory, bag; bottom -> top, footwear, accessory, bag; dress -> footwear,
accessory, bag; footwear/bag/accessory -> the other roles.

**Hard rules** (a candidate failing one is never shown):
1. its role is a complement of the query's role;
2. gender-compatible: same gender or Unisex (an adult Unisex query accepts Men, Women and Unisex);
3. it shares a **style family** with the query (casual, formal, sporty, ethnic, party; `article_styles` in the config),
   so formal trousers are never paired with sports shoes;
4. its catalog articleType is **confirmed by the Phase 2 classifier** (agreement at confidence >= 0.5; 84.1% of items
   pass), so mislabelled photos never appear in a look;
5. it is not the query item, there is at most one item per near-duplicate photo group, and in the accessory and bag
   roles at most one item per article type.

**Score** = 0.30 x pairing + 0.30 x colour + 0.15 x usage + 0.10 x season + 0.15 x style (weights in the config; each
part in 0..1; hand-set, not tuned).
- pairing: 1 if the two article types share two or more style families, 0.7 for one (zero is excluded by hard rule 3).
- colour: 1 if either colour is a neutral (black, white, grey, beige, brown) or the pair is in a curated list of
  pleasing pairs (e.g. blue + white, red + black); 0.7 same colour (tonal); 0.3 if a multi-colour print is involved;
  0.5 unknown; else 0.2.
- usage: 1 same usage, 0.5 if either is unknown/"Other", else 0.  season: 1 same, 0.5 adjacent (Spring-Summer-Fall-Winter
  ring) or unknown, else 0.
- style: CLIP image similarity to the query, min-max scaled among the remaining candidates of that role, i.e. a
  *relative* fit (the best of the sensible options scores near 1).
Each recommendation carries a plain-language `reasons` string. History: the first version had no style-family or label
rules and produced poor outfits (flip-flops with shirts, mislabelled photos); see `docs/decisions.md`. The rules have
not been validated against stylists or conversion data, and usage/season carry little information because 77% of items
are Casual and 49% Summer.

## Duplicate detection
Candidates are each item's `candidate_k` nearest neighbours. A pair is flagged when CLIP cosine similarity >=
threshold and, optionally, the 64-bit perceptual hashes differ by at most `phash_max_distance` bits (the hash vetoes
look-alike but different products). The threshold is chosen on **100 hand-labelled pairs** sampled across similarity
bins (so clear and borderline cases are both present); each pair carries a sampling weight so precision and recall
estimate the whole candidate population.

Labelling guide: *duplicate* = the same product listed twice (same photo, or another photo of the identical item).
*Not a duplicate* = a different product, including the same style in another colour or size and merely similar
items. *Unsure* pairs are excluded from the evaluation.

## Evaluation protocol
Gallery = train + val photos (35,918); queries = test photos (Phase 1 split is group-aware, so no near-duplicate twin
of a query is in the gallery). Relevant = same articleType AND same baseColour. Metrics (`search/metrics.py`):
Recall@k = share of queries with at least one relevant result in the top k (hit rate); Precision@k; mAP@10 with
AP@10 normalised by min(10, number of relevant gallery items). Text queries are attribute combinations such as
"black shirts for men", relevant if the photo matches colour, type and gender. Baselines: the Phase 2 fine-tuned
classifier's 960-d features and frozen ImageNet MobileNetV3 features (note: the classifier was trained on these same
labels, so it is a supervised baseline against zero-shot CLIP), plus random vectors as a floor.
