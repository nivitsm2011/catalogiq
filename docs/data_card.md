# Data card

Numbers below were computed by `scripts/build_fashion.py` and `scripts/build_reviews.py`
(see `reports/fashion_audit.json`, `reports/fashion_cleaning_log.json`, `reports/target_evidence.csv`).

## 1. Fashion Product Images (Small)

| | |
|---|---|
| Source | Kaggle `paramaggarwal/fashion-product-images-small` (product catalog of an Indian fashion retailer) |
| Licence | MIT, as declared on the Kaggle dataset page. The photos are retailer product images, so check terms before any commercial use. |
| Size | 592.6 MB zip, 44,441 images (about 60x80 px) + `styles.csv` (44,446 rows). About 0.55 GB on disk. |
| Labels | gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName (entered manually by the retailer's catalog team) |
| Local path | `data/raw/fashion/` (gitignored). A duplicate `myntradataset/` folder inside the archive (byte-identical) was removed. |

### Cleaning (reasoning in `docs/decisions.md`)
| Step | Rows after |
|---|---|
| Parsed `styles.csv` (22 rows had commas in the product name and were repaired, 0 lost) | 44,446 |
| Duplicate ids | 44,446 (0 removed) |
| Missing image file (5) / unreadable or tiny image (0) | 44,441 |
| Missing core taxonomy | 44,441 (0 removed) |
| Rare masterCategory / articleType / gender classes dropped (repeated until stable) | 42,257 |
| Rare subCategory / baseColour / usage classes merged into "Other" | 42,257 |

After cleaning: 4 masterCategory, 25 subCategory, 55 articleType, 15 baseColour (from 47), 5 gender, 4 season and 6 usage classes.
Near-duplicate photos (pHash distance <= 1): 1,553 groups covering 3,932 images.
Splits (`data/processed/fashion_{train,val,test}.parquet`): 29,579 / 6,339 / 6,339 rows (70/15/15), stratified by articleType and group-aware, with 0 groups spanning splits.

### Known biases and limitations
- **Studio photography:** clean, single-item, mostly white-background shots, often on models. Small sellers' phone photos will look different, so expect lower real-world accuracy (to be measured).
- **Single retailer, one market:** an Indian fashion catalogue skewed to Men/Women apparel, footwear and accessories. T-shirts are 17% of the cleaned set; many categories are rare.
- **Class imbalance:** the biggest/smallest class ratio is 69x for articleType and 293x for usage; 78% of products are "Casual".
- **Label noise:** in groups of near-identical photos, labels disagree for colour 44% (partly colourways), season 18%, articleType 12%, usage 10%, gender 8%, subCategory 4% and masterCategory 0.5%.
- **Rare product types excluded:** 88 articleTypes (2,184 rows removed in total, including two rare masterCategories) are not learnable here.
- **Small thumbnails:** about 60x80 px limits fine detail (fabric, print), and the quality checker's blur thresholds will need calibration for this size.
- Season and year reflect the retailer's selling season, not a visible property.

## 2. Amazon Reviews 2023 - Amazon_Fashion (sample)

| | |
|---|---|
| Source | McAuley Lab, Hugging Face `McAuley-Lab/Amazon-Reviews-2023`, files `raw/review_categories/Amazon_Fashion.jsonl` (1.05 GB) and `raw/meta_categories/meta_Amazon_Fashion.jsonl` (1.42 GB), streamed, never stored |
| Licence | No licence field on the Hugging Face dataset card. Intended for research; cite Hou et al. (2024), "Bridging Language and Items for Retrieval and Recommendation". Review text belongs to its authors; do not redistribute. |
| Scanned | 2,500,939 reviews over 825,869 items |
| Sample | the 800 items with the most reviews (each at least 20) that have usable metadata; 139,400 cleaned reviews; 17.5 MB parquet (cap 300 MB) |
| Local path | `data/processed/amazon_reviews.parquet`, `data/processed/amazon_items.parquet` (gitignored) |

### Cleaning
HTML stripped; reviews under 30 characters dropped; identical review texts per item dropped; at most 300 reviews per item kept (seeded reservoir sample). Kept review fields: rating, title, text, helpful_vote, verified_purchase, timestamp. Kept item fields: title, features, description, average_rating, rating_number, price, store, categories, main_category.

### Listing quality score
`shrunk = v/(v+50)*R + 50/(v+50)*C` and `score = (shrunk-1)/4`, where R is the item's average rating, v its rating count and C the catalog mean rating. `is_exemplar` means score >= 0.85, at least 50 ratings, title >= 20 characters and at least 3 bullets: 191 of 800 items (23.9%). These are used later to pick well-performing real listings as RAG examples.

### Known biases and limitations
- **Popularity bias:** only the most-reviewed items are kept, so results describe established listings, not new small sellers.
- **Reviewer selection:** 59% of reviews are 5-star and 96% are verified purchases; about 17% are 1-2 star, so complaints are comparatively few.
- **Sparse listing fields:** 69% of items have no description, 39% no price and 20% no bullets, so complete exemplar listings are a minority.
- **Near-duplicate variants:** 5 items share a title with another (colour or size variants).
- **Per-item cap** of 300 reviews underrepresents the most popular items.
- **Keyword topic counts** do not capture sentiment or negation.
- **Language and region:** mostly English, US marketplace; style and sizing conventions differ from the Kaggle (Indian) catalog, so the two datasets are never joined on products and serve different features.
