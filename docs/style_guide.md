# CatalogIQ category style guide

_Generated from `configs/rag.yaml` by `scripts/build_knowledge_base.py`. Edit the config, not this file._

These rules are (1) indexed in the vector store so the generator can retrieve the ones for a product's category and (2) enforced after generation by `catalogiq.rag.guardrails`.

## Rules for every category

### Title
- Format: `{Brand} {Gender} {Type}, {Key attribute}, {Colour}` (skip a part when that fact is unknown).
- Example: `Puma Men's T-Shirt, Casual, Black`.
- Length: 25 to 80 characters.
- Start with the brand when the seller gave one. Never invent a brand.
- Only confident attributes go in the title. Uncertain ones are left out.

### Bullets, description, tags
- Exactly 5 bullets, each at most 160 characters, one fact per bullet.
- Description: 30 to 120 words, plain sentences, no hype.
- 5 to 12 search tags, lower case, buyer search words (type, gender, colour, occasion).

### Banned words (hype and sales language)
- `\bbest\b`
- `\b#1\b`
- `\bnumber one\b`
- `\bperfect\b`
- `\bguarantee`
- `\bunbeatable\b`
- `\bcheapest\b`
- `\bamazing\b`
- `\bworld.?class\b`
- `\bunmatched\b`
- `\bpremium quality\b`
- `\bhighest quality\b`
- `\bbuy now\b`
- `\blimited time\b`
- `\bfree shipping\b`
- `\bsale\b`

### Banned claims (health or unverifiable benefits)
- `\bcure[sd]?\b`
- `\bheal(s|ing)?\b`
- `\bmedical\b`
- `\bclinically\b`
- `\bdermatolog`
- `\bhypoallergenic\b`
- `\banti.?bacterial\b`
- `\banti.?microbial\b`
- `\brelieves?\b`
- `\btherapeutic\b`
- `\bpain\b`
- `\bimprove[sd]? circulation\b`
- `\bFDA\b`
- `\b100% (safe|natural|organic)\b`

### Facts policy
- Use only facts from the predicted attributes and the seller notes.
- Materials, brands, sizes, pack sizes and performance claims (waterproof, breathable, machine washable, UV protection ...) need to be in the seller notes.
- An attribute the classifier is not sure about is written as _please confirm_ and listed in `warnings`.

## Rules per category

| Category | Required attributes | Example Kaggle article types |
|---|---|---|
| costume_novelty | articleType | - |
| watches | gender, articleType, baseColour | Watches |
| eyewear | gender, articleType, baseColour | Sunglasses |
| jewellery | gender, articleType, baseColour | Earrings, Pendant, Ring, Necklace and Chains |
| innerwear_socks | gender, articleType, baseColour | Socks, Bra, Briefs, Innerwear Vests |
| footwear | gender, articleType, baseColour, usage | Casual Shoes, Flip Flops, Sandals, Formal Shoes |
| bags_wallets | gender, articleType, baseColour | Handbags, Backpacks, Wallets, Clutches |
| dresses_ethnic | gender, articleType, baseColour, usage | Dresses, Sarees, Kurtas, Kurtis |
| bottomwear | gender, articleType, baseColour, usage | Jeans, Track Pants, Shorts, Trousers |
| topwear | gender, articleType, baseColour, usage | Tshirts, Shirts, Tops, Sweatshirts |
| accessories | gender, articleType, baseColour | Belts, Caps, Scarves, Ties |
| beauty | articleType, baseColour | Perfume and Body Mist, Deodorant, Lipstick, Nail Polish |

### costume_novelty
- Required: articleType.
- Say clearly that this is a costume or novelty item and who it is sized for.

### watches
- Required: gender, articleType, baseColour.
- Mention strap or band colour only if given. Never state water resistance or movement type unless the seller notes say so.

### eyewear
- Required: gender, articleType, baseColour.
- Never claim UV protection, polarisation or lens material unless the seller notes say so.

### jewellery
- Required: gender, articleType, baseColour.
- Never state metal, stone or plating (gold, silver, sterling, crystal, diamond) unless the seller notes say so.

### innerwear_socks
- Required: gender, articleType, baseColour.
- Fabric and fit claims (cotton, stretch, seamless, compression) only if the seller notes say so. State pack size only if given.

### footwear
- Required: gender, articleType, baseColour, usage.
- Sole, upper material, cushioning and waterproofing claims only if the seller notes say so. Mention size range if given.

### bags_wallets
- Required: gender, articleType, baseColour.
- Capacity, compartments and material (leather, canvas) only if the seller notes say so.

### dresses_ethnic
- Required: gender, articleType, baseColour, usage.
- Length, neckline, fabric and occasion only if known. Ethnic wear: mention the occasion (usage) when it is confident.

### bottomwear
- Required: gender, articleType, baseColour, usage.
- Fit, rise and fabric only if the seller notes say so. Mention size range if given.

### topwear
- Required: gender, articleType, baseColour, usage.
- Fit, sleeve length, neckline and fabric only if the seller notes say so. Mention size range if given.

### accessories
- Required: gender, articleType, baseColour.
- Material (leather, wool, silk) and adjustability only if the seller notes say so.

### beauty
- Required: articleType, baseColour.
- No skin, health or longevity claims (dermatologically tested, long-lasting, hypoallergenic) unless the seller notes say so.
