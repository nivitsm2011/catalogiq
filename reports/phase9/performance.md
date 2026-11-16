# Phase 9 performance: photo to listing

Measured against the running API on this laptop (CPU, on battery), with `python -m scripts.profile_listing`
(raw numbers: `profile_before_cold.json`, `profile_after_cold.json`, `profile_before.json`).
"Before" is the Phase 8 commit `8fca932`; "after" is the Phase 9 code. Small samples (3 photos, 2 new
listings, one cold start each): read the numbers as indicative, not as precise.

| Stage (seconds) | Before | After |
|---|---|---|
| **Cold start:** first photo flow right after the server is ready | 2.14 | 1.31 |
| &nbsp;&nbsp;attributes (classifier) | 0.73 | 0.36 |
| &nbsp;&nbsp;quality check | 0.60 | 0.14 |
| **Warm, new photo:** attributes | 0.03 | 0.04 |
| &nbsp;&nbsp;similar listings (CLIP + search) | 0.20 | 0.22 |
| &nbsp;&nbsp;quality check | 0.22-0.26 | 0.02 |
| &nbsp;&nbsp;fixed stages total | about 0.45 | about 0.27 |
| **New listing** (LLM really runs), whole flow | 35.6 and 37.8 | 36.2 |
| &nbsp;&nbsp;of which the language model | 35.1 and 37.2 | 36.1 |
| Model reload after Ollama unloaded it (measured separately) | 4.2 added | avoided for 30 min idle |

**What changed**
1. **Embedding and probability cache per photo** (`AppState.photo_memo`): the seller flow sends one photo to
   several endpoints; CLIP and the classifier now run once per photo. The quality check dropped from about 0.22 s to 0.02 s.
2. **Warm-up at startup**: the classifier, the CLIP encoder and the language model are exercised in the background
   thread, so the first user does not pay for lazy initialisation. First-flow time fell from 2.14 s to 1.31 s.
3. **`keep_alive: 30m`** for Ollama (default is 5 minutes): after an idle gap the model stays loaded. A reload
   costs about 4.2 s (measured: 4.27 s for a one-word prompt against 0.09 s warm).

**What did not change, and why**
- A new listing still takes about 36 s because the 3B model generating on CPU is about 99% of the time. That is
  not an "easy win": it needs a smaller prompt, a smaller or quantised model, or a GPU. Demo products stay instant because their replies are cached.
- **Batching** does not apply to this path: it handles one photo at a time. The batch paths (catalog embeddings, the
  quality report) already batch (`classifier_probabilities`, `encode_images_u8`).
- The before and after servers each had one cold start and the laptop was on battery both times; the 0.73 s and 0.60 s
  cold figures in particular vary between runs.
