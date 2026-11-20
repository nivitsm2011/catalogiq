# Demo script (about 2 minutes 40 seconds)

Story: a seller lists a product, a shopper finds it and asks about it, the catalog team keeps quality up. One product, three people.
The spoken lines are written to be read aloud at a normal pace (about 130 words a minute, about 350 words in total). Every number said
here is in [reports/scorecard.md](../reports/scorecard.md).

## Before you record (5 minutes, once)

1. Start the servers in two PowerShell windows (from the project folder), and wait until the API log says "startup complete" (1-2 minutes):

   ```powershell
   .\tasks.ps1 api
   .\tasks.ps1 app
   ```
2. Open `http://localhost:8501`, switch **Demo mode on** in the sidebar (the 10 sample products answer instantly, so you never wait for the local model on camera).
3. Rehearse once with the same clicks so the caches are warm. Run the flow top to bottom, then reload the page.
4. Browser window about 1280 x 720, zoom 100%, bookmarks bar hidden, notifications off (Windows: Focus assist / Do not disturb), the laptop plugged in.
5. Keep ollama running (`ollama list` answers) in case you want to show one live generation.

## How to record it (Windows 11, nothing to install)

- **Snipping Tool** (press `Win + Shift + S`, choose the video camera icon, drag a box around the browser, press Start). Turn the microphone on in its settings, press Stop, then Save. It makes an MP4.
- **Xbox Game Bar** also works: click the browser, press `Win + G`, press the record button. It records one window at a time.
- Do the first take without speaking to learn the click order, then record with your voice. Read the lines below as notes, not word for word.
- Trim the start and end in the **Clipchamp** app (built into Windows 11): Open the MP4, drag the edges, Export at 1080p.
- For a GIF (the README uses a screenshot slideshow today), a free tool such as ScreenToGif can convert the MP4; it is a download, so check its source and size before installing. For LinkedIn upload the MP4 directly, it plays better than a GIF.
- Put the final file outside git (for example in your Videos folder or on YouTube as unlisted). Do not commit videos to the repository.

## The script

### 0:00 - Home (10 s)

*Show: the home page with the three cards.*

> "CatalogIQ is an AI copilot for online marketplaces. It helps sellers write better listings, shoppers get answers, and catalog teams find quality problems. It runs entirely on this laptop, on CPU."

### 0:10 - Seller Studio (60 s)

*Click: Seller Studio. Pick the demo shirt photo, **Use this photo**.*

> "A seller uploads one photo. My own trained classifier reads seven attributes in about 14 milliseconds. Where it is confident it fills the value in. Where it is not, like occasion here, it says *please confirm* instead of guessing."

*Click: **Generate listing**.*

> "A local 3-billion-parameter model writes the title, five bullets, a description and search tags."

*Point at the **Fact check** card.*

> "The important part is this fact check. In my tests a small model copied claims from example listings into 70 percent of listings when I gave it examples. So every listing is checked by code for materials, claims or brands the seller never gave, and rewritten once if needed. Without examples plus the rewrite, that rate was zero in my 50-product test."

*Scroll to **Quality score**.*

> "Then the photo and the listing get a score out of 100 with plain-English fixes. Blur, bad light, wrong colour or missing attributes. Last, similar listings from the catalog, found with CLIP embeddings."

*(Optional, 15 s, only if the laptop is plugged in: change **Occasion** to Formal and click Generate listing again. Say: "A new listing really runs the local model, about 15 to 40 seconds, so for this demo I use the cached ones.")*

### 1:10 - Shopper View (50 s)

*Click: Shopper View. Search `black formal shoes for men`, click **View** on the first result.*

> "Shoppers can search in words or with a photo. CLIP gets the right item in the top ten for about 89 percent of my test queries. And shop the look suggests what goes with it, using explicit rules, so it will not pair flip-flops with a shirt."

*Click the **Reviews & questions** tab, then the suggested question.*

> "Now questions. This answers only from this product's reviews, and shows which reviews it used. It cites them, and every cited id exists. Ask something the reviews do not cover..."

*Type `Is it waterproof?`*

> "...and it says there is not enough evidence. On my test it refused 93 percent of unanswerable questions. The honest cost: it also refused about 31 percent of questions it could have answered. I would rather say that than hide it."

### 2:00 - Catalog Ops (30 s)

*Click: Catalog Ops. Scroll: KPIs, worst listings, then the duplicate queue.*

> "For the catalog team: the quality checker flagged 8 percent of the 6,339 test listings, and this table shows the worst ones with the fix for each. Down here is the duplicate queue. A person decides: merge, or keep both."

*Click **Keep both** on one pair.*

> "I chose precision over recall for duplicates, because wrongly hiding a real listing is worse than missing a duplicate. These labels are provisional, so I treat the recall as unknown."

### 2:30 - Close (10 s)

*Show: Model Card page, scorecard.*

> "Every number comes from one command that re-runs the evaluations on fixed test sets, and the limitations are on this page too. Code and write-up are on GitHub."

## If something goes wrong on camera

- API not ready: the app shows "The CatalogIQ API is not running" with the command. Wait for "startup complete" and reload.
- A live generation is slow: keep talking over the loader (the Lottie animation) or cut it in editing; do not skip the fact-check card.
- Ollama stopped (it can after the laptop sleeps): run `ollama list` to restart it; the app shows an Ollama-not-running message with the same advice, and demo products keep working because their replies are cached.
