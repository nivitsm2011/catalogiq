"""Build a stopwatch page for the owner to time 5 manual listings (and, optionally, 5 draft reviews).

For each of 5 fixed test products the page shows the photo and what a seller knows (type, gender, colour,
brand). Task A: write the listing (title, 5 bullets, description, tags) by hand and time it. Task B
(optional): review and fix the CatalogIQ draft for the same product and time that. The saved CSV
(product_id, manual_minutes, review_minutes) feeds the product-impact section of the scorecard.
Usage: python scripts/make_timing_sheet.py
Output: reports/phase7/listing_timing.html   (save the downloaded CSV as reports/phase7/manual_listing_times.csv)
"""

# ruff: noqa: E501  (embedded HTML template)

from __future__ import annotations

import base64
import html
import json

import pandas as pd

from catalogiq.config import get_config, get_eval_config, get_rag_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.rag.evaluate import sample_products

N_ITEMS = 5
PAGE = """<!doctype html><meta charset="utf-8"><title>Time 5 listings</title>
<style>body{font-family:system-ui,sans-serif;max-width:920px;margin:20px auto;padding:0 14px;color:#222}
.card{border:1px solid #ccc;border-radius:8px;padding:14px;margin:14px 0;display:flex;gap:16px}
.card img{width:180px;height:240px;object-fit:contain;image-rendering:pixelated;border:1px solid #eee}
.t{font-size:22px;font-variant-numeric:tabular-nums;margin:6px 0}button{padding:8px 14px;font-size:15px;margin-right:6px}
pre{background:#f6f6f6;border-radius:6px;padding:8px;white-space:pre-wrap;font-size:13px}
.done{color:#080}</style>
<h2>How long does a listing take?</h2>
<p><b>Task A (required).</b> For each product write the listing by hand in any editor: a title, 5 bullet points, a description and search tags.
Press <b>Start</b>, write, press <b>Stop</b>. Work as you normally would, with the facts shown. <b>Do not look at the CatalogIQ draft first.</b></p>
<p><b>Task B (optional, but it makes the result much better).</b> Open the CatalogIQ draft, read it, fix anything wrong and say when it is ready to publish. Time that the same way.</p>
%CARDS%
<button onclick="save()">Save timings (CSV)</button> <span id="msg"></span>
<script>
const ids=%IDS%; const t={};
function fmt(s){const m=Math.floor(s/60);return m+":"+String(Math.floor(s%60)).padStart(2,"0")}
function tick(k){const o=t[k];const e=o.acc+(o.run?(Date.now()-o.start)/1000:0);document.getElementById(k).textContent=fmt(e);return e}
function toggle(k){t[k]=t[k]||{acc:0,run:false,start:0};const o=t[k];
  if(o.run){o.acc+=(Date.now()-o.start)/1000;o.run=false}else{o.start=Date.now();o.run=true}
  document.getElementById("b"+k).textContent=o.run?"Stop":(o.acc>0?"Resume":"Start");}
setInterval(()=>{for(const k in t)tick(k)},500);
function minutes(k){const o=t[k];if(!o)return "";const e=o.acc+(o.run?(Date.now()-o.start)/1000:0);return e>0?(e/60).toFixed(2):""}
function save(){
  const rows=["product_id,manual_minutes,review_minutes"];let missing=0;
  for(const i of ids){const a=minutes("a"+i);if(!a)missing++;rows.push([i,a,minutes("b"+i)].join(","))}
  if(missing){document.getElementById("msg").textContent=missing+" Task A timing(s) missing";return}
  const l=document.createElement("a");l.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));l.download="manual_listing_times.csv";l.click();
  document.getElementById("msg").textContent="Saved. Move manual_listing_times.csv to reports/phase7/";}
</script>
"""


def main() -> None:
    """Pick the first 5 evaluation products, embed their photos and the saved drafts, write the page."""
    products = sample_products().head(N_ITEMS)
    images_dir = resolve_path(get_config()["data"]["fashion"]["images_dir"])
    drafts = {}
    path = resolve_path(get_rag_config()["paths"]["eval_dir"]) / "gen_no_rag_fix.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        drafts[rec["id"]] = rec["listing"]
    cards = []
    for k, p in enumerate(products.to_dict("records"), start=1):
        pid = int(p["id"])
        img = base64.b64encode(image_path(images_dir, pid).read_bytes()).decode()
        facts = (
            f"{p['articleType']} | {p['gender']} | colour {p['baseColour']} | usage {p['usage']}"
            + (f" | brand {p['brand']}" if p.get("brand") else "")
        )
        d = drafts.get(pid)
        draft = (
            "(draft not available)"
            if not d
            else html.escape(
                f"TITLE: {d['title']}\n"
                + "\n".join(f"- {b}" for b in d["bullets"])
                + f"\nDESCRIPTION: {d['description']}\nTAGS: {', '.join(d['search_tags'])}"
            )
        )
        cards.append(
            f'<div class="card"><img src="data:image/jpeg;base64,{img}"><div><h3>Product {k} (id {pid})</h3><p>{html.escape(facts)}</p>'
            f'<div>Task A, write by hand: <span class="t" id="a{pid}">0:00</span> <button id="ba{pid}" onclick="toggle(\'a{pid}\')">Start</button></div>'
            f'<div>Task B, review the draft: <span class="t" id="b{pid}">0:00</span> <button id="bb{pid}" onclick="toggle(\'b{pid}\')">Start</button></div>'
            f"<details><summary>CatalogIQ draft (open only for Task B)</summary><pre>{draft}</pre></details></div></div>"
        )
    page = PAGE.replace("%CARDS%", "\n".join(cards)).replace(
        "%IDS%", json.dumps([int(i) for i in products["id"]])
    )
    out = resolve_path(get_eval_config()["paths"]["out_dir"]) / "listing_timing.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    pd.DataFrame({"product_id": products["id"]}).to_csv(
        out.with_name("timing_products.csv"), index=False
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
