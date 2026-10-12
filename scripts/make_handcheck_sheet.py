"""Build the blind hand-check page: 10 generated listings to rate 1-5 for clarity and accuracy.

Mode (no-RAG / RAG) and the judge's scores are hidden; they are saved in the key file only.
Usage: python scripts/make_handcheck_sheet.py   (after scripts/evaluate_rag.py)
Outputs: reports/phase4/handcheck.html, reports/phase4/handcheck_key.csv
After rating: click "Save ratings" and keep the file as reports/phase4/handcheck_ratings.csv.
"""

# ruff: noqa: E501  (embedded HTML template)

from __future__ import annotations

import base64
import html
import json

import pandas as pd

from catalogiq.config import get_config, get_rag_config, resolve_path
from catalogiq.data.fashion import image_path

PAGE = """<!doctype html><meta charset="utf-8"><title>Listing hand-check</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}
.card{border:1px solid #ccc;border-radius:8px;padding:16px;margin:18px 0;display:flex;gap:16px}
img{width:140px;height:140px;object-fit:contain;border:1px solid #eee}
h3{margin:.2em 0}.facts{background:#f6f6f6;padding:8px;border-radius:6px;font-size:14px;white-space:pre-line}
.rate label{margin-right:10px}.rate{margin:6px 0}button{padding:10px 18px;font-size:16px}
.warn{color:#8a5a00;font-size:13px}
</style>
<h2>Rate 10 product listings</h2>
<p><b>Clarity</b>: easy to read, well organised, specific (1 = confusing, 5 = excellent).<br>
<b>Accuracy</b>: states only what the PRODUCT FACTS and SELLER NOTES say; nothing invented
(1 = several invented facts, 5 = all supported). Look at the photo too: the facts come from a
classifier that can be wrong, so judge the listing against the FACTS shown, not the photo.</p>
%CARDS%
<button onclick="save()">Save ratings (CSV)</button> <span id="msg"></span>
<script>
const ids=%IDS%;
function save(){
  let rows=["item,clarity,accuracy"],missing=0;
  for(const i of ids){
    const c=document.querySelector('input[name="c'+i+'"]:checked'),a=document.querySelector('input[name="a'+i+'"]:checked');
    if(!c||!a){missing++;continue}
    rows.push([i,c.value,a.value].join(","));
  }
  if(missing){document.getElementById("msg").textContent=missing+" item(s) not rated yet";return}
  const blob=new Blob([rows.join("\\n")],{type:"text/csv"});
  const l=document.createElement("a");l.href=URL.createObjectURL(blob);l.download="handcheck_ratings.csv";l.click();
  document.getElementById("msg").textContent="Saved. Move handcheck_ratings.csv to reports/phase4/";
}
</script>
"""


def _radios(name: str) -> str:
    return "".join(
        f'<label><input type="radio" name="{name}" value="{v}">{v}</label>' for v in range(1, 6)
    )


def main() -> None:
    """Sample 10 listings across three arms, shuffle, and write the page and the key."""
    cfg = get_rag_config()
    out = resolve_path(cfg["paths"]["eval_dir"])
    arms = ["no_rag"] * 3 + ["rag"] * 4 + ["rag_fix"] * 3  # arm hidden from the rater
    frames = {}
    for mode in set(arms):
        lines = (out / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines()
        frames[mode] = pd.DataFrame([json.loads(x) for x in lines]).set_index("id")
    ok_ids = sorted(set.intersection(*[set(f.index[f["ok"]]) for f in frames.values()]))
    pick = pd.Series(ok_ids).sample(len(arms), random_state=cfg["seed"]).tolist()  # 10 products
    rng_arms = pd.Series(arms).sample(frac=1.0, random_state=cfg["seed"]).tolist()
    recs = [
        frames[m].loc[[pid]].reset_index().assign(mode=m)
        for pid, m in zip(pick, rng_arms, strict=True)
    ]
    sample = pd.concat(recs).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    sample["item"] = range(1, len(sample) + 1)
    images_dir = resolve_path(get_config()["data"]["fashion"]["images_dir"])
    cards = []
    for r in sample.itertuples():
        img = base64.b64encode(image_path(images_dir, int(r.id)).read_bytes()).decode()
        facts = "\n".join(
            f"{k}: {v[0]}" + (" (UNCERTAIN, please confirm)" if v[2] else "")
            for k, v in r.attributes.items()
            if k in cfg["generation"]["prompt_attributes"]
        )
        notes = "\n".join(f"{k}: {v}" for k, v in r.notes.items()) or "(none)"
        li = r.listing
        bullets = "".join(f"<li>{html.escape(b)}</li>" for b in li["bullets"])
        warn = "; ".join(li["warnings"])
        cards.append(
            f'<div class="card"><img src="data:image/jpeg;base64,{img}"><div>'
            f"<h3>Item {r.item}</h3><div class='facts'>FACTS\n{html.escape(facts)}\n"
            f"SELLER NOTES\n{html.escape(notes)}</div>"
            f"<p><b>{html.escape(li['title'])}</b></p><ul>{bullets}</ul>"
            f"<p>{html.escape(li['description'])}</p>"
            f"<p><i>Tags: {html.escape(', '.join(li['search_tags']))}</i></p>"
            f"<p class='warn'>Warnings: {html.escape(warn)}</p>"
            f"<div class='rate'>Clarity: {_radios(f'c{r.item}')}</div>"
            f"<div class='rate'>Accuracy: {_radios(f'a{r.item}')}</div></div></div>"
        )
    page = PAGE.replace("%CARDS%", "\n".join(cards)).replace("%IDS%", str(sample["item"].tolist()))
    (out / "handcheck.html").write_text(page, encoding="utf-8")
    sample[["item", "id", "mode"]].to_csv(out / "handcheck_key.csv", index=False)


if __name__ == "__main__":
    main()
