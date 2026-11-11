"""Build the 200-review aspect labelling page.

100 chunks are drawn at random (unbiased prevalence); 100 more are enriched so that every aspect has
positives: per aspect, some chunks a keyword rule matched and some that only look similar by meaning
(MiniLM similarity to the aspect description, rule-miss). The page hides which kind each chunk is.
Usage: python scripts/make_aspect_label_sheet.py
Outputs: reports/phase5/aspect_sample.csv, reports/phase5/aspect_labeling.html
Save the ticked labels as reports/phase5/aspect_labels.csv (the page has a Save button).
"""

# ruff: noqa: E501  (embedded HTML template)

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.rag.aspects import aspect_names, embedding_scores, rule_tag
from catalogiq.rag.reviews_index import load_chunks

logger = get_logger("make_aspect_label_sheet")

PAGE = """<!doctype html><meta charset="utf-8"><title>Aspect labelling</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 14px;color:#222}
.card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:10px 0}.txt{white-space:pre-wrap;margin-bottom:8px}
label{margin-right:14px;white-space:nowrap}.def{background:#f4f4f4;padding:10px;border-radius:6px;font-size:14px}
button{padding:10px 18px;font-size:16px}#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}</style>
<h2>Tick every aspect the review text TALKS ABOUT (praise or complaint)</h2>
<div class="def"><b>fit/size</b>: how it fits, size, runs small/large &middot; <b>fabric/material</b>: what it is made of, texture, thickness &middot;
<b>colour</b>: the colour or whether it matches the photo &middot; <b>quality/durability</b>: build quality, lasting, breaking, stitching &middot;
<b>comfort</b>: how comfortable it is &middot; <b>delivery/packaging</b>: shipping, packaging, seller service.<br>
A review can have several aspects or none (leave all unticked). Judge only the text shown.</div>
<div id="bar"><span id="count">0</span> / %N% reviewed (a review counts as reviewed when you tick anything or press "none") &nbsp; <button onclick="save()">Save labels (CSV)</button> <span id="msg"></span></div>
%CARDS%
<script>
const N=%N%, ASP=%ASP%;
function upd(){let c=0;for(let i=0;i<N;i++){if(ASP.some(a=>document.getElementById(a+i).checked)||document.getElementById("none"+i).checked)c++}document.getElementById("count").textContent=c}
document.addEventListener("change",upd);
function save(){
  let rows=["chunk_id,"+ASP.join(",")],miss=0;const ids=%IDS%;
  for(let i=0;i<N;i++){
    const none=document.getElementById("none"+i).checked,v=ASP.map(a=>document.getElementById(a+i).checked?1:0);
    if(!none&&v.every(x=>x==0)){miss++;continue}
    rows.push(ids[i]+","+v.join(","));
  }
  if(miss){document.getElementById("msg").textContent=miss+" not reviewed yet";return}
  const l=document.createElement("a");l.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));l.download="aspect_labels.csv";l.click();
  document.getElementById("msg").textContent="Saved. Move aspect_labels.csv to reports/phase5/";
}
</script>
"""

SHORT = {
    "fit_size": "fit/size",
    "fabric_material": "fabric/material",
    "colour_accuracy": "colour",
    "quality_durability": "quality/durability",
    "comfort": "comfort",
    "delivery_packaging": "delivery/packaging",
}


def main() -> None:
    """Draw the sample and write the CSV and the HTML page."""
    cfg = get_qa_config()
    lab = cfg["labelling"]
    rng = np.random.default_rng(lab["seed"])
    names = aspect_names()
    chunks = load_chunks()
    chunks = chunks[chunks["text"].str.len() >= 40].reset_index(drop=True)
    random_ids = rng.choice(len(chunks), size=lab["n_random"], replace=False)
    pool_ids = rng.choice(
        np.setdiff1d(np.arange(len(chunks)), random_ids), size=12000, replace=False
    )
    pool = chunks.iloc[pool_ids]
    rules = rule_tag(pool["text"].tolist())
    sims = embedding_scores(pool["text"].tolist())
    per_aspect = lab["n_enriched"] // len(names) + 1
    picked: list[int] = []
    for j, a in enumerate(names):
        rule_hit = np.flatnonzero(rules[:, j])
        cutoff = np.quantile(sims[:, j], 0.98)
        sim_only = np.flatnonzero((sims[:, j] >= cutoff) & ~rules[:, j])
        take = list(
            rng.choice(rule_hit, size=min(per_aspect // 2 + 1, len(rule_hit)), replace=False)
        )
        take += list(rng.choice(sim_only, size=min(per_aspect // 2, len(sim_only)), replace=False))
        picked += [pool_ids[i] for i in take]
        logger.info(
            "%s: %d rule-matched + %d similarity-only candidates", a, len(rule_hit), len(sim_only)
        )
    picked = list(dict.fromkeys(picked))[: lab["n_enriched"]]
    sample = pd.concat(
        [
            chunks.iloc[random_ids].assign(source="random"),
            chunks.iloc[picked].assign(source="enriched"),
        ]
    )
    sample = sample.sample(frac=1.0, random_state=lab["seed"]).reset_index(drop=True)
    out = resolve_path(cfg["paths"]["eval_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sample[["chunk_id", "review_id", "item_id", "rating", "source", "text"]].to_csv(
        resolve_path(cfg["paths"]["aspect_sample"]), index=False
    )
    cards = []
    for i, r in enumerate(sample.itertuples()):
        boxes = (
            "".join(f'<label><input type="checkbox" id="{a}{i}">{SHORT[a]}</label>' for a in names)
            + f'<label><input type="checkbox" id="none{i}"><i>none of these</i></label>'
        )
        cards.append(
            f'<div class="card"><div class="txt"><b>{i + 1}.</b> {html.escape(r.text)}</div>{boxes}</div>'
        )
    page = (
        PAGE.replace("%CARDS%", "\n".join(cards))
        .replace("%N%", str(len(sample)))
        .replace("%ASP%", json.dumps(names))
        .replace("%IDS%", json.dumps(sample["chunk_id"].tolist()))
    )
    (out / "aspect_labeling.html").write_text(page, encoding="utf-8")
    logger.info("Wrote %d chunks to label (%s)", len(sample), out / "aspect_labeling.html")


if __name__ == "__main__":
    main()
