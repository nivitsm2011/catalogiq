"""Build the blind faithfulness hand-check page: 15 answers, the owner ticks faithful yes/no.

Answers are drawn from those the pipeline did not refuse, about half judged faithful and half not by
the LLM judge (the verdict is hidden). The page shows the question, the answer text and the full text
of every cited review.
Usage: python scripts/make_qa_handcheck_sheet.py   (after scripts/evaluate_qa.py generate + judge)
Outputs: reports/phase5/qa_handcheck.html, reports/phase5/qa_handcheck_key.csv
Save the ticked file as reports/phase5/qa_handcheck_ratings.csv.
"""

# ruff: noqa: E501  (embedded HTML template)

from __future__ import annotations

import html
import json

import pandas as pd

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.rag.eval_qa import review_texts

PAGE = """<!doctype html><meta charset="utf-8"><title>Answer faithfulness check</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 14px;color:#222}
.card{border:1px solid #ccc;border-radius:8px;padding:14px;margin:14px 0}.rev{background:#f6f6f6;border-radius:6px;padding:8px;margin:6px 0;font-size:14px;white-space:pre-wrap}
.ans{background:#eef6ff;border-radius:6px;padding:8px;margin:8px 0}label{margin-right:18px}button{padding:10px 18px;font-size:16px}</style>
<h2>Is every claim in the answer supported by the cited reviews?</h2>
<p>For each item read the <b>answer</b> (blue box) and the <b>cited reviews</b> (grey boxes; ids like R123 match the ids in the answer).
Choose <b>Yes</b> only if EVERY claim in the answer (including the "agree" and "disagree" lines) is stated or clearly implied by a cited review.
Choose <b>No</b> if any claim is missing from the cited reviews or contradicts them. Ignore whether the answer is complete or well written.
Optionally write which claim was unsupported.</p>
%CARDS%
<button onclick="save()">Save ratings (CSV)</button> <span id="msg"></span>
<script>
const ids=%IDS%;
function q(s){return '"'+String(s).replace(/"/g,'""')+'"'}
function save(){
  let rows=["item,faithful,note"],miss=0;
  for(const i of ids){
    const c=document.querySelector('input[name="f'+i+'"]:checked');
    if(!c){miss++;continue}
    rows.push([i,c.value,q(document.getElementById("n"+i).value)].join(","));
  }
  if(miss){document.getElementById("msg").textContent=miss+" item(s) not rated yet";return}
  const l=document.createElement("a");l.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));l.download="qa_handcheck_ratings.csv";l.click();
  document.getElementById("msg").textContent="Saved. Move qa_handcheck_ratings.csv to reports/phase5/";
}
</script>
"""


def main() -> None:
    """Pick 15 answered questions (balanced by judge verdict) and write the page and the key."""
    cfg = get_qa_config()
    out = resolve_path(cfg["paths"]["eval_dir"])
    n = cfg["evaluation"]["handcheck_n"]
    answers = {
        a["id"]: a for a in map(json.loads, (out / "answers.jsonl").read_text("utf-8").splitlines())
    }
    judged = {
        j["id"]: j["judge"]
        for j in map(json.loads, (out / "faithfulness.jsonl").read_text("utf-8").splitlines())
    }
    ok = pd.DataFrame(
        [{"id": i, "judge_faithful": j["faithful"]} for i, j in judged.items() if j["n_claims"] > 0]
    )
    pos, neg = ok[ok["judge_faithful"]], ok[~ok["judge_faithful"]]
    n_pos = min(len(pos), n // 2 + n % 2)
    pick = (
        pd.concat(
            [
                pos.sample(n_pos, random_state=cfg["seed"]),
                neg.sample(min(len(neg), n - n_pos), random_state=cfg["seed"]),
            ]
        )
        .sample(frac=1.0, random_state=cfg["seed"])
        .reset_index(drop=True)
    )
    pick["item"] = range(1, len(pick) + 1)
    rv = review_texts().set_index("review_id")["full"]
    cards = []
    for r in pick.itertuples():
        a = answers[r.id]
        o = a["output"]
        revs = "".join(
            f'<div class="rev"><b>{c}</b> ({html.escape(str(rv[c][:900]))})</div>'
            for c in o["citations"]
        )
        cards.append(
            f'<div class="card"><h3>Item {r.item}</h3><p><b>Question:</b> {html.escape(a["q"])}</p>'
            f'<div class="ans"><b>Answer:</b> {html.escape(o["answer"])}<br><b>Reviewers agree:</b> {html.escape(o["reviewers_agree"])}'
            f'<br><b>Reviewers disagree:</b> {html.escape(o["reviewers_disagree"])}</div>'
            f"<p><b>Cited reviews</b></p>{revs}"
            f'<label><input type="radio" name="f{r.item}" value="yes">Yes, all claims supported</label>'
            f'<label><input type="radio" name="f{r.item}" value="no">No, something is unsupported</label><br>'
            f'<input id="n{r.item}" style="width:90%;margin-top:6px" placeholder="optional: which claim?"></div>'
        )
    page = PAGE.replace("%CARDS%", "\n".join(cards)).replace("%IDS%", str(pick["item"].tolist()))
    (out / "qa_handcheck.html").write_text(page, encoding="utf-8")
    pick[["item", "id", "judge_faithful"]].to_csv(out / "qa_handcheck_key.csv", index=False)
    print(f"wrote {len(pick)} items")


if __name__ == "__main__":
    main()
