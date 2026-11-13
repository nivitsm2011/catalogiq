"""CatalogIQ home: what it does, headline numbers, and the three ways in."""

# ruff: noqa: E501  (page copy)

from __future__ import annotations

import streamlit as st

from app import client, ui

ui.setup("Home", "✨")
ui.require_api()
ui.hero(
    "Better listings. Smarter shoppers. Cleaner catalogs.",
    "CatalogIQ turns a product photo into a ready-to-review listing, answers shopper questions from real reviews, "
    "and tells catalog teams exactly which listings need attention. Everything runs locally on a laptop CPU.",
    "AI copilot for e-commerce marketplaces",
)

try:
    sc = client.scorecard_metrics()["metrics"]
except (client.ApiError, client.ApiUnavailable):
    sc = {}

if sc:
    st.markdown("##### Measured on held-out test sets")
    cols = st.columns(4)
    with cols[0]:
        ui.counter(
            "Attribute accuracy (macro-F1)",
            sc.get("vision.mean.macro_f1", 0),
            decimals=3,
            color=ui.INDIGO,
        )
    with cols[1]:
        ui.counter(
            "Look-alike search, Recall@10",
            100 * sc.get("search.clip.recall@10", 0),
            "%",
            1,
            ui.VIOLET,
        )
    with cols[2]:
        ui.counter(
            "Unanswerable questions refused",
            100 * sc.get("qa.correct_refusal_rate", 0),
            "%",
            1,
            ui.TEAL,
        )
    with cols[3]:
        ui.counter(
            "Listings flagged for review",
            sc.get("quality.batch.pct_with_any_issue", 0),
            "%",
            1,
            ui.CORAL,
        )

st.markdown("##### Choose your role")
c1, c2, c3 = st.columns(3)
for col, icon, title, text, page, label in (
    (
        c1,
        "📸",
        "Seller Studio",
        "Snap a photo. Confirm the attributes. Get a listing, a quality score with fix tips and similar existing products.",
        "pages/1_Seller_Studio.py",
        "Open Seller Studio",
    ),
    (
        c2,
        "🛍️",
        "Shopper View",
        "Search by words or photo, see outfits that go together, and ask questions answered from real reviews with citations.",
        "pages/2_Shopper_View.py",
        "Open Shopper View",
    ),
    (
        c3,
        "📊",
        "Catalog Ops",
        "See quality KPIs, the worst listings with fixes, and a queue of probable duplicates to merge or keep.",
        "pages/3_Catalog_Ops.py",
        "Open Catalog Ops",
    ),
):
    with col:
        ui.card(
            f'<div style="font-size:2rem">{icon}</div><h3>{title}</h3><div class="ciq-sub" style="min-height:4.2em">{text}</div>'
        )
        st.page_link(page, label=label, icon="➡️", use_container_width=True)

st.markdown("##### How it works")
st.graphviz_chart(
    """digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#D9D8F5", fillcolor="white", fontcolor="#1E1B4B"];
    edge [color="#8B5CF6"]; photo [label="Product photo", fillcolor="#E9E9FF"]; attrs [label="Attribute classifier\\n(MobileNetV3)"]; listing [label="Listing writer\\n(local LLM + guardrails)"];
    quality [label="Quality checker\\n(OpenCV + CLIP)"]; search [label="Visual search\\n(CLIP + FAISS)"]; reviews [label="Review Q&A\\n(RAG + citations)", fillcolor="#FFF4DB"];
    photo -> attrs -> listing; photo -> quality; photo -> search; listing -> quality [style=dashed]; reviews -> listing [style=invis]; }""",
    use_container_width=True,
)
st.info(
    "Honest by design: every number comes from a held-out test set, every generated claim is checked by guardrails, and the **Model Card** page shows the full scorecard including limitations."
)
