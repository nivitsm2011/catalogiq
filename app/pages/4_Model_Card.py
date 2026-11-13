"""Model Card & Metrics: the evaluation scorecard and a simple architecture diagram."""

# ruff: noqa: E501  (page copy)

from __future__ import annotations

import re

import streamlit as st

from app import client, ui

ui.setup("Model Card", "🧪")
ui.require_api()
ui.hero(
    "Model Card & Metrics",
    "What each model does, how well it works on held-out data, and where it fails. Nothing here is a marketing number.",
    "Transparency",
)

try:
    sc = client.scorecard_metrics()
    m, g = sc["metrics"], sc["git"]
except (client.ApiError, client.ApiUnavailable):
    sc, m, g = None, {}, {}

if sc:
    cols = st.columns(4)
    with cols[0]:
        ui.counter(
            "Vision macro-F1 (7 attributes)", m.get("vision.mean.macro_f1", 0), "", 3, ui.INDIGO
        )
    with cols[1]:
        ui.counter("Search Recall@10", 100 * m.get("search.clip.recall@10", 0), "%", 1, ui.VIOLET)
    with cols[2]:
        ui.counter(
            "Q&A correct refusals", 100 * m.get("qa.correct_refusal_rate", 0), "%", 1, ui.TEAL
        )
    with cols[3]:
        ui.counter(
            "Drift vs earlier phases",
            sc["drift"].get("match", 0),
            f"of {sum(sc['drift'].values())} match",
            0,
            ui.CORAL,
        )
    st.caption(
        f"Scorecard generated {sc.get('generated')} from commit {g.get('short', '?')}. Experiment `catalogiq-eval` in MLflow."
    )

st.markdown("#### Architecture")
st.graphviz_chart(
    """digraph G { rankdir=LR; bgcolor="transparent"; fontname="Helvetica";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#D9D8F5", fillcolor="white", fontcolor="#1E1B4B"]; edge [color="#8B5CF6", fontname="Helvetica", fontsize=9, fontcolor="#6B6B8D"];
    subgraph cluster_ui { label="Streamlit app"; style="rounded,dashed"; color="#CFCBFF"; fontcolor="#5B5BF0"; seller [label="Seller Studio"]; shopper [label="Shopper View"]; ops [label="Catalog Ops"]; card [label="Model Card"]; }
    api [label="FastAPI\\nvalidation · metrics · logging", fillcolor="#E9E9FF"];
    subgraph cluster_models { label="Models loaded once at startup (CPU)"; style="rounded,dashed"; color="#CFCBFF"; fontcolor="#5B5BF0";
      clf [label="Attribute classifier\\nMobileNetV3, 7 heads"]; clip [label="CLIP ViT-B/32\\nFAISS index"]; llm [label="llama3.2:3b via Ollama\\nLLMClient interface", fillcolor="#FFF4DB"]; qc [label="Quality checker\\nOpenCV + CLIP"]; }
    subgraph cluster_data { label="Local data"; style="rounded,dashed"; color="#CFCBFF"; fontcolor="#5B5BF0"; kb [label="ChromaDB\\n800 listings · 166k review chunks"]; cat [label="Catalog\\n42k photos · 6.3k test listings"]; }
    seller -> api; shopper -> api; ops -> api; card -> api;
    api -> clf [label="attributes"]; api -> clip [label="search · outfits"]; api -> qc [label="score + fixes"]; api -> llm [label="listing · answers"]; llm -> kb [label="RAG"]; clip -> cat; qc -> clf [style=dashed, label="consistency"]; }""",
    use_container_width=True,
)

st.markdown("#### Scorecard")
try:
    text = client.scorecard()
except client.ApiError as exc:
    st.warning(exc.message)
    st.stop()
text = re.sub(r"^# .*\n", "", text, count=1)
sections = re.split(r"(?m)^## ", text)
st.markdown(sections[0], unsafe_allow_html=True)
for sec in sections[1:]:
    title, _, body = sec.partition("\n")
    with st.expander(title, expanded=title.startswith(("1.", "9."))):
        st.markdown(body, unsafe_allow_html=True)
