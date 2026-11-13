"""Catalog Ops dashboard: quality KPIs, issues by category, worst listings with fixes, duplicate review queue."""

# ruff: noqa: E501  (page copy)

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from app import client, ui

ui.setup("Catalog Ops", "📊")
ui.require_api()
S = st.session_state
ui.hero(
    "Catalog Ops Dashboard",
    "Where is catalog quality slipping, which listings should a person look at first, and which products are probably duplicates?",
    "For catalog and operations teams",
)

NICE_ISSUE = {
    "blurry": "Blurry photo",
    "too_dark": "Too dark",
    "overexposed": "Washed out",
    "low_resolution": "Low resolution",
    "bad_aspect_ratio": "Odd shape",
    "cluttered_background": "Cluttered background",
    "text_or_watermark": "Text / watermark",
    "mismatch_baseColour": "Colour mismatch",
    "mismatch_articleType": "Type mismatch",
    "mismatch_gender": "Gender mismatch",
}
PALETTE = [ui.INDIGO, ui.VIOLET, ui.CORAL, ui.TEAL, ui.AMBER, "#38BDF8", "#F472B6", "#94A3B8"]


def animated_bar(
    x: list[str],
    y: list[float],
    title: str,
    colour: str,
    horizontal: bool = False,
    height: int = 340,
) -> go.Figure:
    """A bar chart whose bars grow from zero (press Replay); updates animate smoothly."""
    steps = [i / 12 for i in range(13)]

    def mk(f: float) -> go.Bar:
        vals = [v * f for v in y]
        return go.Bar(
            x=vals if horizontal else x,
            y=x if horizontal else vals,
            orientation="h" if horizontal else "v",
            marker=dict(color=colour),
        )

    fig = go.Figure(data=[mk(1.0)], frames=[go.Frame(data=[mk(f)]) for f in steps])
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=ui.INK)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", color=ui.INK),
        transition=dict(duration=500, easing="cubic-in-out"),
        xaxis=dict(gridcolor="#ECEBFF"),
        yaxis=dict(gridcolor="#ECEBFF", autorange="reversed" if horizontal else True),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=1.0,
                xanchor="right",
                y=1.18,
                buttons=[
                    dict(
                        label="▶ Replay",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=45, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=False,
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    return fig


try:
    ov = client.ops_summary(None, 12)
except client.ApiError as exc:
    st.error(exc.message)
    st.stop()

dup = None
try:
    dup = client.ops_duplicates(6, S.get("dup_version", 0))
except client.ApiError:
    pass

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.counter("Average quality score", ov["avg_score"], "/ 100", 1, ui.TEAL)
with k2:
    ui.counter("Listings with an issue", ov["pct_with_issue"], "%", 1, ui.CORAL)
with k3:
    ui.counter("Need attention (score < 85)", ov["pct_needs_attention"], "%", 1, ui.AMBER)
with k4:
    ui.counter(
        f"Duplicate clusters ({dup['n_products']:,} products)" if dup else "Duplicate clusters",
        dup["n_clusters"] if dup else 0,
        "",
        0,
        ui.VIOLET,
    )

st.caption(
    f"{ov['n_listings']:,} test listings checked. Flags include some false alarms (see the Model Card): jewellery and watches are over-flagged by one-size-fits-all thresholds."
)

c1, c2 = st.columns(2, gap="large")
with c1:
    counts = sorted(ov["issue_counts"].items(), key=lambda kv: -kv[1])
    fig = animated_bar(
        [NICE_ISSUE.get(k, k) for k, _ in counts],
        [v for _, v in counts],
        "Issues by type (number of listings)",
        ui.INDIGO,
        horizontal=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with c2:
    groups = ov["by_group"]
    fig2 = animated_bar(
        [g["group"].replace("_", " ") for g in groups],
        [g["pct_with_issue"] for g in groups],
        "Share of listings with an issue, by category (%)",
        ui.CORAL,
        horizontal=True,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.markdown("#### Worst listings")
cats = ["All categories"] + sorted({g["group"] for g in ov["by_group"]})
pick = st.selectbox("Filter by category", cats, key="ops_cat", label_visibility="collapsed")
worst = ov["worst"] if pick == "All categories" else client.ops_summary(pick, 12)["worst"]
if not worst:
    st.info("No listings in this category.")
for row in worst[:8]:
    img, info = st.columns([1, 6])
    with img:
        st.image(client.url(row["image_url"]), width=84)
    with info:
        flags = (
            [(NICE_ISSUE.get(i, i), "amber") for i in row["issues"]]
            + [(f"{NICE_ISSUE.get('mismatch_' + m, m)}", "red") for m in row["mismatches"]]
            + [(f"missing {m}", "grey") for m in row["missing"]]
        )
        score_chip = ui.chip(
            f"score {row['quality_score']:.0f}", "red" if row["quality_score"] < 70 else "amber"
        )
        flag_html = "".join(ui.chip(t, k) for t, k in flags)
        fix_html = "<br>".join("💡 " + html.escape(f) for f in row["fixes"][:3])
        st.markdown(
            f'<div class="ciq-card" style="padding:12px 16px;margin-bottom:8px"><div style="display:flex;justify-content:space-between;gap:1rem">'
            f'<div class="ciq-title">{html.escape(row["title"])}</div>{score_chip}</div><div style="margin:.3rem 0">{flag_html}</div><div class="ciq-sub">{fix_html}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("#### Duplicate review queue")
if not dup:
    st.info(
        "The duplicate queue is not available yet (it is built on the first request and can take a minute)."
    )
else:
    st.caption(
        f"{dup['n_pairs']:,} probable duplicate pairs (CLIP similarity ≥ the tuned threshold and a matching perceptual hash); {dup['n_decided']} decided. The labels behind the threshold are provisional."
    )
    for p in dup["pairs"]:
        a, b, m = st.columns([1.2, 1.2, 1.6])
        with a:
            st.markdown(
                ui.item_card_html({"title": p["title_a"], "image_url": p["image_a"]}),
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                ui.item_card_html({"title": p["title_b"], "image_url": p["image_b"]}),
                unsafe_allow_html=True,
            )
        with m:
            ui.chips(
                [
                    (f"{p['similarity']:.1%} similar", "blue"),
                    (f"pHash distance {p['phash_distance']:.0f}", "grey"),
                ]
            )
            st.caption(
                "Same product listed twice? Colour variants and size variants are not duplicates."
            )
            c1, c2 = st.columns(2)
            for col, label, dec, kind in (
                (c1, "Merge", "merge", "primary"),
                (c2, "Keep both", "keep_both", "secondary"),
            ):
                if col.button(
                    label,
                    key=f"dup_{dec}_{p['id_a']}_{p['id_b']}",
                    type=kind,
                    use_container_width=True,
                ):
                    client.post_json(
                        "/ops/duplicates/decision",
                        {"id_a": p["id_a"], "id_b": p["id_b"], "decision": dec},
                    )
                    S.dup_version = S.get("dup_version", 0) + 1
                    st.rerun()
    if not dup["pairs"]:
        st.success("The queue is empty. Nice work.")
