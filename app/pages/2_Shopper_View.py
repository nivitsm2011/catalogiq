"""Shopper View: search by words or photo, outfits that go together, and review Q&A with citations."""

# ruff: noqa: E501  (page copy)

from __future__ import annotations

import html
import json
import re
from typing import Any

import streamlit as st

from app import client, ui

ui.setup("Shopper View", "🛍️")
ui.require_api()
S = st.session_state
ui.hero(
    "Shopper View",
    "Find it by describing it, see what goes with it, and ask real reviews your questions. Answers always cite the reviews they come from.",
    "For shoppers",
)
tab_find, tab_reviews = st.tabs(["🔎 Discover products", "💬 Reviews & questions"])
opts = client.options()["filters"]


def any_of(label: str, key: str, values: list[str]) -> str | None:
    """A filter dropdown with an 'Any' entry."""
    v = st.selectbox(label, ["Any"] + values, key=key)
    return None if v == "Any" else v


# ------------------------------------------------------------------------------------- discover
with tab_find:
    qp = st.query_params
    mode = st.radio(
        "Search by",
        ["Describe it", "Upload a photo"],
        horizontal=True,
        label_visibility="collapsed",
        key="shop_mode",
    )
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        gender = any_of("For", "f_gender", opts["gender"])
    with f2:
        colour = any_of("Colour", "f_colour", opts["baseColour"])
    with f3:
        atype = any_of("Product type", "f_type", opts["articleType"])
    with f4:
        usage = any_of("Occasion", "f_usage", opts["usage"])
    filters = {
        k: v
        for k, v in {
            "gender": gender,
            "baseColour": colour,
            "articleType": atype,
            "usage": usage,
        }.items()
        if v
    }

    results: dict[str, Any] | None = None
    if mode == "Describe it":
        examples = [
            "black formal shoes for men",
            "red dress for women",
            "white sports shoes",
            "blue denim jeans",
        ]
        S.setdefault("shop_q", qp.get("q", ""))
        c1, c2 = st.columns([4, 1])
        with c1:
            q = st.text_input(
                "What are you looking for?",
                key="shop_q",
                placeholder="e.g. black formal shoes for men",
                label_visibility="collapsed",
            )
        with c2:
            st.button("Search", type="primary", use_container_width=True)
        ex = st.columns(len(examples))
        for col, e in zip(ex, examples, strict=True):
            if col.button(e, key=f"ex_{e}", use_container_width=True):
                S.shop_q = e
                st.rerun()
        if q.strip() and len(q.strip()) >= 2:
            try:
                with st.spinner("Searching…"):
                    results = client.search_text(q.strip(), 10, json.dumps(filters))
            except client.ApiError as exc:
                st.error(exc.message)
    else:
        up = st.file_uploader(
            "Upload a photo to find look-alikes", type=["jpg", "jpeg", "png", "webp"], key="shop_up"
        )
        if up is not None:
            try:
                with ui.working(
                    "Finding look-alikes…", "CLIP embeddings over 42,257 products", "scan"
                ):
                    results = client.post_photo(
                        "/search/image",
                        up.getvalue(),
                        up.name,
                        up.type or "image/jpeg",
                        data={"k": 10, **filters},
                    )
            except client.ApiError as exc:
                st.error(exc.message)

    if results:
        st.markdown(f"##### {len(results['results'])} results · {results['latency_ms']:.0f} ms")
        items = results["results"]
        for start in range(0, len(items), 5):
            cols = st.columns(5)
            for col, it in zip(cols, items[start : start + 5], strict=False):
                with col:
                    caption = f"{it['score']:.0%} similar" if mode == "Upload a photo" else None
                    st.markdown(ui.item_card_html(it, caption=caption), unsafe_allow_html=True)
                    if st.button("View", key=f"view_{it['id']}", use_container_width=True):
                        S.shop_item = it
                        st.rerun()
    elif mode == "Describe it" and not q.strip():
        ui.card(
            '<h3>What are you shopping for?</h3><div class="ciq-sub">Describe it in your own words, or pick one of the examples above. Filters narrow the results.</div>'
        )

    item = S.get("shop_item")
    if item:
        st.divider()
        st.markdown("#### Product")
        a, b = st.columns([1, 2.4], gap="large")
        with a:
            st.markdown(ui.item_card_html(item), unsafe_allow_html=True)
        with b:
            ui.chips(
                [
                    (item[k], c)
                    for k, c in (
                        ("articleType", "blue"),
                        ("baseColour", "grey"),
                        ("gender", "grey"),
                        ("usage", "green"),
                        ("season", "grey"),
                    )
                    if item.get(k)
                ]
            )
            try:
                with ui.working(
                    "Styling an outfit…",
                    "Complementary items that match the product type, colour, occasion and season",
                ):
                    look = client.post_form("/shop-the-look", {"item_id": item["id"]}, 60)
            except client.ApiError as exc:
                look = {"items": [], "error": {"message": exc.message}}
            roles = {}
            for it in look.get("items", []):
                roles.setdefault(it.get("role") or "other", []).append(it)
            if look.get("error"):
                st.error(look["error"]["message"])
            elif not roles:
                st.info(look.get("note") or "Nothing to pair this with.")
            else:
                st.markdown("##### Shop the look")
                for role, rows in roles.items():
                    st.markdown(
                        f'<div class="attr-label">{html.escape(role)}</div>', unsafe_allow_html=True
                    )
                    cols = st.columns(4)
                    for col, it in zip(cols, rows[:4], strict=False):
                        with col:
                            st.markdown(
                                ui.item_card_html(it, caption=it.get("reasons")),
                                unsafe_allow_html=True,
                            )

# --------------------------------------------------------------------------------- reviews + Q&A
with tab_reviews:
    ui.card(
        '<b>About this tab.</b> <span class="ciq-sub">Reviews come from a separate Amazon Fashion sample (139,400 reviews of 800 products), not from the photo catalog on the other tab. Products marked ⚡ have instant demo answers; anything else is answered live and can take up to a minute on this CPU.</span>'
    )
    products = client.review_products()
    pool = [p for p in products if p["demo"]] if S.demo_mode else products
    labels = {
        p["item_id"]: f"{'⚡ ' if p['demo'] else ''}{p['title'][:80]} · {p['n_reviews']} reviews"
        for p in pool
    }
    default = (
        list(labels).index(st.query_params["product"])
        if st.query_params.get("product") in labels
        else 0
    )
    item_id = st.selectbox(
        "Choose a product", list(labels), format_func=labels.get, index=default, key="rev_product"
    )
    prod = next(p for p in products if p["item_id"] == item_id)

    st.markdown(f"#### {html.escape(prod['title'])}", unsafe_allow_html=True)
    ui.chips(
        [
            (
                (
                    f"★ {prod['average_rating']:.1f} average"
                    if prod.get("average_rating")
                    else "reviewed"
                ),
                "amber",
            ),
            (f"{prod['n_reviews']} reviews", "grey"),
        ]
        + ([("instant demo answers", "green")] if prod["demo"] else [])
    )
    try:
        summary = client.review_summary(item_id, False)
    except client.ApiError as exc:
        st.error(exc.message)
        st.stop()
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("##### What reviewers talk about")
        rows = "".join(
            ui.stacked_bar(
                summary["labels"][a],
                v["pct_positive"],
                v["pct_mixed"],
                v["pct_negative"],
                v["mentions"],
            )
            for a, v in summary["aspects"].items()
        )
        st.markdown(
            f'<div class="ciq-card">{rows or "<div class=ciq-sub>Not enough reviews mention any topic.</div>"}<div style="margin-top:.4rem">{ui.chip("positive (4-5★)", "green")}{ui.chip("mixed (3★)", "grey")}{ui.chip("negative (1-2★)", "red")}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Percentages are computed from star ratings of reviews that mention the topic, not generated by the model."
        )
    with right:
        st.markdown("##### Summary")
        text = summary["summary"]
        if S.get(f"phrase_{item_id}"):
            try:
                with ui.working("Phrasing the summary…", "Numbers stay exactly as computed"):
                    text = client.review_summary(item_id, True)["summary"]
            except client.ApiError as exc:
                st.warning(exc.message)
        st.markdown(
            f'<div class="ciq-card" style="font-size:.9rem">{html.escape(text)}</div>',
            unsafe_allow_html=True,
        )
        if not S.get(f"phrase_{item_id}") and st.button("✨ Phrase with AI", key=f"ph_{item_id}"):
            S[f"phrase_{item_id}"] = True
            st.rerun()

    st.markdown("##### Ask the reviews")
    chat = S.setdefault("chat", {}).setdefault(item_id, [])
    asked: str | None = None
    if prod["suggested_questions"]:
        st.caption("Try one:")
        cols = st.columns(min(len(prod["suggested_questions"]), 4))
        for n, (col, sq) in enumerate(zip(cols, prod["suggested_questions"], strict=False)):
            if col.button(sq, key=f"sq_{item_id}_{n}", use_container_width=True):
                asked = sq
    typed = st.chat_input("Ask about fit, material, colour, quality, comfort or delivery…")
    asked = typed or asked
    if asked:
        chat.append({"role": "user", "text": asked})
        try:
            with ui.working(
                "Reading the reviews…", "Retrieving evidence, then writing a cited answer"
            ):
                ans = client.post_json(f"/ask-reviews/{item_id}", {"question": asked}, client.SLOW)
            chat.append({"role": "assistant", "ans": ans})
        except client.ApiError as exc:
            chat.append({"role": "assistant", "error": exc.message})

    for msg in chat:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["text"])
            continue
        with st.chat_message("assistant", avatar="✨"):
            if msg.get("error"):
                ui.error_box(msg["error"])
                continue
            a = msg["ans"]
            body = re.sub(
                r"\[(R\d+)\]",
                lambda m: f'<span class="chip cite">{m.group(1)}</span>',
                html.escape(a["answer"]),
            )
            st.markdown(
                f'<div class="answer {"refused" if a["refused"] else ""}">{body}</div>',
                unsafe_allow_html=True,
            )
            if a["refused"]:
                ui.chips(
                    [
                        ("not enough evidence", "grey"),
                        (f"best match {a['top_similarity']:.0%}", "grey"),
                    ]
                )
            else:
                ui.chips(
                    [
                        (
                            f"{a['confidence']:.0%} confidence",
                            "green" if a["confidence"] >= 0.7 else "amber",
                        ),
                        (f"{a['evidence_count']} reviews used", "blue"),
                        (a["aspect"].replace("_", " "), "grey"),
                        (f"{a['elapsed_s']:.1f}s", "grey"),
                    ]
                )
                if a["reviewers_agree"] or a["reviewers_disagree"]:
                    ag = re.sub(
                        r"\[(R\d+)\]",
                        lambda m: f'<span class="chip cite">{m.group(1)}</span>',
                        html.escape(a["reviewers_agree"]),
                    )
                    dis = re.sub(
                        r"\[(R\d+)\]",
                        lambda m: f'<span class="chip cite">{m.group(1)}</span>',
                        html.escape(a["reviewers_disagree"] or "—"),
                    )
                    st.markdown(
                        f'<div class="ciq-sub"><b>Reviewers agree:</b> {ag}<br><b>Reviewers disagree:</b> {dis}</div>',
                        unsafe_allow_html=True,
                    )
                cols = st.columns(min(max(len(a["citations"]), 1), 4))
                for col, c in zip(cols, a["citations"], strict=False):
                    with col.popover(
                        f"{c['review_id']} · {int(c['rating'] or 0)}★"
                        + (" ✔" if c.get("verified_purchase") else ""),
                        use_container_width=True,
                    ):
                        st.markdown(
                            f"**Review {c['review_id']}** · {'verified purchase · ' if c.get('verified_purchase') else ''}{c.get('helpful_vote') or 0} found helpful"
                        )
                        st.write(c["text"])
    if chat and st.button("Clear conversation", key=f"clear_{item_id}"):
        S.chat[item_id] = []
        st.rerun()
