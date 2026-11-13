"""Seller Studio: photo -> attributes (editable) -> listing -> quality score with fixes -> similar listings."""

# ruff: noqa: E501  (page copy)

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from typing import Any

import streamlit as st

from app import client, ui

ui.setup("Seller Studio", "📸")
ui.require_api()
S = st.session_state
ATTRS = ["masterCategory", "subCategory", "articleType", "baseColour", "gender", "season", "usage"]
NICE = {
    "masterCategory": "Department",
    "subCategory": "Sub-category",
    "articleType": "Product type",
    "baseColour": "Colour",
    "gender": "For",
    "season": "Season",
    "usage": "Occasion",
}
MODES = {
    "Safe and factual (recommended)": "no_rag_fix",
    "Style-guided with examples (may invent facts)": "rag",
}
RESET = ("studio_pred", "studio_listing", "studio_quality", "studio_similar", "studio_error")


def choose(
    photo: bytes, name: str, ctype: str, source: str, demo: dict[str, Any] | None = None
) -> None:
    """Set the working photo; reset everything computed from a previous one."""
    key = hashlib.md5(photo).hexdigest()[:12]
    if S.get("studio_key") == key:
        return
    for k in RESET:
        S.pop(k, None)
    S.studio_key = key
    S.studio_photo = {"bytes": photo, "name": name, "ctype": ctype, "source": source, "demo": demo}
    return True


ui.hero(
    "Seller Studio",
    "Snap a photo, confirm what we see, and get a listing you can publish after a quick check, with a quality score and fix tips.",
    "For sellers",
)

# ------------------------------------------------------------------------------------ 1. photo
demo = client.demo_products()
qp = st.query_params
if qp.get("demo") and not S.get("studio_photo"):
    d = next((x for x in demo if str(x["id"]) == qp["demo"]), None)
    if d:
        choose(client.image_bytes(d["image_url"]), f"{d['id']}.jpg", "image/jpeg", "demo", d)

with st.expander("📷 Choose or change the photo", expanded=not S.get("studio_photo")):
    tabs = st.tabs(
        (["⚡ Demo products"] if S.demo_mode else []) + ["⬆️ Upload a photo", "📷 Take a photo"]
    )
    i = 0
    if S.demo_mode:
        with tabs[0]:
            st.caption(
                "Ten sample products from the held-out test set. Their listings are cached, so generating is instant."
            )
            for row in (demo[:5], demo[5:10]):
                cols = st.columns(5)
                for col, d in zip(cols, row, strict=False):
                    with col:
                        st.markdown(
                            ui.item_card_html(
                                {
                                    "title": d["title"],
                                    "image_url": d["image_url"],
                                    "articleType": d["attributes"]["articleType"]["label"],
                                    "baseColour": d["attributes"]["baseColour"]["label"],
                                }
                            ),
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Use this photo", key=f"demo_{d['id']}", use_container_width=True
                        ):
                            choose(
                                client.image_bytes(d["image_url"]),
                                f"{d['id']}.jpg",
                                "image/jpeg",
                                "demo",
                                d,
                            )
                            st.rerun()
        i = 1
    with tabs[i]:
        up = st.file_uploader(
            "Drop a product photo (JPEG, PNG or WebP, up to 5 MB)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if up is not None and choose(up.getvalue(), up.name, up.type or "image/jpeg", "upload"):
            st.rerun()
    with tabs[i + 1]:
        if st.toggle(
            "Turn on the camera", key="cam_on", help="The browser will ask for permission."
        ):
            cam = st.camera_input(
                "Take a photo of the product on a plain background", label_visibility="collapsed"
            )
            if cam is not None and choose(cam.getvalue(), "camera.jpg", "image/jpeg", "camera"):
                st.rerun()

photo = S.get("studio_photo")
if not photo:
    c1, c2 = st.columns([1, 2])
    with c1:
        ui.lottie("scan", 200)
    with c2:
        ui.card(
            '<h3>Start with a photo</h3><div class="ciq-sub">Use a demo product, upload an image or take a picture. We will read the product type, colour, gender, season and occasion from it, and you stay in control of every value.</div>'
        )
    st.stop()

# ----------------------------------------------------------------------------- 2. attributes
if "studio_pred" not in S:
    try:
        with ui.working("Reading your photo…", "Classifier on CPU, about a second", "scan"):
            S.studio_pred = client.post_photo(
                "/predict-attributes", photo["bytes"], photo["name"], photo["ctype"]
            )
    except client.ApiError as exc:
        st.error(f"{exc.message}")
        st.stop()
pred = S.studio_pred["attributes"]
opts = client.options()["attribute_labels"]
demo_attrs = (photo["demo"] or {}).get("attributes")
key = S.studio_key

# similar listings (fast): once per photo
if "studio_similar" not in S:
    try:
        S.studio_similar = client.post_photo(
            "/search/image",
            photo["bytes"],
            photo["name"],
            photo["ctype"],
            data={"k": 6, "exclude_id": (photo["demo"] or {}).get("id")},
        )
    except client.ApiError as exc:
        S.studio_similar = {"results": [], "error": exc.message}

ui.stepper(
    ["Photo", "Attributes", "Listing", "Quality", "Publish"], 4 if "studio_quality" in S else 1
)

left, right = st.columns([1, 2.2], gap="large")
with left:
    st.markdown(
        f'<div class="ciq-card" style="text-align:center"><img src="data:{photo["ctype"]};base64,{__import__("base64").b64encode(photo["bytes"]).decode()}" style="max-width:100%;max-height:340px;border-radius:14px"><div class="ciq-sub" style="margin-top:.5rem">{html.escape(photo["name"])} · {S.studio_pred["image"]["width"]}×{S.studio_pred["image"]["height"]} px</div></div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown("#### 1 · Confirm the attributes")
    st.caption(
        "Green = confident. Amber = please confirm. Change any value and it becomes yours (blue)."
    )
    chosen: dict[str, dict[str, Any]] = {}
    cols = st.columns(3)
    for n, attr in enumerate(ATTRS):
        p = pred[attr]
        with cols[n % 3]:
            st.markdown(f'<div class="attr-label">{NICE[attr]}</div>', unsafe_allow_html=True)
            labels = opts[attr]
            idx = labels.index(p["label"]) if p["label"] in labels else 0
            val = st.selectbox(
                NICE[attr],
                labels,
                index=idx,
                key=f"attr_{key}_{attr}",
                label_visibility="collapsed",
            )
            edited = val != p["label"]
            if edited:
                st.markdown(ui.chip("confirmed by you", "blue"), unsafe_allow_html=True)
                chosen[attr] = {"label": val, "confidence": 1.0, "needs_review": False}
            else:
                kind = "amber" if p["needs_review"] else "green"
                text = (
                    f"please confirm · {p['confidence']:.0%}"
                    if p["needs_review"]
                    else f"confident · {p['confidence']:.0%}"
                )
                st.markdown(ui.chip(text, kind), unsafe_allow_html=True)
                chosen[attr] = {
                    "label": val,
                    "confidence": p["confidence"],
                    "needs_review": p["needs_review"],
                }
    if demo_attrs and all(chosen[a]["label"] == demo_attrs[a]["label"] for a in ATTRS):
        chosen = {
            a: demo_attrs[a] for a in ATTRS
        }  # exact stored values: the cached listing is found instantly

    st.markdown("#### 2 · Anything the photo cannot show?")
    st.caption(
        "Facts you type here are allowed in the listing. Anything else the writer states about materials, sizes or claims is flagged."
    )
    n1, n2, n3 = st.columns(3)
    notes_default = (photo["demo"] or {}).get("seller_notes", {})
    notes = {
        "brand": n1.text_input("Brand", value=notes_default.get("brand", ""), key=f"n_brand_{key}"),
        "material": n2.text_input("Material", placeholder="e.g. 100% cotton", key=f"n_mat_{key}"),
        "size": n3.text_input("Size / fit", placeholder="e.g. M, regular fit", key=f"n_size_{key}"),
    }
    notes = {k: v.strip() for k, v in notes.items() if v.strip()}

    st.markdown("#### 3 · Write the listing")
    mode_label = st.radio(
        "Writer", list(MODES), horizontal=True, label_visibility="collapsed", key=f"mode_{key}"
    )
    if MODES[mode_label] == "rag":
        st.warning(
            "In our test, style-guided drafts contained an invented fact in 70% of listings. Always review them.",
            icon="⚠️",
        )
    go = st.button("✨ Generate listing", type="primary", key=f"go_{key}")
    if st.query_params.get("autorun") and "studio_listing" not in S and "autorun_done" not in S:
        S.autorun_done = True
        go = True
    if go:
        try:
            note = "Instant for demo products (cached). New products take about 15 s, style-guided about a minute, on this CPU."
            with ui.working("Writing and fact-checking your listing…", note):
                S.studio_listing = client.post_json(
                    "/generate-listing",
                    {"attributes": chosen, "seller_notes": notes, "mode": MODES[mode_label]},
                    client.SLOW,
                )
            S.studio_payload = {"attributes": chosen, "notes": notes}
            stated = {
                a: chosen[a]["label"] for a in ("articleType", "baseColour", "gender", "usage")
            }
            title = (S.studio_listing.get("listing") or {}).get("title")
            S.studio_quality = client.post_photo(
                "/quality-check",
                photo["bytes"],
                photo["name"],
                photo["ctype"],
                data={"attributes": json.dumps(stated), "title": title},
            )
            st.rerun()
        except client.ApiError as exc:
            S.studio_error = exc.message
    if S.get("studio_error"):
        st.error(S.studio_error)

# ------------------------------------------------------------------------------------ results
res = S.get("studio_listing")
if res and res.get("listing"):
    L, G = res["listing"], res["guardrails"]
    st.divider()
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        st.markdown("#### Your listing")
        bullets = "".join(f"<li>{html.escape(x)}</li>" for x in L["bullets"])
        tags = "".join(ui.chip(t, "blue") for t in L["search_tags"])
        st.markdown(
            f'<div class="ciq-card"><div class="ciq-sub">Title</div><div style="font-size:1.35rem;font-weight:800;color:{ui.INK};margin-bottom:.6rem">{html.escape(L["title"])}</div>'
            f'<div class="ciq-sub">Highlights</div><ul style="margin:.2rem 0 .8rem 1.1rem">{bullets}</ul><div class="ciq-sub">Description</div><div style="margin:.2rem 0 .8rem">{html.escape(L["description"])}</div>'
            f'<div class="ciq-sub" style="margin-bottom:.3rem">Search tags</div>{tags}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Written in {res['elapsed_s']:.1f} s here (the model originally took {res['generation_s']:.0f} s) · pipeline: {res['mode']} · category: {res['group']}"
        )
    with b:
        st.markdown("#### Fact check")
        problems = []
        if G["hallucinated"]:
            for u in G["unsupported"]:
                problems.append(
                    ui.flag_html(
                        f"Not in your inputs: {u['kind']} “{u['term']}”",
                        "Remove it from the listing, or type it under “Anything the photo cannot show?” if it is true.",
                        True,
                    )
                )
        if G["banned"]:
            problems.append(
                ui.flag_html(
                    "Banned or risky wording: " + ", ".join(G["banned"]),
                    "Replace it with a plain description of the product.",
                )
            )
        for msg in G.get("style_messages") or G["style_violations"]:
            problems.append(ui.flag_html(msg, "Adjust the wording to the marketplace style guide."))
        if G["missing_attributes"]:
            problems.append(
                ui.flag_html(
                    "Attributes not used: " + ", ".join(G["missing_attributes"]),
                    "Mention them or confirm they do not apply.",
                )
            )
        for w in L.get("warnings", [])[:3]:
            problems.append(ui.flag_html(w, "Check this before publishing."))
        if problems:
            st.markdown("".join(problems), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                ui.lottie("success", 90)
            with c2:
                st.markdown(
                    '<div style="padding-top:1.2rem"><b>All guardrails passed.</b><br><span class="ciq-sub">No invented facts, no risky wording.</span></div>',
                    unsafe_allow_html=True,
                )
        st.progress(
            min(1.0, G["coverage"]),
            text=f"{G['coverage']:.0%} of your attributes are used in the listing",
        )

q = S.get("studio_quality")
if q:
    st.markdown("#### Quality score")
    g1, g2 = st.columns([1, 2.4], gap="large")
    with g1:
        st.markdown(
            f'<div class="ciq-card">{ui.ring_gauge(q["score"])}</div>', unsafe_allow_html=True
        )
        comps = q["components"]
        st.markdown(
            "".join(
                f'<div class="bar-row"><div class="bar-name" style="width:110px">{n.title()}</div><div class="bar-track"><div class="bar-fill" style="width:{v}%;background:{ui.TEAL if v >= 85 else ui.AMBER if v >= 60 else ui.CORAL}"></div></div><div class="bar-n" style="width:40px">{v:.0f}</div></div>'
                for n, v in comps.items()
            ),
            unsafe_allow_html=True,
        )
    with g2:
        if q["flags"]:
            st.markdown(
                "".join(
                    ui.flag_html(f["message"].capitalize(), f["fix"], f["area"] == "consistency")
                    for f in q["flags"]
                ),
                unsafe_allow_html=True,
            )
        else:
            st.success("Great photo and listing: nothing to fix.")
        st.caption(
            "Photo checks: blur, light, background clutter and overlaid text; photo-vs-listing consistency; completeness. Thresholds were tuned on a 60×80 px studio catalog, so phone photos may behave differently."
        )

sim = S.get("studio_similar", {}).get("results", [])
if sim:
    st.markdown("#### Similar existing listings")
    cols = st.columns(len(sim))
    for col, it in zip(cols, sim, strict=True):
        with col:
            st.markdown(
                ui.item_card_html(it, caption=f"{it['score']:.0%} similar"), unsafe_allow_html=True
            )

if res and res.get("listing"):
    st.divider()
    payload = {
        "attributes": S.get("studio_payload", {}).get("attributes"),
        "seller_notes": S.get("studio_payload", {}).get("notes"),
        "listing": res["listing"],
        "guardrails": res["guardrails"],
        "quality": q,
    }
    flat = {
        "title": res["listing"]["title"],
        "bullet_1": "",
        "description": res["listing"]["description"],
        "search_tags": "; ".join(res["listing"]["search_tags"]),
        "quality_score": q["score"] if q else "",
    }
    for n, b in enumerate(res["listing"]["bullets"], start=1):
        flat[f"bullet_{n}"] = b
    for a in ATTRS:
        flat[a] = (S.get("studio_payload", {}).get("attributes") or {}).get(a, {}).get("label", "")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(flat))
    w.writeheader()
    w.writerow(flat)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button(
        "⬇️ Download JSON",
        json.dumps(payload, indent=2),
        "listing.json",
        "application/json",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Download CSV", buf.getvalue(), "listing.csv", "text/csv", use_container_width=True
    )
