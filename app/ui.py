"""CatalogIQ look and feel: brand CSS, header, chips, cards, animated counters, loaders and charts."""

# ruff: noqa: E501  (CSS and HTML templates)

from __future__ import annotations

import html
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from app import client

ASSETS = Path(__file__).parent / "assets"
INDIGO, VIOLET, CORAL, TEAL, AMBER, INK = (
    "#5B5BF0",
    "#8B5CF6",
    "#FF6B6B",
    "#14B8A6",
    "#F59E0B",
    "#1E1B4B",
)

CSS = f"""
<style>
:root {{ --indigo:{INDIGO}; --violet:{VIOLET}; --coral:{CORAL}; --teal:{TEAL}; --amber:{AMBER}; --ink:{INK}; }}
html, body, [class*="css"], .stApp {{ font-family: "Inter","Segoe UI",system-ui,-apple-system,sans-serif; }}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, #E9E7FF 0%, transparent 60%), radial-gradient(900px 400px at -10% 10%, #FFE9EE 0%, transparent 55%), #F6F7FF; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
@keyframes fadeUp {{ from {{ opacity:0; transform: translateY(14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes shift {{ 0% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} 100% {{ background-position:0% 50%; }} }}
@keyframes grow {{ from {{ width:0; }} }}
@keyframes ring {{ from {{ stroke-dashoffset: var(--full); }} }}
@keyframes pop {{ 0% {{ transform:scale(.6); opacity:0; }} 70% {{ transform:scale(1.08); }} 100% {{ transform:scale(1); opacity:1; }} }}
.ciq-hero {{ background: linear-gradient(120deg, {INDIGO}, {VIOLET}, {CORAL}); background-size: 220% 220%; animation: shift 14s ease infinite, fadeUp .6s ease both;
  border-radius: 26px; padding: 26px 32px; color:#fff; box-shadow: 0 18px 40px -18px rgba(91,91,240,.55); margin-bottom: 18px; position:relative; overflow:hidden; }}
.ciq-hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px; border-radius:50%; background:rgba(255,255,255,.12); }}
.ciq-hero .eyebrow {{ letter-spacing:.14em; text-transform:uppercase; font-size:.72rem; opacity:.85; font-weight:700; }}
.ciq-hero h1 {{ margin:.15rem 0 .3rem; font-size:2.1rem; line-height:1.15; color:#fff; font-weight:800; }}
.ciq-hero p {{ margin:0; font-size:1.02rem; opacity:.93; max-width: 760px; }}
.ciq-card {{ background:#fff; border:1px solid #ECEBFF; border-radius:20px; padding:18px 20px; box-shadow:0 8px 24px -14px rgba(30,27,75,.25); animation: fadeUp .5s ease both; transition: transform .2s ease, box-shadow .2s ease; margin-bottom:14px; }}
.ciq-card:hover {{ transform: translateY(-2px); box-shadow:0 16px 32px -16px rgba(91,91,240,.35); }}
.ciq-card h3, .ciq-card h4 {{ margin:0 0 .5rem; color:var(--ink); }}
.ciq-title {{ font-weight:700; color:var(--ink); line-height:1.25; }}
.ciq-sub {{ color:#6B6B8D; font-size:.85rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; border-radius:999px; padding:.22rem .7rem; font-size:.8rem; font-weight:600; margin:0 .3rem .3rem 0; border:1px solid transparent; animation: pop .35s ease both; }}
.chip.green {{ background:#E6F8F4; color:#0B7F70; border-color:#BFEDE5; }}
.chip.amber {{ background:#FFF4DB; color:#9A6200; border-color:#FFE2A3; }}
.chip.blue  {{ background:#E9E9FF; color:#3B3BC4; border-color:#CFCFFF; }}
.chip.red   {{ background:#FFE9EA; color:#B4232C; border-color:#FFC9CD; }}
.chip.grey  {{ background:#F0F0F7; color:#4B4B6B; border-color:#E1E1EE; }}
.chip.cite  {{ background:#EEF0FF; color:{INDIGO}; border-color:#D4D8FF; font-family: ui-monospace, Consolas, monospace; font-size:.74rem; padding:.05rem .45rem; }}
.attr-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#7A7A9C; font-weight:700; margin-bottom:.2rem; }}
.bar-row {{ display:flex; align-items:center; gap:.7rem; margin:.45rem 0; }}
.bar-name {{ width:170px; font-size:.86rem; font-weight:600; color:var(--ink); }}
.bar-track {{ flex:1; height:16px; border-radius:999px; background:#EEEDFB; overflow:hidden; display:flex; }}
.bar-fill {{ height:100%; animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both; }}
.bar-n {{ width:70px; text-align:right; font-size:.8rem; color:#6B6B8D; }}
.item-card {{ background:#fff; border:1px solid #ECEBFF; border-radius:18px; overflow:hidden; box-shadow:0 8px 20px -14px rgba(30,27,75,.3); transition: transform .2s ease, box-shadow .2s ease; animation: fadeUp .5s ease both; height:100%; }}
.item-card:hover {{ transform: translateY(-4px) scale(1.01); box-shadow:0 18px 34px -16px rgba(91,91,240,.45); }}
.item-card .photo {{ background:linear-gradient(180deg,#FAFAFF,#EFEFFB); display:flex; align-items:center; justify-content:center; height:190px; }}
.item-card img {{ max-height:176px; max-width:100%; image-rendering:auto; }}
.item-card .body {{ padding:10px 12px 12px; }}
.item-card .name {{ font-weight:700; font-size:.88rem; color:var(--ink); line-height:1.25; min-height:2.2em; }}
.flag {{ border-left:4px solid var(--amber); background:#FFFBF0; border-radius:10px; padding:.55rem .8rem; margin:.4rem 0; font-size:.9rem; }}
.flag.red {{ border-color:var(--coral); background:#FFF5F5; }}
.flag b {{ color:var(--ink); }}
.answer {{ background:linear-gradient(135deg,#F4F3FF,#FFF6F6); border:1px solid #E5E3FF; border-radius:18px; padding:14px 16px; }}
.answer.refused {{ background:#F7F7FB; border-style:dashed; color:#5B5B7B; }}
div[data-testid="stBaseButton-primary"] > button, button[data-testid="stBaseButton-primary"] {{ background: linear-gradient(120deg,{INDIGO},{VIOLET}); border:0; color:#fff; font-weight:700; border-radius:14px; padding:.55rem 1.2rem; box-shadow:0 10px 22px -10px rgba(91,91,240,.7); transition: transform .15s ease; }}
button[data-testid="stBaseButton-primary"]:hover {{ transform: translateY(-1px) scale(1.02); }}
button[data-testid="stBaseButton-secondary"] {{ border-radius:12px; border:1px solid #D9D8F5; background:#fff; font-weight:600; }}
button[data-baseweb="tab"] {{ font-weight:700; border-radius:12px 12px 0 0; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#FFFFFF,#F1F0FF); border-right:1px solid #E7E5FF; }}
[data-testid="stFileUploader"] section {{ border-radius:18px; border:2px dashed #CFCBFF; background:#FBFAFF; }}
.stepper {{ display:flex; gap:.5rem; margin:.2rem 0 1rem; flex-wrap:wrap; }}
.step {{ padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:700; background:#fff; border:1px solid #E1E0F7; color:#8A89A8; }}
.step.done {{ background:#E6F8F4; color:#0B7F70; border-color:#BFEDE5; }}
.step.now {{ background:linear-gradient(120deg,{INDIGO},{VIOLET}); color:#fff; border-color:transparent; box-shadow:0 8px 18px -8px rgba(91,91,240,.7); }}
.logo {{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.25rem; color:var(--ink); }}
.logo b {{ background: linear-gradient(120deg,{INDIGO},{CORAL}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.pill {{ display:inline-block; border-radius:999px; padding:.15rem .6rem; font-size:.72rem; font-weight:700; }}
.pill.on {{ background:#E6F8F4; color:#0B7F70; }} .pill.off {{ background:#FFE9EA; color:#B4232C; }}
</style>
"""

LOGO_SVG = f"""<svg width="38" height="38" viewBox="0 0 38 38" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{INDIGO}"/><stop offset="1" stop-color="{CORAL}"/></linearGradient></defs><rect width="38" height="38" rx="11" fill="url(#g)"/><path d="M10 13.5 19 9l9 4.5v11L19 29l-9-4.5z" fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/><path d="M15 19.2l3.1 3.1L24 16" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def setup(title: str, icon: str = "✨", sidebar_extra: bool = True) -> None:
    """Page config, brand CSS and the sidebar (logo, API status, demo-mode switch)."""
    st.set_page_config(
        page_title=f"{title} · CatalogIQ",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("demo_mode", True)
    with st.sidebar:
        st.markdown(
            f'<div class="logo">{LOGO_SVG}<span>Catalog<b>IQ</b></span></div><div class="ciq-sub" style="margin:.2rem 0 .8rem">AI copilot for marketplace catalogs</div>',
            unsafe_allow_html=True,
        )
        try:
            h = client.health()
            ok = h["status"] == "ok"
            st.markdown(
                f'<span class="pill {"on" if ok else "off"}">● API {"online" if ok else "degraded"}</span> '
                f'<span class="pill {"on" if h["llm_reachable"] else "off"}">● {h["llm_model"]}</span>',
                unsafe_allow_html=True,
            )
        except (client.ApiUnavailable, client.ApiError):
            st.markdown('<span class="pill off">● API offline</span>', unsafe_allow_html=True)
        if sidebar_extra:
            st.toggle(
                "Demo mode",
                key="demo_mode",
                help="Uses 10 preloaded sample products whose listings and answers are cached, so everything is instant.",
            )
            st.caption(
                "Everything runs locally on CPU. New listings and answers take seconds to a minute; demo products are instant."
            )


def require_api() -> bool:
    """Show a friendly message and stop when the API cannot be reached."""
    try:
        client.health()
        return True
    except client.ApiUnavailable as exc:
        st.error(str(exc))
        st.code("python -m uvicorn api.main:app --port 8000", language="bash")
        st.stop()
    except client.ApiError as exc:
        st.error(exc.message)
        st.stop()
    return False


def hero(title: str, subtitle: str, eyebrow: str = "CatalogIQ") -> None:
    """Gradient header with the page title."""
    st.markdown(
        f'<div class="ciq-hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def stepper(steps: list[str], current: int) -> None:
    """Progress chips: done / now / upcoming."""
    parts = []
    for i, s in enumerate(steps):
        cls = "done" if i < current else "now" if i == current else ""
        parts.append(
            f'<span class="step {cls}">{"✓ " if i < current else f"{i + 1} "}{html.escape(s)}</span>'
        )
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "grey") -> str:
    """HTML of one coloured chip."""
    return f'<span class="chip {kind}">{html.escape(str(text))}</span>'


def chips(items: list[tuple[str, str]]) -> None:
    """A row of (text, kind) chips."""
    st.markdown("".join(chip(t, k) for t, k in items), unsafe_allow_html=True)


def card(inner_html: str) -> None:
    """A white rounded card around raw HTML."""
    st.markdown(f'<div class="ciq-card">{inner_html}</div>', unsafe_allow_html=True)


def item_card_html(
    item: dict[str, Any], badge: str | None = None, caption: str | None = None
) -> str:
    """HTML of a product card (photo served by the API)."""
    meta = " · ".join(
        str(item.get(k)) for k in ("articleType", "baseColour", "gender") if item.get(k)
    )
    badge_html = f'<div style="margin-bottom:.25rem">{badge}</div>' if badge else ""
    cap = (
        f'<div class="ciq-sub" style="margin-top:.25rem">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    return (
        f'<div class="item-card"><div class="photo"><img src="{client.url(item["image_url"])}" alt="{html.escape(item["title"])}"></div>'
        f'<div class="body">{badge_html}<div class="name">{html.escape(item["title"])}</div><div class="ciq-sub">{html.escape(meta)}</div>{cap}</div></div>'
    )


def counter(
    label: str,
    value: float,
    suffix: str = "",
    decimals: int = 0,
    color: str = INDIGO,
    height: int = 118,
) -> None:
    """A KPI tile whose number counts up (runs in a small iframe because it needs JavaScript)."""
    doc = f"""<div style="font-family:'Segoe UI',system-ui,sans-serif;background:#fff;border:1px solid #ECEBFF;border-radius:20px;padding:16px 18px;height:{height - 12}px;box-sizing:border-box;box-shadow:0 8px 24px -14px rgba(30,27,75,.25);overflow:hidden">
    <div style="font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:#7A7A9C;font-weight:700">{html.escape(label)}</div>
    <div style="font-size:2.35rem;font-weight:800;color:{color};margin-top:.15rem;line-height:1.1"><span id="n">0</span><span style="font-size:1.1rem;color:#7A7A9C;margin-left:.2rem">{html.escape(suffix)}</span></div></div>
    <script>const t={json.dumps(value)},d={decimals},el=document.getElementById('n'),t0=performance.now();
    function f(now){{const p=Math.min(1,(now-t0)/1100),e=1-Math.pow(1-p,3);el.textContent=(t*e).toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}});if(p<1)requestAnimationFrame(f);}}requestAnimationFrame(f);</script>"""
    components.html(doc, height=height)


def ring_gauge(score: float, label: str = "quality score", size: int = 170) -> str:
    """HTML of an animated circular gauge (the ring draws itself)."""
    r = 62
    circ = 2 * 3.14159 * r
    colour = TEAL if score >= 85 else AMBER if score >= 70 else CORAL
    off = circ * (1 - score / 100)
    return (
        f'<div style="text-align:center"><svg width="{size}" height="{size}" viewBox="0 0 160 160"><circle cx="80" cy="80" r="{r}" fill="none" stroke="#EEEDFB" stroke-width="14"/>'
        f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="{colour}" stroke-width="14" stroke-linecap="round" transform="rotate(-90 80 80)" '
        f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}" style="--full:{circ:.1f};animation:ring 1.3s cubic-bezier(.2,.8,.2,1) both"/>'
        f'<text x="80" y="82" text-anchor="middle" font-size="38" font-weight="800" fill="{INK}">{score:.0f}</text>'
        f'<text x="80" y="104" text-anchor="middle" font-size="11" fill="#7A7A9C" font-weight="700" letter-spacing="1">{html.escape(label.upper())}</text></svg></div>'
    )


def stacked_bar(name: str, pos: float, mixed: float, neg: float, mentions: int) -> str:
    """HTML of one aspect row: share of positive / mixed / negative mentions."""

    def seg(w: float, c: str) -> str:
        return f'<div class="bar-fill" style="width:{w}%;background:{c}"></div>'

    return (
        f'<div class="bar-row"><div class="bar-name">{html.escape(name)}</div><div class="bar-track">{seg(pos, TEAL)}{seg(mixed, "#C9C8EA")}{seg(neg, CORAL)}</div>'
        f'<div class="bar-n">{mentions} reviews</div></div>'
    )


def lottie(name: str, height: int = 150, key: str | None = None) -> None:
    """Play one of the local Lottie animations (loader, scan, success)."""
    from streamlit_lottie import st_lottie

    st_lottie(
        json.loads((ASSETS / f"{name}.json").read_text(encoding="utf-8")),
        height=height,
        key=key or f"lottie_{name}",
        speed=1,
        loop=name != "success",
    )


@contextmanager
def working(message: str, note: str = "", animation: str = "loader") -> Iterator[None]:
    """Show an animated loader with a message while a block runs, then clear it."""
    box = st.empty()
    with box.container():
        c1, c2 = st.columns([1, 3])
        with c1:
            lottie(animation, 120)
        with c2:
            st.markdown(
                f'<div style="padding-top:1.6rem"><div class="ciq-title" style="font-size:1.1rem">{html.escape(message)}</div><div class="ciq-sub">{html.escape(note)}</div></div>',
                unsafe_allow_html=True,
            )
    try:
        yield
    finally:
        box.empty()


def flag_html(message: str, fix: str, severe: bool = False) -> str:
    """HTML of an issue with its plain-English fix."""
    return f'<div class="flag {"red" if severe else ""}"><b>{html.escape(message)}</b><br><span class="ciq-sub">💡 {html.escape(fix)}</span></div>'
