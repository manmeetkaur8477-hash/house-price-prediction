"""
ui_styles.py - the look and feel of the dashboard (CSS + small HTML helpers).

Streamlit lets us inject custom CSS. The functions below return small pieces
of HTML (hero banner, KPI cards, notes, price card) so app.py stays tidy.
"""

import html

import streamlit as st

from src import config

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4 {{ font-family: 'Poppins', sans-serif !important; color: {config.NAVY}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }}

/* ---------- hero banner ---------- */
.hero {{
    background: linear-gradient(120deg, {config.NAVY} 0%, #12406B 55%, {config.TEAL_DARK} 100%);
    border-radius: 20px; padding: 34px 38px; color: #fff; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(11, 37, 69, 0.22);
}}
.hero h1 {{ color: #fff !important; font-size: 2.0rem; margin: 0 0 6px 0; line-height: 1.25; }}
.hero p {{ color: #D6E4F0; margin: 0 0 14px 0; font-size: 1.02rem; max-width: 820px; }}
.badge {{
    display: inline-block; padding: 5px 13px; margin: 0 8px 6px 0; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
    color: #fff; font-size: 0.78rem; font-weight: 500;
}}

/* ---------- cards ---------- */
.kpi {{
    background: {config.CARD}; border-radius: 16px; padding: 18px 20px;
    border: 1px solid #E3EAF1; border-top: 4px solid var(--accent, {config.TEAL});
    box-shadow: 0 4px 14px rgba(11, 37, 69, 0.06); height: 100%;
}}
.kpi .label {{ color: {config.MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.kpi .value {{ font-family: 'Poppins', sans-serif; color: {config.NAVY}; font-size: 1.7rem; font-weight: 600; margin: 4px 0 2px 0; }}
.kpi .note {{ color: {config.MUTED}; font-size: 0.82rem; }}

.section-title {{ font-family: 'Poppins', sans-serif; color: {config.NAVY}; font-size: 1.25rem; font-weight: 600; margin: 26px 0 2px 0; }}
.section-sub {{ color: {config.MUTED}; font-size: 0.92rem; margin-bottom: 12px; }}

.callout {{
    border-radius: 12px; padding: 13px 17px; margin: 10px 0 14px 0; font-size: 0.92rem;
    border-left: 5px solid {config.TEAL}; background: #E8F6F5; color: {config.INK};
}}
.callout.warn {{ border-left-color: {config.GOLD}; background: #FFF6E5; }}
.callout.alert {{ border-left-color: {config.CORAL}; background: #FDECE7; }}

.step {{
    background: {config.CARD}; border: 1px solid #E3EAF1; border-radius: 14px;
    padding: 16px 14px; text-align: center; height: 100%;
    box-shadow: 0 3px 10px rgba(11, 37, 69, 0.05);
}}
.step .num {{
    width: 32px; height: 32px; line-height: 32px; border-radius: 50%; margin: 0 auto 8px auto;
    background: {config.TEAL}; color: #fff; font-weight: 700; font-family: 'Poppins', sans-serif;
}}
.step .title {{ font-weight: 600; color: {config.NAVY}; font-size: 0.92rem; }}
.step .text {{ color: {config.MUTED}; font-size: 0.8rem; margin-top: 3px; }}

.price-card {{
    background: linear-gradient(135deg, {config.NAVY} 0%, {config.TEAL_DARK} 100%);
    border-radius: 20px; padding: 26px 30px; color: #fff; text-align: center;
    box-shadow: 0 10px 28px rgba(11, 37, 69, 0.25);
}}
.price-card .small {{ color: #CFE3EE; font-size: 0.85rem; letter-spacing: .08em; text-transform: uppercase; }}
.price-card .big {{ font-family: 'Poppins', sans-serif; font-size: 2.6rem; font-weight: 700; margin: 6px 0; }}
.price-card .range {{ color: #E3F1F6; font-size: 0.95rem; }}

/* ---------- tabs & sidebar ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #DCE4EC; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; color: {config.MUTED};
}}
.stTabs [aria-selected="true"] {{ color: {config.TEAL_DARK}; background: #E8F6F5; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #E3EAF1; }}
div[data-testid="stForm"] {{ background: {config.CARD}; border: 1px solid #E3EAF1; border-radius: 16px; padding: 18px 22px; }}
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{
    border-radius: 10px; font-weight: 600;
}}
.footer {{ text-align: center; color: {config.MUTED}; font-size: 0.82rem; margin-top: 40px; padding-top: 16px; border-top: 1px solid #DCE4EC; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def usd(value: float) -> str:
    return f"${value:,.0f}"


def hero(title: str, subtitle: str, badges: list) -> None:
    pills = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p>{pills}</div>',
        unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", accent: str = config.TEAL) -> None:
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    css_class = {"info": "callout", "warn": "callout warn", "alert": "callout alert"}[kind]
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def step_card(number: int, title: str, text: str) -> None:
    st.markdown(
        f'<div class="step"><div class="num">{number}</div>'
        f'<div class="title">{html.escape(title)}</div><div class="text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True)


def price_card(model_name: str, price: float, low: float, high: float, error_pct: float) -> None:
    st.markdown(
        f'<div class="price-card"><div class="small">Estimated price - {html.escape(model_name)}</div>'
        f'<div class="big">{usd(price)}</div>'
        f'<div class="range">Typical error band: {usd(low)} to {usd(high)}<br>'
        f'(on the test set, half of the houses were predicted within +/-{error_pct:.1f}%; '
        f'this is not a guaranteed range)</div></div>',
        unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="footer">Punjab College of Technical Engineering | B.Tech 3rd Year AIML | '
        'Made by Manmeet Kaur (Roll No. 2445863) and Avneet Kaur (Roll No. 2445835)<br>'
        'Educational project. Predictions are learned from the dataset in use and are not '
        'professional property valuations.</div>',
        unsafe_allow_html=True)
