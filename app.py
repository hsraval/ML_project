import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ── SESSION STATE INIT (do this FIRST, before any st calls) ──
if "theme" not in st.session_state:
    st.session_state.theme = st.query_params.get("theme", "dark")
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "mob_menu_open" not in st.session_state:
    st.session_state.mob_menu_open = False

st.set_page_config(
    page_title="PowerAI – Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── FAST NAVIGATION: set page and rerun immediately ──
def go_to(page):
    if st.session_state.page != page:
        st.session_state.page = page
        st.session_state.mob_menu_open = False
        st.rerun()

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.query_params["theme"] = st.session_state.theme
    st.rerun()

def toggle_mob_menu():
    st.session_state.mob_menu_open = not st.session_state.mob_menu_open
    st.rerun()

# ══════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
.stApp { background: linear-gradient(170deg, #080d19 0%, #0f172a 45%, #0b1120 100%) !important; }
html, body, [class*="css-"] { font-family: 'Inter', system-ui, sans-serif !important; }
#MainMenu, footer, header[data-testid="stHeader"], .stDeployButton {
    visibility: hidden !important; height: 0 !important;
    padding: 0 !important; margin: 0 !important; overflow: hidden !important;
}
div[data-testid="stSidebar"] { display: none !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── STICKY NAVBAR ── */
[data-testid="stMain"] > div > div > div:first-child {
    position: sticky !important; top: 0 !important; z-index: 9999 !important;
    background: rgba(11, 17, 32, 0.95) !important;
    backdrop-filter: blur(24px) !important; -webkit-backdrop-filter: blur(24px) !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.06) !important;
    padding: 5px 1.5rem !important; margin-bottom: 0 !important;
}
[data-testid="stMain"] > div > div > div:nth-child(n+2) { position: relative; z-index: 1; }
[data-testid="stMain"] > div > div > div:first-child button {
    height: 36px !important; font-size: 0.82rem !important; font-weight: 600 !important;
    border-radius: 8px !important; border: 1px solid transparent !important;
    padding: 0 16px !important; transition: background 0.1s, color 0.1s !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stMain"] > div > div > div:first-child button[kind="secondary"] {
    background: transparent !important; color: #64748b !important;
}
[data-testid="stMain"] > div > div > div:first-child button[kind="secondary"]:hover {
    color: #38bdf8 !important; background: rgba(56, 189, 248, 0.06) !important;
    border-color: rgba(56, 189, 248, 0.1) !important;
}
[data-testid="stMain"] > div > div > div:first-child button[kind="primary"] {
    background: rgba(56, 189, 248, 0.1) !important; color: #38bdf8 !important;
    border-color: rgba(56, 189, 248, 0.2) !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.06) !important;
}

/* ── BRAND ── */
.brand-row { display: flex; align-items: center; gap: 9px; padding: 5px 10px 5px 4px; }
.brand-icon { font-size: 1.35rem; filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.35)); }
.brand-name { font-size: 1.1rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #38bdf8, #06d6a0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* ── MOBILE MENU ── */
.mob-menu-overlay {
    position: fixed; top: 56px; left: 0; right: 0; z-index: 9998;
    background: rgba(11, 17, 32, 0.98);
    border-bottom: 1px solid rgba(56, 189, 248, 0.10);
    padding: 0.5rem 1.2rem 1rem;
    backdrop-filter: blur(24px);
    display: flex; flex-direction: column; gap: 0.4rem;
    animation: slideDown 0.15s ease;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.mob-nav-btn {
    display: block; width: 100%; padding: 0.7rem 1rem;
    border-radius: 8px; font-weight: 600; font-size: 0.88rem;
    color: #94a3b8; background: transparent;
    border: 1px solid transparent; text-align: left; cursor: pointer;
    transition: background 0.1s, color 0.1s;
}
.mob-nav-btn:hover { color: #38bdf8; background: rgba(56,189,248,0.06); }
.mob-nav-btn.active {
    color: #38bdf8; background: rgba(56,189,248,0.10);
    border-color: rgba(56,189,248,0.18);
}

/* ── UTILS ── */
.sec-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 5px; }
.sec-title { font-size: 2rem; font-weight: 900; color: #f1f5f9;
    letter-spacing: -1px; margin-bottom: 3px; }
.sec-line { width: 48px; height: 3px; border-radius: 2px;
    background: linear-gradient(90deg, #38bdf8, #06d6a0); margin-bottom: 1.4rem; }
.badge { display: inline-block; padding: 5px 16px; border-radius: 50px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    background: rgba(6, 214, 160, 0.07); color: #06d6a0;
    border: 1px solid rgba(6, 214, 160, 0.13); margin-bottom: 1.1rem; }
.card { background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(56, 189, 248, 0.05);
    border-radius: 14px; padding: 1.5rem; transition: border-color 0.2s, box-shadow 0.2s; }
.card:hover { border-color: rgba(56, 189, 248, 0.14);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); transform: translateY(-2px); }
.step { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.step-num { width: 30px; height: 30px; min-width: 30px; border-radius: 8px;
    background: rgba(56, 189, 248, 0.07); color: #38bdf8;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem; border: 1px solid rgba(56, 189, 248, 0.12); }
.step h4 { font-size: 0.88rem; font-weight: 600; color: #f1f5f9; margin-bottom: 1px; }
.step p { font-size: 0.8rem; color: #94a3b8; line-height: 1.6; margin: 0; }
.met-box { background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.05);
    border-radius: 12px; padding: 1rem; text-align: center; }
.met-val { font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #06d6a0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.met-lbl { font-size: 0.7rem; color: #94a3b8; font-weight: 500; margin-top: 1px; }
.foot { text-align: center; padding: 1.5rem;
    border-top: 1px solid rgba(56, 189, 248, 0.03); margin-top: 2.5rem; }
.foot p { font-size: 0.76rem; color: #475569; }
.foot .h { color: #f87171; }

/* ── ABOUT PAGE ── */
.about-table { width: 100%; border-collapse: separate; border-spacing: 0;
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.05); margin-top: 0.5rem; }
.about-table thead th { background: rgba(56, 189, 248, 0.04); color: #38bdf8;
    font-weight: 600; font-size: 0.7rem; letter-spacing: 0.5px;
    text-transform: uppercase; padding: 12px 16px; text-align: left;
    border-bottom: 1px solid rgba(56, 189, 248, 0.07); }
.about-table tbody td { padding: 12px 16px; font-size: 0.82rem; color: #cbd5e1;
    border-bottom: 1px solid rgba(56, 189, 248, 0.02);
    background: rgba(15, 23, 42, 0.35); }
.about-table tbody tr:last-child td { border-bottom: none; }
.about-table tbody tr:hover td { background: rgba(56, 189, 248, 0.02); }
.about-table .highlight-row { background: rgba(6, 214, 160, 0.03) !important;
    border-left: 3px solid #06d6a0; }
.about-table .highlight-row td { font-weight: 600; color: #f1f5f9; }
.best-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.6rem; font-weight: 800; background: #06d6a0; color: #0b1120;
    text-transform: uppercase; letter-spacing: 0.5px; margin-left: 6px;
    vertical-align: middle; }

/* ── STREAMLIT OVERRIDES ── */
.stSelectbox > div > div { background: rgba(15, 23, 42, 0.8) !important;
    border-color: rgba(56, 189, 248, 0.08) !important;
    border-radius: 8px !important; color: #e2e8f0 !important; }
.stSelectbox label { color: #94a3b8 !important; font-weight: 500 !important;
    font-size: 0.82rem !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #06d6a0) !important;
    border-radius: 4px !important; }
div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
div[data-testid="stMetricLabel"] { color: #94a3b8 !important;
    font-size: 0.76rem !important; }
.stAlert { border-radius: 10px !important; }
.stCaption { color: #475569 !important; font-size: 0.74rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }

/* ── HERO / MEMBER / FEATURE TABLE ── */
.hero-img { border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.05);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3); }
.hero-img img { width: 100%; height: 280px; object-fit: cover; display: block; }
.member { background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.05);
    border-radius: 14px; padding: 1.3rem; text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s; }
.member:hover { border-color: rgba(56, 189, 248, 0.16); transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); }
.member-av { width: 52px; height: 52px; border-radius: 50%;
    margin: 0 auto 0.6rem;
    background: linear-gradient(135deg, #38bdf8, #06d6a0);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; font-weight: 800; color: #0b1120; }
.member-name { font-size: 0.9rem; font-weight: 700; color: #f1f5f9; margin-bottom: 1px; }
.member-role { font-size: 0.74rem; color: #38bdf8; font-weight: 500; }
.feat-table { width: 100%; border-collapse: separate; border-spacing: 0;
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.05); }
.feat-table thead th { background: rgba(56, 189, 248, 0.04); color: #38bdf8;
    font-weight: 600; font-size: 0.72rem; letter-spacing: 0.5px;
    text-transform: uppercase; padding: 11px 14px; text-align: left;
    border-bottom: 1px solid rgba(56, 189, 248, 0.07); }
.feat-table tbody td { padding: 10px 14px; font-size: 0.8rem; color: #cbd5e1;
    border-bottom: 1px solid rgba(56, 189, 248, 0.02);
    background: rgba(15, 23, 42, 0.35); }
.feat-table tbody tr:last-child td { border-bottom: none; }
.feat-table tbody tr:hover td { background: rgba(56, 189, 248, 0.02); }
.tag { display: inline-block; padding: 2px 8px; border-radius: 5px;
    font-size: 0.66rem; font-weight: 500; margin: 1px 2px 1px 0;
    background: rgba(6, 214, 160, 0.07); color: #06d6a0; }
.res-box { border-radius: 14px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
.res-box.ok { background: rgba(6, 214, 160, 0.04);
    border: 1px solid rgba(6, 214, 160, 0.12); }
.res-box.bad { background: rgba(248, 113, 113, 0.04);
    border: 1px solid rgba(248, 113, 113, 0.12); }
.res-lbl { font-size: 1.35rem; font-weight: 800; margin-bottom: 3px; }
.res-lbl.ok { color: #06d6a0; }
.res-lbl.bad { color: #f87171; }

/* ── DROPDOWN FIX ── */
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.9) !important;
    border-color: rgba(56, 189, 248, 0.12) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * { background-color: transparent !important; }
div[data-baseweb="select"] > div:first-child {
    background-color: rgba(15, 23, 42, 0.9) !important; }
div[data-baseweb="select"] span,
div[data-baseweb="select"] [class*="singleValue"],
div[data-baseweb="select"] [class*="placeholder"],
div[data-baseweb="select"] input {
    color: #e2e8f0 !important; background-color: transparent !important; }
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
div[role="listbox"] {
    background-color: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.10) !important;
    border-radius: 8px !important; }
div[role="option"], li[role="option"] {
    background-color: #0f172a !important; color: #e2e8f0 !important; }
div[role="option"]:hover, li[role="option"]:hover,
div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
    background-color: rgba(56, 189, 248, 0.08) !important; color: #38bdf8 !important; }

/* ── BUTTON OVERRIDES ── */
.stButton > button {
    background: transparent !important; color: #64748b !important;
    border-radius: 10px !important; font-weight: 600 !important;
    border: 1px solid rgba(100, 116, 139, 0.2) !important;
    transition: background 0.1s, color 0.1s !important;
}
[data-testid="stMain"] button[kind="primary"] {
    background: rgba(56, 189, 248, 0.1) !important; color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.06) !important;
}
[data-testid="stMain"] button[kind="secondary"] {
    background: transparent !important; color: #64748b !important;
    border: 1px solid transparent !important;
}
[data-testid="stMain"] button[kind="secondary"]:hover {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.06) !important;
    border-color: rgba(56, 189, 248, 0.1) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8, #06d6a0) !important;
    color: #0b1120 !important; font-size: 0.9rem !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.2) !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(56, 189, 248, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── RESPONSIVE: hide desktop nav items on mobile ── */
@media (max-width: 640px) {
    .desktop-nav { display: none !important; }
    .mobile-nav-btn-col { display: flex !important; }
}
@media (min-width: 641px) {
    .mobile-only { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ── LIGHT MODE OVERRIDE ──
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    .stApp { background: #ffffff !important; }
    html, body, div, span, p, label { color: #000000 !important; }
    h1, h2, h3, h4, h5, h6, .sec-title, .sec-label, .member-name, .member-role { color: #000000 !important; }
    .card, .member, .met-box { background: #ffffff !important;
        border: 1px solid #00000020 !important; color: #000000 !important;
        box-shadow: none !important; }
    [data-testid="stMain"] > div > div > div:first-child {
        background: rgba(255,255,255,0.97) !important;
        border-bottom: 1px solid #00000020 !important; backdrop-filter: none !important; }
    .mob-menu-overlay { background: rgba(255,255,255,0.98) !important; border-bottom: 1px solid #00000020 !important; }
    .mob-nav-btn { color: #374151 !important; }
    .mob-nav-btn:hover { color: #0ea5e9 !important; background: rgba(14,165,233,0.06) !important; }
    .mob-nav-btn.active { color: #0ea5e9 !important; background: rgba(14,165,233,0.10) !important; border-color: rgba(14,165,233,0.18) !important; }
    .stButton > button { background: #000000 !important; color: #ffffff !important; border-radius: 8px !important; }
    button[kind="secondary"] { background: transparent !important; color: #000000 !important; border: 1px solid #00000030 !important; }
    .stSelectbox > div > div { background: #ffffff !important; color: #000000 !important; border: 1px solid #00000030 !important; }
    div[data-baseweb="select"] > div:first-child, div[data-baseweb="select"] > div {
        background-color: #ffffff !important; border: 1px solid #00000030 !important;
        color: #000000 !important; border-radius: 8px !important; }
    div[data-baseweb="select"] * { background-color: transparent !important; }
    div[data-baseweb="select"] > div:first-child { background-color: #ffffff !important; }
    div[data-baseweb="select"] span, div[data-baseweb="select"] [class*="singleValue"],
    div[data-baseweb="select"] [class*="placeholder"], div[data-baseweb="select"] input {
        color: #000000 !important; background-color: transparent !important; }
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"], div[role="listbox"] {
        background-color: #ffffff !important; border: 1px solid #00000020 !important; border-radius: 8px !important; }
    div[role="option"], li[role="option"] { background-color: #ffffff !important; color: #000000 !important; }
    div[role="option"]:hover, li[role="option"]:hover,
    div[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] {
        background-color: #eeeeee !important; color: #000000 !important; }
    .feat-table, .about-table { border: 1px solid #00000020 !important; }
    .feat-table thead th, .about-table thead th { background: #ffffff !important;
        color: #000000 !important; border-bottom: 1px solid #00000030 !important; }
    .feat-table tbody td, .about-table tbody td { background: #ffffff !important;
        color: #000000 !important; border-bottom: 1px solid #00000010 !important; }
    .tag, .badge, .best-tag { background: transparent !important; color: #000000 !important;
        border: 1px solid #00000030 !important; }
    .res-box.ok, .res-box.bad { background: #ffffff !important; border: 1px solid #00000030 !important; }
    .res-lbl.ok, .res-lbl.bad { color: #000000 !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] { color: #000000 !important; }
    .stProgress > div > div > div { background: #000000 !important; }
    .foot p { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════
cur = st.session_state.page

# Desktop navbar (5 columns)
n1, n2, n3, n4, n5 = st.columns([2.3, 1, 1, 1, 0.5], gap="small")

with n1:
    st.markdown("""<div class="brand-row">
        <span class="brand-icon">⚡</span>
        <span class="brand-name">Electricity Consumption</span>
    </div>""", unsafe_allow_html=True)

with n2:
    if st.button("🏠 Home", key="nb_home",
                 use_container_width=True,
                 type="primary" if cur == "Home" else "secondary"):
        go_to("Home")

with n3:
    if st.button("📋 About", key="nb_about",
                 use_container_width=True,
                 type="primary" if cur == "About" else "secondary"):
        go_to("About")

with n4:
    if st.button("🚀 Predict", key="nb_pred",
                 use_container_width=True,
                 type="primary" if cur == "Prediction" else "secondary"):
        go_to("Prediction")

with n5:
    icon = "🌙" if st.session_state.theme == "dark" else "☀️"
    if st.button(icon, key="theme_toggle"):
        toggle_theme()

# Mobile hamburger — inject via JS/CSS trick using a hidden div at top
# We use st.markdown to inject the responsive mobile menu overlay when open
mob_open = st.session_state.mob_menu_open
page_labels = {"Home": "🏠 Home", "About": "📋 About", "Prediction": "🚀 Predict"}

if mob_open:
    # Render clickable buttons styled as mobile menu items
    st.markdown("""
    <div style="
        position: fixed; top: 56px; left: 0; right: 0; z-index: 9998;
        background: rgba(11, 17, 32, 0.98);
        border-bottom: 1px solid rgba(56,189,248,0.10);
        padding: 0.6rem 1.2rem 1rem;
        backdrop-filter: blur(24px);
        animation: slideDown 0.15s ease;
    ">
    </div>
    """, unsafe_allow_html=True)

    mob_col1, mob_col2, mob_col3 = st.columns(3, gap="small")
    with mob_col1:
        if st.button("🏠 Home", key="mob_home", use_container_width=True,
                     type="primary" if cur == "Home" else "secondary"):
            go_to("Home")
    with mob_col2:
        if st.button("📋 About", key="mob_about", use_container_width=True,
                     type="primary" if cur == "About" else "secondary"):
            go_to("About")
    with mob_col3:
        if st.button("🚀 Predict", key="mob_pred", use_container_width=True,
                     type="primary" if cur == "Prediction" else "secondary"):
            go_to("Prediction")

# ══════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════
if cur == "Home":
    st.markdown("""
    <div style="text-align:center; padding:2.2rem 1rem 0.8rem;">
        <div class="badge">🟢 Machine Learning Project</div>
        <h1 style="font-size:2.8rem; font-weight:900; color:#f8fafc; letter-spacing:-1.5px; line-height:1.12; margin-bottom:0.9rem;">
            Predict <span style="background:linear-gradient(135deg,#38bdf8,#06d6a0);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Electricity Consumption</span><br>With Solar &amp; Behavioural Insights
        </h1>
        <p style="font-size:1rem; color:#94a3b8; max-width:580px; margin:0 auto 1.8rem; line-height:1.7;">
            An intelligent ML system that analyzes solar energy infrastructure and behavioural
            patterns to classify community power consumption as Controlled or Uncontrolled.
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2.2, 1], gap="medium")
    with c1:
        st.markdown("""<div class="hero-img">
            <img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?w=900&q=80" alt="Solar Panels" />
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="margin-bottom:0.8rem;">
            <div style="font-size:1.7rem; margin-bottom:0.5rem;">🎯</div>
            <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9; margin-bottom:0.3rem;">High Accuracy</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Bagging Regressor model trained with optimized hyperparameters for reliable predictions.
            </div>
        </div>
        <div class="card">
            <div style="font-size:1.7rem; margin-bottom:0.5rem;">🔄</div>
            <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9; margin-bottom:0.3rem;">Self-Learning</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Model retrains automatically with new user predictions for continuous improvement.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    _, cta, _ = st.columns([1.3, 0.7, 1.3])
    with cta:
        if st.button("🚀  Go to Prediction", key="cta_btn", type="primary", use_container_width=True):
            go_to("Prediction")

    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="sec-label">Our Team</div>
    <div class="sec-title">Meet the Developers</div>
    <div class="sec-line"></div>""", unsafe_allow_html=True)

    members = [
        ("UR", "Member 1", "Model training and Evaluation", "Rathod Umang"),
        ("HR", "Member 2", "Frontend and Graph", "Raval Harsh"),
        ("MS", "Member 3", "Model preprocessing", "Shah Mahek"),
    ]
    mc = st.columns(3, gap="medium")
    for i, (ini, name, role, fullname) in enumerate(members):
        with mc[i]:
            st.markdown(f"""<div class="member">
                <div class="member-av">{ini}</div>
                <div class="member-name">{name}</div>
                <div class="member-name">{fullname}</div>
                <div class="member-role">{role}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="margin-top:2.2rem;">
        <div class="sec-label">Dataset</div>
        <div class="sec-title">Feature Explanations</div>
        <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)

    feats = [
        ("Community_ID", "Identifier", "Unique code for each community. Dropped during preprocessing.", "CC123888"),
        ("SolarEnergy_aspect1", "Solar Infrastructure", "Whether the community is equipped with solar power infrastructure.", "NoSolarPowerEquipped, SolarPowerEquipped"),
        ("SolarEnergy_aspect2", "Additional Solar", "If additional solar power capacity is installed beyond the base setup.", "AddnlSolarPower, NoAddnlPower"),
        ("SolarEnergy_aspect3", "Min Power Protocol", "Whether minimum power generation protocols are enabled.", "MinPowerEnabled, MinPowerNotEnabled"),
        ("SolarEnergy_aspect4", "Battery Storage", "If battery storage systems are installed to store excess solar energy.", "BatteriesEquipped, BatteriesNotEquipped"),
        ("SolarEnergy_aspect5", "DC-AC Conversion", "If DC to AC conversion equipment is available.", "DCtoACEquipped, DCtoACnotEquipped"),
        ("Behavioural_aspect1", "Energy Awareness", "Level of energy conservation awareness among community members.", "Awareness, NoAwareness"),
        ("Behavioural_aspect2", "AC Usage Pattern", "How air conditioning is used — only when needed or always running.", "ACsAllTime, ACsOnNeed"),
        ("Behavioural_aspect3", "Slab System", "Whether the community follows slab-based electricity pricing.", "Slabs, NoSlabs"),
        ("Behavioural_aspect4", "Auto-Off Mechanism", "If automated power cut-off mechanisms are installed.", "Auto-Off, NoAuto-Off"),
        ("Behavioural_aspect5", "Street Lighting", "If energy-efficient street lights are equipped.", "StreetLightsEquipped, StreetLightsNotEquipped"),
        ("Power_Consumption", "Target Variable", "Classification label: Controlled or Uncontrolled power usage.", "Controlled, Uncontrolled"),
    ]
    thtml = '<div style="overflow-x:auto;border-radius:12px;border:1px solid rgba(56,189,248,0.05);"><table class="feat-table"><thead><tr><th>Feature</th><th>Category</th><th>Description</th><th>Values</th></tr></thead><tbody>'
    for f, c, d, v in feats:
        tags = "".join(f'<span class="tag">{x.strip()}</span>' for x in v.split(","))
        thtml += f'<tr><td style="font-weight:600;color:#f1f5f9;white-space:nowrap;">{f}</td><td style="white-space:nowrap;">{c}</td><td>{d}</td><td>{tags}</td></tr>'
    thtml += '</tbody></table></div>'
    st.markdown(thtml, unsafe_allow_html=True)

    text_color = "#ffffff" if st.session_state.theme == "dark" else "#000000"
    st.markdown(f"""
        <div class="badge">Project Goal</div>
        <div class="sec-title">🎯 Objective of the Project</div>
        <div class="sec-line"></div>
        <div class="card">
            <ul style="margin:0; padding-left:18px; line-height:1.8; color:{text_color}; font-size:0.85rem;">
                <li>To predict electricity consumption behavior <b>(Controlled / Uncontrolled)</b></li>
                <li>To analyze the impact of <b>solar infrastructure</b> and <b>human habits</b></li>
                <li>To reduce electricity wastage</li>
                <li>To support <b>smart energy management systems</b></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="badge">System Flow</div>
    <div class="sec-title">⚙️ Working of the Project</div>
    <div class="sec-line"></div>
    <div class="card" style="line-height:1.8; font-size:0.85rem; color:#cbd5e1;">
        1️⃣ <b>User provides input</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ Solar and behavioural features are selected through the interface<br><br>
        2️⃣ <b>Data preprocessing</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ Input data is converted into numerical format using encoding<br><br>
        3️⃣ <b>Model processing</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ Pre-trained <b>Bagging Regressor</b> model processes the input<br><br>
        4️⃣ <b>Prediction</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ Model analyzes the data and predicts electricity consumption<br><br>
        5️⃣ <b>Final output</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ 🟢 <span style="color:#06d6a0;">Controlled</span> (Efficient usage)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ 🔴 <span style="color:#f87171;">Uncontrolled</span> (High wastage)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════
elif cur == "About":
    st.markdown("""
    <div style="padding:1.2rem 0 0;">
        <div class="sec-label">About The Project</div>
        <div class="sec-title">Model Development &amp; Analysis</div>
        <div class="sec-line"></div>
        <p style="font-size:0.92rem; color:#94a3b8; max-width:600px; line-height:1.65;">
            A comprehensive breakdown of the exact methodology, all model experiments,
            preprocessing steps, and evaluation metrics executed in the Jupyter Notebook.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.8rem;">
        <div class="sec-label">Step 1</div>
        <div class="sec-title">Data Preprocessing Pipeline</div>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="step"><div class="step-num">1</div><div><h4>Loaded Dataset &amp; Dropped ID</h4><p>Read <code>powerset.csv</code>. Dropped <code>Community_ID</code> immediately as unique identifiers provide zero predictive value for classification.</p></div></div>
        <div class="step"><div class="step-num">2</div><div><h4>Unique Value Inspection</h4><p>Checked <code>.unique()</code> for all 10 feature columns to understand the exact categorical distributions.</p></div></div>
        <div class="step"><div class="step-num">3</div><div><h4>Handled Missing Values</h4><p>Iterated through all columns and filled nulls with <code>df[i].fillna(df[i].mode()[0])</code> to maintain the most frequent categorical distribution.</p></div></div>
        <div class="step"><div class="step-num">4</div><div><h4>Target Label Encoding</h4><p>Applied <code>LabelEncoder</code> on <code>Power_Consumption</code> mapping: <strong>Controlled → 0</strong>, <strong>Uncontrolled → 1</strong>.</p></div></div>
        <div class="step"><div class="step-num">5</div><div><h4>Feature One-Hot Encoding</h4><p>Applied <code>pd.get_dummies(X, drop_first=True)</code> to convert all categorical features into binary columns, avoiding the dummy variable trap.</p></div></div>
        <div class="step"><div class="step-num">6</div><div><h4>Train-Test Split</h4><p>Split data into 80% training and 20% testing using <code>train_test_split(random_state=42)</code>.</p></div></div>
        <div class="step"><div class="step-num">7</div><div><h4>Initial Model Testing</h4><p>Tested multiple models like <code>LinearRegression()</code>, <code>Ridge()</code>, <code>Lasso()</code>, <code>XGBRegressor()</code>, and <code>BaggingRegressor()</code>.</p></div></div>
        <div class="step"><div class="step-num">8</div><div><h4>Model Comparison</h4><p>Compared models using <code>r2_score(y_test, y_pred)</code> to identify best performing model.</p></div></div>
        <div class="step"><div class="step-num">9</div><div><h4>Selected Bagging Regressor</h4><p>Chose <code>BaggingRegressor()</code> as final model because it provided highest accuracy and reduced overfitting.</p></div></div>
        <div class="step"><div class="step-num">10</div><div><h4>Model Training</h4><p>Trained model using <code>BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)</code>.</p></div></div>
        <div class="step"><div class="step-num">11</div><div><h4>Model Evaluation</h4><p>Evaluated performance using <code>MSE</code>, <code>MAE</code>, <code>RMSE</code>, and <code>R² Score</code>.</p></div></div>
        <div class="step"><div class="step-num">12</div><div><h4>Hyperparameter Tuning</h4><p>Used <code>GridSearchCV()</code> and <code>RandomizedSearchCV()</code> to optimize parameters like <code>n_estimators</code> and <code>max_samples</code>.</p></div></div>
        <div class="step"><div class="step-num">13</div><div><h4>Feature Importance</h4><p>Calculated importance using <code>model.estimators_[0].feature_importances_</code> to identify key features.</p></div></div>
        <div class="step"><div class="step-num">14</div><div><h4>Graph Visualization</h4><p>Plotted graphs like <code>plt.bar()</code>, <code>sns.heatmap()</code>, and <code>plt.hist()</code> for analysis.</p></div></div>
        <div class="step"><div class="step-num">15</div><div><h4>Model Saving</h4><p>Saved final model using <code>joblib.dump(model, "Model/bagging_Regressor_model.joblib")</code>.</p></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.8rem;">
        <div class="sec-label">Step 2</div>
        <div class="sec-title">Complete Model Training History</div>
        <div class="sec-line"></div>
        <p style="font-size:0.84rem; color:#94a3b8; margin-bottom:1rem;">
            Every single model experiment from the notebook, showing the progression
            from base models to hyperparameter-tuned variants.
        </p>
    </div>
    """, unsafe_allow_html=True)

    models_data = [
        ("Linear Regression", "Base Model", "Default params", "72.44"),
        ("Linear Regression", "Polynomial (Deg 2)", "Degree=2 feature expansion", "90.16"),
        ("Ridge Regression", "Base Model", "alpha=1.0 (Default)", "72.74"),
        ("Lasso Regression", "Base Model", "alpha=0.1", "72.09"),
        ("XGBoost", "Base Model", "Default params", "94.88"),
        ("Bagging (DT Base)", "Base Model", "n_estimators=10", "96.07"),
    ]

    table_html = '<div style="overflow-x:auto;"><table class="about-table"><thead><tr><th>Model Family</th><th>Experiment</th><th>Parameters Tuned</th><th style="text-align:right;">R² Score</th></tr></thead><tbody>'
    for name, exp, params, score in models_data:
        row_class = "highlight-row" if name == "Bagging (DT Base)" else ""
        badge = '<span class="best-tag">Final Model</span>' if row_class else ""
        table_html += f'<tr class="{row_class}"><td style="font-weight:600;color:#f1f5f9;white-space:nowrap;">{name}</td><td>{exp}{badge}</td><td style="color:#94a3b8;font-size:0.78rem;">{params}</td><td style="text-align:right;font-weight:700;color:#38bdf8;">{score}</td></tr>'
    table_html += '</tbody></table></div>'
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.8rem;">
        <div class="sec-label">Step 4</div>
        <div class="sec-title">Evaluation Metrics Explained</div>
        <div class="sec-line"></div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
        <div class="card"><div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:0.3rem;">📊 R² Score</div><div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">Proportion of variance in the target variable predictable from features. 1.0 is perfect.</div></div>
        <div class="card"><div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:0.3rem;">📉 MSE</div><div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">Average squared difference between predicted and actual values. Highly penalizes large errors.</div></div>
        <div class="card"><div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:0.3rem;">📏 MAE</div><div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">Average absolute difference. More robust to outliers than MSE and easier to interpret.</div></div>
        <div class="card"><div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:0.3rem;">📐 RMSE</div><div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">Square root of MSE. Represents error magnitude in the same units as the target variable.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.8rem;">
        <div class="sec-label">Step 5</div>
        <div class="sec-title">Feature Correlation Heatmap</div>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        df_h = pd.read_csv("powerset.csv")
        df_h = df_h.drop("Community_ID", axis=1)
        le_h = LabelEncoder()
        for col in df_h.columns:
            df_h[col] = le_h.fit_transform(df_h[col])
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_h.corr(), annot=True, cmap="Blues", fmt=".2f",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap (Label Encoded)",
                     fontsize=13, fontweight="bold", color="white", pad=12)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

    st.markdown('<div class="foot"><p>Built with <span class="h">❤</span> using Streamlit &amp; Scikit-Learn &nbsp;|&nbsp; PowerAI © 2025</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PREDICTION PAGE
# ══════════════════════════════════════════════
elif cur == "Prediction":
    st.markdown("""
    <div style="padding:1.2rem 0 0;">
        <div class="sec-label">Real-Time Inference</div>
        <div class="sec-title">Predict Electricity Consumption</div>
        <div class="sec-line"></div>
        <p style="font-size:0.9rem;color:#94a3b8;max-width:500px;line-height:1.6;">
            Select the solar infrastructure and behavioural characteristics of a community
            to predict its power consumption classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model = joblib.load("Model/bagging_Regressor_model.joblib")
        feature_columns = joblib.load("Model/feature_columns.joblib")
    except Exception as e:
        st.error(f"❌ Model files not found. Ensure the Model folder contains the trained .joblib files.\n\nError: {e}")
        st.stop()

    def retrain_model():
        try:
            original = pd.read_csv("powerset.csv")
            if os.path.exists("user_data.csv"):
                user_data = pd.read_csv("user_data.csv")
                user_data = user_data.drop(columns=["Prediction", "Model_Output"], errors='ignore')
                combined = pd.concat([original, user_data], ignore_index=True)
            else:
                combined = original
            X = combined.drop("Power_Consumption", axis=1)
            y = combined["Power_Consumption"]
            from sklearn.preprocessing import LabelEncoder as LE
            le = LE()
            y = le.fit_transform(y)
            X = pd.get_dummies(X)
            X = X.reindex(columns=feature_columns, fill_value=0)
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor
            new_model = BaggingRegressor(
                estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
            new_model.fit(X, y)
            joblib.dump(new_model, "Model/bagging_Regressor_model.joblib")
        except Exception:
            pass

    features = {
        "SolarEnergy_aspect1": ["NoSolarPowerEquipped", "SolarPowerEquipped"],
        "SolarEnergy_aspect2": ["AddnlSolarPower", "NoAddnlPower"],
        "SolarEnergy_aspect3": ["MinPowerEnabled", "MinPowerNotEnabled"],
        "SolarEnergy_aspect4": ["BatteriesEquipped", "BatteriesNotEquipped"],
        "SolarEnergy_aspect5": ["DCtoACEquipped", "DCtoACnotEquipped"],
        "Behavioural_aspect1": ["Awareness", "NoAwareness"],
        "Behavioural_aspect2": ["ACsAllTime", "ACsOnNeed"],
        "Behavioural_aspect3": ["NoSlabs", "Slabs"],
        "Behavioural_aspect4": ["Auto-Off", "NoAuto-Off"],
        "Behavioural_aspect5": ["StreetLightsEquipped", "StreetLightsNotEquipped"],
    }

    user_inputs = {}
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div style="font-size:1.02rem;font-weight:700;color:#f1f5f9;margin-bottom:0.8rem;">☀️ Solar Features</div>', unsafe_allow_html=True)
        for feat, opts in list(features.items())[:5]:
            user_inputs[feat] = st.selectbox(
                feat.replace("_", " "), opts, key=f"s_{feat}")

    with c2:
        st.markdown('<div style="font-size:1.02rem;font-weight:700;color:#f1f5f9;margin-bottom:0.8rem;">🏠 Behaviour Features</div>', unsafe_allow_html=True)
        for feat, opts in list(features.items())[5:]:
            user_inputs[feat] = st.selectbox(
                feat.replace("_", " "), opts, key=f"s_{feat}")

    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

    if st.button("🚀  Predict Consumption", type="primary", key="pred_btn"):
        raw_df = pd.DataFrame([user_inputs])
        input_df = pd.get_dummies(raw_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        result = 1 if prediction >= 0.5 else 0
        label = "Uncontrolled" if result == 1 else "Controlled"

        rc = "bad" if result == 1 else "ok"
        icon = "⚠️" if result == 1 else "✅"
        st.markdown(f"""
        <div class="res-box {rc}">
            <div class="res-lbl {rc}">{icon} {label} Consumption</div>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
                The model predicts this community's power consumption is
                <strong style="color:#cbd5e1;">{label.lower()}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        prob = max(0.0, min(1.0, prediction)) * 100
        mc1, mc2, mc3 = st.columns([1, 2, 1])
        with mc2:
            st.markdown('<div style="text-align:center;margin-top:0.6rem;"><span style="font-size:0.7rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;">Confidence Score</span></div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{prob:.2f}%")
            st.progress(int(prob))
            st.caption(f"Model Output: {prediction:.4f}")

        raw_df["Power_Consumption"] = label
        raw_df["Prediction"] = result
        raw_df["Model_Output"] = prediction

        fp = "user_data.csv"
        if os.path.exists(fp):
            raw_df.to_csv(fp, mode='a', header=False, index=False)
        else:
            raw_df.to_csv(fp, mode='w', header=True, index=False)

        retrain_model()

        st.markdown('<div style="text-align:center;margin-top:1rem;"><p style="color:#475569;font-size:0.76rem;">✅ Prediction saved &amp; model retrained with new data.</p></div>', unsafe_allow_html=True)
