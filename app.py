import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# =====================================================
# 1. PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "supply_chain_transformer.pth"
PREPROCESS_PATH = BASE_DIR / "supply_chain_preprocessing.pkl"

# Banner photo: keep hero.jpg in the same folder as this file.
# (hero.png / hero.webp also work.) Without it, a navy gradient is used.
HERO_IMAGE_CANDIDATES = ["hero.jpg", "hero.jpeg", "hero.png", "hero.webp"]


# =====================================================
# 2. TRANSFORMER MODEL  (unchanged)
# =====================================================

class SupplyChainTransformer(nn.Module):

    def __init__(
        self,
        category_counts,
        num_numerical,
        d_model=32,
        nhead=4
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(count, d_model)
            for count in category_counts
        ])

        self.num_projection = nn.Linear(
            num_numerical, d_model
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2)
        )

    def forward(self, categorical, numerical):

        embedded = torch.stack([
            embedding(categorical[:, i])
            for i, embedding in enumerate(self.embeddings)
        ], dim=1)

        numerical_token = self.num_projection(
            numerical
        ).unsqueeze(1)

        cls = self.cls_token.expand(
            categorical.size(0), -1, -1
        )

        tokens = torch.cat(
            [cls, embedded, numerical_token],
            dim=1
        )

        encoded = self.transformer(tokens)

        return self.classifier(encoded[:, 0, :])


# =====================================================
# 3. THEME  (NAVY BLUE & LUXURY GOLD)
# =====================================================



def hero_background():
    """Return a CSS background-image value for the hero banner."""

    # Use hero.jpg / hero.png / hero.webp from the app folder
    for name in HERO_IMAGE_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            ext = path.suffix.lower().strip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            data = base64.b64encode(path.read_bytes()).decode()
            return f"url('data:{mime};base64,{data}')"

    # No hero image found: fall back to a plain navy gradient
    return "linear-gradient(135deg, #0c1b36, #12274a)"


def inject_css():
    hero_bg = hero_background()

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --navy-950: #050c1c;
    --navy-900: #081328;
    --navy-800: #0c1b36;
    --navy-700: #12274a;
    --navy-600: #1b3560;
    --line: #23406b;
    --gold: #e8b85b;
    --gold-soft: #f5d58b;
    --gold-deep: #c99a3c;
    --text: #f1f5f9;
    --muted: #9fb0c8;
    --danger: #ff6b6b;
    --ok: #4ade80;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif;
}}

/* ---------- App background ---------- */
.stApp {{
    background:
        radial-gradient(1200px 500px at 70% -10%, rgba(232,184,91,0.10), transparent 60%),
        linear-gradient(160deg, var(--navy-950) 0%, var(--navy-900) 45%, #0a1a38 100%);
    color: var(--text);
}}

header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    max-width: 1280px;
    padding-top: 1.4rem !important;
    padding-bottom: 2rem;
}}

h1, h2, h3, h4 {{ color: var(--gold-soft) !important; letter-spacing: 0.2px; }}
p, label, .stMarkdown, li {{ color: var(--text); }}
hr {{ border-color: var(--line) !important; opacity: 0.6; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--navy-900), var(--navy-950));
    border-right: 1px solid var(--line);
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0.6rem; }}

.brand {{
    display: flex; align-items: center; gap: 14px;
    padding: 6px 4px 18px 4px;
}}
.brand-icon {{
    font-size: 38px; line-height: 1;
    filter: drop-shadow(0 4px 10px rgba(232,184,91,0.35));
}}
.brand-title {{
    color: var(--gold); font-size: 25px; font-weight: 700; line-height: 1.1;
}}
.brand-sub {{
    color: var(--gold-soft); font-size: 10.5px; letter-spacing: 3px; opacity: 0.85;
}}

/* Sidebar navigation buttons */
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: transparent;
    border: 1px solid transparent;
    border-left: 4px solid transparent;
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--text);
    font-size: 16px;
    font-weight: 600;
    justify-content: flex-start;
    text-align: left;
    transition: all .2s ease;
}}
section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button p {{
    justify-content: flex-start;
    text-align: left;
    color: var(--text);
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(232,184,91,0.08);
    border-left-color: rgba(232,184,91,0.6);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, rgba(232,184,91,0.22), rgba(232,184,91,0.04));
    border-left-color: var(--gold);
    color: var(--gold-soft);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p {{
    color: var(--gold-soft);
}}

.side-card {{
    border-top: 1px solid var(--line);
    padding: 16px 6px 4px 6px;
    margin-top: 14px;
}}
.side-card .t {{ color: var(--text); font-weight: 600; font-size: 14px; }}
.side-card .s {{ color: var(--muted); font-size: 12.5px; margin-top: 3px; line-height: 1.5; }}

/* ---------- Hero ---------- */
.hero {{
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid var(--line);
    padding: 40px 44px 36px 44px;
    margin-bottom: 26px;
    min-height: 280px;
    background-color: #081328;
    background-image:
        linear-gradient(90deg, #060e20 0%, #081328 42%, rgba(8,19,40,0.55) 58%, rgba(8,19,40,0) 75%),
        {hero_bg};
    background-size: 100% 100%, auto 100%;
    background-position: center, right center;
    background-repeat: no-repeat, no-repeat;
    box-shadow: 0 20px 50px rgba(0,0,0,0.45);
}}
.hero.compact {{ min-height: 150px; padding: 28px 44px; }}
.hero .eyebrow {{
    color: var(--gold); font-size: 12px; letter-spacing: 3px; font-weight: 600;
    text-transform: uppercase; margin: 0 0 10px 0;
}}
.hero h1 {{
    font-size: 44px; font-weight: 800; margin: 0 0 6px 0; line-height: 1.1;
    color: #fff !important;
}}
.hero h1 span {{ color: var(--gold-soft); }}
.hero h3 {{
    color: var(--gold) !important; font-size: 22px; font-weight: 600; margin: 0 0 14px 0;
}}
.hero p.desc {{ color: #d7e0ee; max-width: 520px; font-size: 15.5px; line-height: 1.6; margin: 0; }}

/* ---------- Section headers ---------- */
.section-head {{ display: flex; align-items: center; gap: 14px; margin: 8px 0 4px 0; }}
.section-head .ico {{
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 24px;
    background: rgba(232,184,91,0.10); border: 1px solid rgba(232,184,91,0.5);
}}
.section-head .ttl {{ color: var(--gold-soft); font-size: 27px; font-weight: 700; }}
.section-sub {{ color: var(--muted); font-size: 14.5px; margin: 0 0 16px 60px; max-width: 760px; }}

/* ---------- Stat cards ---------- */
.stat-card {{
    background: linear-gradient(160deg, var(--navy-800), var(--navy-900));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 130px;
    transition: transform .2s ease, border-color .2s ease;
}}
.stat-card:hover {{ transform: translateY(-3px); border-color: var(--gold); }}
.stat-card .k {{ color: var(--muted); font-size: 11.5px; letter-spacing: 2px; font-weight: 600; }}
.stat-card .v {{ color: var(--gold-soft); font-size: 24px; font-weight: 700; margin: 6px 0 2px 0; }}
.stat-card .d {{ color: #cbd5e1; font-size: 13.5px; }}

/* ---------- Form ---------- */
div[data-testid="stForm"] {{
    background: linear-gradient(160deg, rgba(18,39,74,0.55), rgba(8,19,40,0.85));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}}
div[data-testid="stForm"] h3 {{ font-size: 18px; margin-top: 4px; }}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] input {{
    background-color: var(--navy-800) !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}}
div[data-baseweb="select"] > div:hover,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
    border-color: var(--gold) !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: var(--navy-700) !important; color: var(--gold-soft) !important;
}}
input, textarea {{ color: var(--text) !important; }}

label[data-testid="stWidgetLabel"] p {{
    color: #e2e8f0 !important; font-weight: 600; font-size: 14px;
}}

/* ---------- Buttons ---------- */
div[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(90deg, #d9a844, #f5d58b, #d9a844);
    background-size: 200% 100%;
    color: #111827 !important;
    font-weight: 700;
    font-size: 16px;
    border: none;
    border-radius: 10px;
    min-height: 50px;
    box-shadow: 0 8px 22px rgba(232,184,91,0.28);
    transition: all .3s ease;
}}
div[data-testid="stFormSubmitButton"] button:hover {{
    background-position: 100% 0;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(232,184,91,0.4);
    color: #111827 !important;
}}

/* ---------- Alerts / metrics ---------- */
div[data-testid="stAlert"] {{ border-radius: 12px; }}
div[data-testid="stMetric"] {{
    background: var(--navy-800); padding: 18px; border: 1px solid var(--line); border-radius: 14px;
}}

/* ---------- Result card ---------- */
.result-card {{
    border-radius: 18px; padding: 26px 28px; border: 1px solid var(--line);
    background: linear-gradient(160deg, var(--navy-700), var(--navy-900));
    box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}}
.result-card.high {{ border-color: rgba(255,107,107,0.65); box-shadow: 0 0 0 1px rgba(255,107,107,0.15), 0 16px 40px rgba(0,0,0,0.35); }}
.result-card.low  {{ border-color: rgba(74,222,128,0.6);  box-shadow: 0 0 0 1px rgba(74,222,128,0.15), 0 16px 40px rgba(0,0,0,0.35); }}
.result-card .lbl {{ color: var(--muted); font-size: 12px; letter-spacing: 2px; font-weight: 600; }}
.result-card .verdict {{ font-size: 28px; font-weight: 800; margin: 6px 0 4px 0; }}
.result-card.high .verdict {{ color: var(--danger); }}
.result-card.low .verdict  {{ color: var(--ok); }}
.result-card .hint {{ color: #cbd5e1; font-size: 14px; }}

.prob-big {{ font-size: 46px; font-weight: 800; color: var(--gold-soft); line-height: 1; }}
.prob-lbl {{ color: var(--muted); font-size: 12px; letter-spacing: 2px; font-weight: 600; margin-bottom: 6px; }}
.gauge {{
    position: relative; height: 14px; border-radius: 999px; margin-top: 16px;
    background: linear-gradient(90deg, #4ade80 0%, #e8b85b 50%, #ff6b6b 100%);
}}
.gauge .knob {{
    position: absolute; top: -5px; width: 8px; height: 24px; border-radius: 4px;
    background: #fff; box-shadow: 0 0 10px rgba(255,255,255,0.8); transform: translateX(-50%);
}}
.gauge-scale {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 11.5px; margin-top: 6px; }}

/* ---------- Sensitivity bars ---------- */
.sens-row {{ display: flex; align-items: center; gap: 14px; margin: 10px 0; }}
.sens-name {{ width: 230px; color: #e2e8f0; font-size: 14px; font-weight: 500; }}
.sens-track {{ flex: 1; height: 12px; background: var(--navy-800); border-radius: 999px; border: 1px solid var(--line); overflow: hidden; }}
.sens-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--gold-deep), var(--gold-soft)); }}
.sens-val {{ width: 70px; text-align: right; color: var(--gold-soft); font-weight: 700; font-size: 14px; }}

/* ---------- How it works ---------- */
.how-title {{ display:flex; align-items:center; gap:10px; color: var(--gold-soft); font-size: 21px; font-weight: 700; margin: 28px 0 14px 0; }}
.how-card {{
    position: relative;
    border: 1px solid var(--gold-deep);
    border-radius: 14px;
    padding: 20px 20px 20px 78px;
    min-height: 118px;
    background: linear-gradient(160deg, rgba(18,39,74,0.55), rgba(8,19,40,0.9));
    transition: all .25s ease;
}}
.how-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 28px rgba(232,184,91,0.15); border-color: var(--gold-soft); }}
.how-card .ic {{ position: absolute; left: 20px; top: 20px; font-size: 32px; }}
.how-card .num {{
    display: inline-block; font-size: 12px; font-weight: 700; color: var(--gold-soft);
    background: rgba(232,184,91,0.18); border-radius: 999px; padding: 3px 9px; margin-bottom: 6px;
}}
.how-card .h {{ color: var(--gold-soft); font-weight: 700; font-size: 17px; }}
.how-card .p {{ color: #b6c4d9; font-size: 13.5px; margin-top: 4px; line-height: 1.45; }}

/* ---------- About ---------- */
.info-card {{
    background: linear-gradient(160deg, var(--navy-800), var(--navy-900));
    border: 1px solid var(--line); border-radius: 16px; padding: 22px 24px; height: 100%;
}}
.info-card h4 {{ margin: 0 0 8px 0; font-size: 18px; }}
.info-card p, .info-card li {{ color: #cbd5e1; font-size: 14.5px; line-height: 1.65; }}
.chip {{
    display: inline-block; margin: 4px 6px 4px 0; padding: 5px 12px; border-radius: 999px;
    border: 1px solid rgba(232,184,91,0.5); color: var(--gold-soft); font-size: 12.5px;
    background: rgba(232,184,91,0.08);
}}

/* ---------- Footer ---------- */
.footer {{
    text-align: center; color: var(--muted); font-size: 12.5px;
    padding: 22px 0 6px 0; margin-top: 30px; border-top: 1px solid var(--line);
}}
.footer b {{ color: var(--gold); letter-spacing: 2px; font-weight: 600; }}

@media (max-width: 900px) {{
    .hero {{ padding: 28px 22px; }}
    .hero h1 {{ font-size: 32px; }}
    .sens-name {{ width: 130px; font-size: 12.5px; }}
}}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD MODEL AND PREPROCESSING  (unchanged)
# =====================================================

@st.cache_resource
def load_model():

    if not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: {PREPROCESS_PATH.name}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    with open(PREPROCESS_PATH, "rb") as file:
        prep = pickle.load(file)

    cat_cols = prep["categorical_cols"]
    num_cols = prep["numerical_cols"]
    encoders = prep["label_encoders"]
    scaler = prep["scaler"]
    category_counts = prep["category_counts"]

    model = SupplyChainTransformer(
        category_counts=category_counts,
        num_numerical=len(num_cols)
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model, cat_cols, num_cols, encoders, scaler


# =====================================================
# 5. UI HELPERS
# =====================================================

NAV_ITEMS = ["  Home", " Prediction", "  About"]


def render_sidebar():

    with st.sidebar:

        st.markdown("""
        <div class="brand">
            <div class="brand-icon">📦</div>
            <div>
                <div class="brand-title">Supply Chain</div>
                <div class="brand-sub">INTELLIGENT LOGISTICS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state["page"] = NAV_ITEMS[0]

        for item in NAV_ITEMS:
            active = st.session_state["page"] == item
            if st.button(
                item,
                key=f"nav_{item}",
                use_container_width=True,
                type="primary" if active else "secondary"
            ):
                st.session_state["page"] = item
                st.rerun()

        page = st.session_state["page"]

        st.markdown("""
        <div class="side-card">
            <div class="t">Model</div>
            <div class="s">Transformer Neural Network</div>
        </div>
        <div class="side-card">
            <div class="t">Input Features</div>
            <div class="s">3 Categorical &nbsp;•&nbsp; 4 Numerical</div>
        </div>
        <div class="side-card">
            <div class="t">Design</div>
            <div class="s">Navy Blue &amp; Luxury Gold</div>
        </div>
        <div class="side-card">
            <div class="t">Supply Chain Risk Analytics</div>
            <div class="s">DataCo Supply Chain Project</div>
        </div>
        """, unsafe_allow_html=True)

    return page


def render_hero(compact=False):

    if compact:
        st.markdown("""
        <div class="hero compact">
            <p class="eyebrow">Smarter Decisions · Stronger Supply Chains</p>
            <h1>Late Delivery <span>Risk Prediction</span></h1>
            <p class="desc">Enter order details and let the Transformer model estimate delivery risk.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="hero">
        <p class="eyebrow">Smarter Decisions · Stronger Supply Chains</p>
        <h1>Supply Chain <span>Intelligence</span></h1>
        <h3>AI-Powered Late Delivery Risk Prediction</h3>
        <p class="desc">
            Analyze order details, estimate delivery risk, and support smarter
            logistics decisions with a trained Transformer model.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_overview():

    c1, c2, c3 = st.columns(3)

    cards = [
        (c1, "MODEL", "Transformer", "Deep learning prediction"),
        (c2, "INPUT FEATURES", "7 Features", "Order and shipment details"),
        (c3, "OBJECTIVE", "Risk Analysis", "Estimate late-delivery risk"),
    ]

    for col, k, v, d in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="k">{k}</div>
                <div class="v">{v}</div>
                <div class="d">{d}</div>
            </div>
            """, unsafe_allow_html=True)


def render_how_it_works():

    st.markdown('<div class="how-title"> How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("📝", "01", "Enter Details", "Provide order features."),
        ("🧠", "02", "Model Analysis", "The Transformer processes the input."),
        ("📈", "03", "Get Prediction", "Review the estimated risk."),
        ("🎯", "04", "Plan Ahead", "Use the estimate to support decisions."),
    ]

    cols = st.columns(4)

    for col, (ic, num, h, p) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="ic">{ic}</div>
                <div class="num">{num}</div>
                <div class="h">{h}</div>
                <div class="p">{p}</div>
            </div>
            """, unsafe_allow_html=True)


def render_footer():

    st.markdown("""
    <div class="footer">
        <b>SUPPLY CHAIN RISK ANALYTICS</b><br>
        Transformer-Based Late Delivery Risk Prediction &nbsp;|&nbsp; DataCo Supply Chain Project
    </div>
    """, unsafe_allow_html=True)


def section_header(icon, title, subtitle=""):

    st.markdown(f"""
    <div class="section-head">
        <div class="ico">{icon}</div>
        <div class="ttl">{title}</div>
    </div>
    """, unsafe_allow_html=True)

    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_result(predicted_class, late_probability):

    pct = late_probability * 100
    high = predicted_class == 1

    left, right = st.columns([1, 1.3])

    with left:
        if high:
            st.markdown("""
            <div class="result-card high">
                <div class="lbl">PREDICTION RESULT</div>
                <div class="verdict">⚠️ High Late-Delivery Risk</div>
                <div class="hint">This order is likely to be delivered late. Consider
                faster shipping or earlier dispatch.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card low">
                <div class="lbl">PREDICTION RESULT</div>
                <div class="verdict">✅ Low Late-Delivery Risk</div>
                <div class="hint">This order is likely to be delivered on schedule.</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="result-card">
            <div class="prob-lbl">LATE-DELIVERY PROBABILITY</div>
            <div class="prob-big">{pct:.2f}%</div>
            <div class="gauge"><div class="knob" style="left:{min(max(pct, 1), 99):.2f}%"></div></div>
            <div class="gauge-scale"><span>Low risk</span><span>Medium</span><span>High risk</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "This result is a model estimate, not a guarantee. "
        "Confirm that class 1 represents late delivery "
        "in your original training labels."
    )


def render_sensitivity(effects_df):

    section_header(
        " ",
        "Model Response to Input Changes",
        "Shows how the predicted probability changes when individual visible "
        "inputs are varied. This does not establish causation.""
    )

    max_val = max(float(effects_df["Probability Change (%)"].max()), 1e-9)

    rows = ""
    for _, r in effects_df.iterrows():
        v = float(r["Probability Change (%)"])
        w = max(v / max_val * 100, 1.5)
        rows += (
            '<div class="sens-row">'
            f'<div class="sens-name">{r["Feature"]}</div>'
            f'<div class="sens-track"><div class="sens-fill" style="width:{w:.1f}%"></div></div>'
            f'<div class="sens-val">{v:.2f}%</div>'
            '</div>'
        )

    st.markdown(f'<div class="info-card">{rows}</div>', unsafe_allow_html=True)

    with st.expander("View data table"):
        st.dataframe(
            effects_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Unshown model inputs remain at default values. "
        "Results may differ from predictions using all features."
    )


# =====================================================
# 6. PREDICTION APPLICATION  (logic unchanged)
# =====================================================

wanted_cat = [
    "Shipping Mode",
    "Market",
    "Customer Segment"
]

wanted_num = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price",
    "Order Item Discount"
]


def render_prediction():

    section_header(
        " ",
        "Late Delivery Risk Analysis",
        "Choose the order categories and enter numerical details. The default "
        "numerical values are only starting points; replace them with realistic "
        "order information when available."
    )

    try:

        model, cat_cols, num_cols, encoders, scaler = load_model()

        missing = [
            col for col in wanted_cat
            if col not in cat_cols or col not in encoders
        ] + [
            col for col in wanted_num
            if col not in num_cols
        ]

        if missing:
            st.error(
                "Input features missing from preprocessing: "
                + ", ".join(missing)
            )
            st.stop()

        if len(category_counts := [
            embedding.num_embeddings
            for embedding in model.embeddings
        ]) != len(cat_cols):
            st.error(
                "The model and preprocessing categorical features do not match."
            )
            st.stop()

        st.success("Transformer model loaded successfully.")

        with st.form("prediction_form"):

            st.markdown("### Customer and Shipping Details")

            cat_values = {}
            cat_columns = st.columns(3)

            for i, col in enumerate(wanted_cat):

                options = list(encoders[col].classes_)

                with cat_columns[i]:
                    cat_values[col] = st.selectbox(
                        col,
                        options,
                        key=f"cat_{col}"
                    )

            st.markdown("### Order and Product Details")

            num_values = {}
            num_columns = st.columns(2)

            integer_fields = {
                "Days for shipment (scheduled)",
                "Order Item Quantity"
            }

            for i, col in enumerate(wanted_num):

                idx = num_cols.index(col)
                default = float(scaler.mean_[idx])

                with num_columns[i % 2]:

                    if col in integer_fields:

                        num_values[col] = st.number_input(
                            col,
                            min_value=0,
                            value=max(0, int(round(default))),
                            step=1
                        )

                    else:

                        num_values[col] = st.number_input(
                            col,
                            min_value=0.0,
                            value=max(0.0, default),
                            format="%.4f"
                        )

            submitted = st.form_submit_button(
                "📈  Predict Delivery Risk",
                use_container_width=True
            )

        # -----------------------------------------
        # PREDICTION RESULT
        # -----------------------------------------

        if submitted:

            encoded = []

            # Default values for any hidden categorical columns
            for col in cat_cols:

                first_option = encoders[col].classes_[0]

                value = encoders[col].transform(
                    [first_option]
                )[0]

                encoded.append(int(value))

            # Replace defaults with selected values
            for col, value in cat_values.items():

                idx = cat_cols.index(col)

                encoded[idx] = int(
                    encoders[col].transform([value])[0]
                )

            # Default numerical values
            numerical_row = np.asarray(
                scaler.mean_,
                dtype=float
            ).copy()

            # Replace selected numerical features
            for col, value in num_values.items():

                numerical_row[num_cols.index(col)] = float(value)

            numerical_df = pd.DataFrame(
                [numerical_row],
                columns=num_cols
            )

            scaled = scaler.transform(numerical_df)

            cat_tensor = torch.tensor(
                [encoded],
                dtype=torch.long
            )

            num_tensor = torch.tensor(
                scaled,
                dtype=torch.float32
            )

            with torch.no_grad():

                output = model(
                    cat_tensor,
                    num_tensor
                )

                probs = torch.softmax(output, dim=1)[0]

                predicted_class = int(
                    torch.argmax(probs).item()
                )

                late_probability = float(probs[1].item())

            st.markdown("<br>", unsafe_allow_html=True)

            section_header("🎯", "Prediction Result")

            render_result(predicted_class, late_probability)

            # -------------------------------------
            # INPUT SENSITIVITY
            # -------------------------------------

            effects = []

            # Categorical feature sensitivity
            for col in wanted_cat:

                idx = cat_cols.index(col)
                values = []

                for option in encoders[col].classes_:

                    changed = encoded.copy()

                    changed[idx] = int(
                        encoders[col].transform([option])[0]
                    )

                    changed_tensor = torch.tensor(
                        [changed],
                        dtype=torch.long
                    )

                    with torch.no_grad():

                        out = model(
                            changed_tensor,
                            num_tensor
                        )

                        prob = torch.softmax(
                            out, dim=1
                        )[0, 1].item()

                    values.append(float(prob))

                effects.append({
                    "Feature": col,
                    "Probability Change (%)": (
                        max(values) - min(values)
                    ) * 100
                })

            # Numerical feature sensitivity
            for col in wanted_num:

                idx = num_cols.index(col)

                changed = scaled.copy()

                # Change the scaled feature by 0.5
                changed[0, idx] += 0.5

                changed_tensor = torch.tensor(
                    changed,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    out = model(
                        cat_tensor,
                        changed_tensor
                    )

                    changed_prob = float(
                        torch.softmax(out, dim=1)[0, 1].item()
                    )

                effects.append({
                    "Feature": col,
                    "Probability Change (%)": abs(
                        changed_prob - late_probability
                    ) * 100
                })

            effects_df = pd.DataFrame(effects)

            effects_df = effects_df.sort_values(
                "Probability Change (%)",
                ascending=False
            )

            st.markdown("<br>", unsafe_allow_html=True)

            render_sensitivity(effects_df)

    # -------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------

    except Exception as error:

        st.error("The model could not be loaded or used.")

        st.write("Check that these files are in the same folder:")

        st.code(
            "app.py\n"
            "supply_chain_transformer.pth\n"
            "supply_chain_preprocessing.pkl"
        )

        st.caption(f"Technical details: {error}")


def render_about():

    section_header(
        "About",
        "A Transformer-based machine learning application for estimating "
        "late-delivery risk from order data."
    )

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="info-card">
            <h4>What it does</h4>
            <p>The app takes order and shipment details, passes them through a
            trained Transformer neural network, and returns the estimated
            probability that the order will be delivered late.</p>
            <p>A sensitivity view shows which inputs move the prediction the most.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h4>Input features</h4>
            <p><b style="color:#f5d58b">Categorical</b></p>
            <span class="chip">Shipping Mode</span>
            <span class="chip">Market</span>
            <span class="chip">Customer Segment</span>
            <p style="margin-top:10px"><b style="color:#f5d58b">Numerical</b></p>
            <span class="chip">Days for shipment (scheduled)</span>
            <span class="chip">Order Item Quantity</span>
            <span class="chip">Product Price</span>
            <span class="chip">Order Item Discount</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Please note</h4>
        <p>Predictions are model estimates, not guarantees. Inputs that are not
        shown in the form are kept at default values, so results may differ from
        a prediction that uses every available feature.</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 7. MAIN
# =====================================================

inject_css()

page = render_sidebar()

if page == NAV_ITEMS[0]:          # Home
    render_hero()
    render_overview()
    st.markdown("<br>", unsafe_allow_html=True)
    render_prediction()
    render_how_it_works()

elif page == NAV_ITEMS[1]:        # Prediction
    render_hero(compact=True)
    render_prediction()

else:                             # About
    render_hero(compact=True)
    render_about()

render_footer()
