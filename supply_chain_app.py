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

 

# Optional: put a banner photo next to app.py named hero.jpg / hero.png / hero.webp

# and it will be used automatically in the header. Otherwise a built-in

# illustration is drawn.

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

 

# Built-in banner photo (cargo ship, plane, truck and port cranes at sunset)

HERO_B64 = (

    "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"

    "2g6MjJEl+bn2Cr6rtR0NTxbzdTP7Bq+aKHTOqa9odS6fMrT5ufhX9J2ea2LwTpqFrP8ANKEPVL2X/HS9s9ZvPbTZIqHNjgdLKeOn"

    "T3WJqe0OmnYKmtt8twrs5a4OIaP2USn7L9ZvkD2UlJAD1BmarFvZTeoyXVWoKKij8wJGlU4oZBVx8FPWa8v9Y5klypAKRpBZFEdp"

    "/gr9nbNfY4BTUdkjpQAAHAgZUOr7P7HbqM1Ny1sz9sEJQ2Ds2NMDU6y3n2CW8XofL4pLyNqe1XWVUO5EzIhnqMKNV9oGrqhm2rvu"

    "5hGDsapstP2TUdI4Nvj55A04OOpVTZ9R6Lo6V5rbU+Y7jjzyEmWsfXGteYor23J1Q8yPuMpceoa3AKLvEwY0moY79HIz+y0MWv8A"

    "RkLQ6l0tI/ByPAo107RY6u9U9VS6ZkhDeMCLqs04/o1xtfpRKZsUj3PYyhq92RyQ5WNPZ7rUzmKltjyWjPjVlWdpl7a0uitJhYMf"

    "/L5yiUl81zcN1xorc9sbx0MeCkyaQ1dvbQCl03qCpecWyNrmnHJCtXaGvDXjvp6eP/uHCjQ23tFqHuqWU72l/UZwjQ6K1jX1O6Wl"

    "maPPMpSX59LQXak/MkiypNC94f6xeY48c4BCbDpyyurJIJtRYHT8KFHobUzasR/DS9eveFXcPZfeamR5rKruyfMJsK5P/aIsuriv"

    "NpTXq0We00kZt13M7yfIZyu6forBVVEzb9KGyhuRzhSNSaLrNJ2ttY6vErR5OGVYaP0RR6lszLnWSlsr/QoFRZ8mJFvkVR4/bvv9"

    "mIuIpI7hPTUL80wPhKqX/wBrH4d74Tkg+a1+urFSabqY6SBxy/zVaNPSVenftM5D4+XtH4nD2Hmssq5Qsw6ELq5Uqehh2hSTWiSh"

    "js0JYG7C44H7qro9Z3WktT7NSGNjeTv2AkZ91vbJonSupNPxVdIC3bxK3OCD7jyWsp+zvSkFO0OovvXDGcrcuPa1pyreZxq98fk8"

    "XZqvUBtbqV1zPB6iNMbcrvIY45Lm7fj0W+1nbqHS1O2GjtokMnQ4yvPXyyMBcGffAg4wslkJJ4zfx5wnDtBDxJXmV7ZK2oc53TGc"

    "FNdG6nc81cLmPA4B816Hp7VOn4qGmgr6LNU5wacxql7U5GsvrhRwhjQxrumM8JkqU4bomHJau+NxMsyhrH259wEDu5B64UTeA9kb"

    "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"

    "LH2rXmpWOkMtrh2+WXhU111bq6e5FzbjT0zPZ44TFBZjKlJv2z002N8UBaaEOzxncuts8sTu5kMLGDknvAvHKjWFbtMddrZsJHUD"

    "lVzdU2Nz3ht8q60u4y0OIJ+aNtQX3FRj3/wZ7f3tphn+HNwga7OCN4Ut7bfE2WKStjDODkEFfPskdto6KS8PtlTUt/FudIQqGPtZ"

    "t09Q2kordKJXuDRl5KZCbnH7FuAWQVbXyfk+jKm7WOlfL/XncY6MyufbVqlj+4Y+Xb+ppblfO2pu0y/2S5Cjp6Jg3sDhuGSOFtdE"

    "3G9at0y65XrVVPRMJxt2gHCXbbOqPbCVwrsl109M+15JnZhtEePImXCr669mnOypp4+/H6Hg4Xkvau9lipaFlh1E+tll/EYj0+iz"

    "vZ3qils2pJJ79FV3Eubw1zSRlArLJV94sKVUYy6+z1+r1zY6dpbNePhqgdWbMrO6s1vZ7jpKEWuvNVJE8EsLME8rJVFbLX61qLwN"

    "FyzULgdjSCOfJQr9aL1TaW+26mzNtcZkyGk+WeE2huxqM2LvhGqPaKPdtJdtRmtgttTp0ZjgLQ8njovmzVNVNWi7SkBu6UkAdByj"

    "i81/dNmhqmgFoHh6qoqp5H962fo8jK9HxuBXRsov2eZu5Xd5hi2XOtaw042AD1CTrhUbXR1DGk9QQFZ19mimqZDTv2nAUJtkuYqo"

    "6eINklf0yU+aURUfJe0naFfoLdFbqWGFsTBjJiBP1VFV3KavuJqp6cbvQDAV2NCazDC4UbQD08SBVaZv1va2Oraxrj7pEZ7IfKqS"

    "Wley4VcY3fAtLR04Wo0c2oueoBWTRNDWNPQYVZFp65P2tkqWtaVprVTwWSGQxT5d7LV1QjX+T3Tsg1za9MyXSCppmuc93BcMlaTt"

    "J7VqG66SksjbW1j5ZGeM845XzS2tmpakTRyf2nJUe63y5VcDZKio3MMjRwOeq493Fbm5J4jpVXwxJrWen6hdpyDW1g+1Kxm0U58P"

    "BwdvCxemtQNoItaRxRsf32Q0ubkgeWFodT6Lt9VBar4JpZnmEEgAnnCwOmYA6q1FC5rvwnqFx64w+NpPf/8ATtzlLtHwX/Ypc6mn"

    "gvZjbnx7s4W+uN71BIPiROIwXAbcLBdjETfgtSyGWOMRuABe4Dz8lc3K56foqqX7S1aJSwghjG5HySuRU5XtpfoZxrIRpXdkS+95"

    "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"

)

 

 

def hero_background():

    """Return a CSS background-image value for the hero banner."""

 

    # A hero.jpg / hero.png / hero.webp next to app.py overrides the built-in photo

    for name in HERO_IMAGE_CANDIDATES:

        path = BASE_DIR / name

        if path.exists():

            mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") \

                else f"image/{path.suffix.lower().strip('.')}"

            data = base64.b64encode(path.read_bytes()).decode()

            return f"url('data:{mime};base64,{data}')"

 

    return f"url('data:image/jpeg;base64,{HERO_B64}')"

 

 

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

 

/* Radio -> navigation items */

section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 6px; }}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{

    background: transparent;

    border: 1px solid transparent;

    border-left: 4px solid transparent;

    border-radius: 10px;

    padding: 14px 16px;

    margin: 0;

    width: 100%;

    transition: all .2s ease;

    cursor: pointer;

}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{

    background: rgba(232,184,91,0.07);

    border-left-color: rgba(232,184,91,0.5);

}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{

    background: linear-gradient(90deg, rgba(232,184,91,0.20), rgba(232,184,91,0.03));

    border-left-color: var(--gold);

}}

section[data-testid="stSidebar"] div[role="radiogroup"] label > div:not([data-testid="stMarkdownContainer"]) {{ display: none !important; }}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{ width: 100%; }}

section[data-testid="stSidebar"] div[role="radiogroup"] p {{

    font-size: 16px; font-weight: 600; color: var(--text);

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

 

NAV_ITEMS = ["  Home", " Prediction", " About"]

 

 

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

 

        page = st.radio(

            "Navigation",

            NAV_ITEMS,

            label_visibility="collapsed",

            key="nav_page"

        )

 

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

 

    st.markdown('<div class="how-title">⚙️ How It Works</div>', unsafe_allow_html=True)

 

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

        "🔬",

        "Model Response to Input Changes",

        "Shows how the predicted probability changes when individual visible "

        "inputs are varied. This does not establish causation."

    )

 

    max_val = max(float(effects_df["Probability Change (%)"].max()), 1e-9)

 

    rows = ""

    for _, r in effects_df.iterrows():

        v = float(r["Probability Change (%)"])

        w = max(v / max_val * 100, 1.5)

        rows += f"""

        <div class="sens-row">

            <div class="sens-name">{r['Feature']}</div>

            <div class="sens-track"><div class="sens-fill" style="width:{w:.1f}%"></div></div>

            <div class="sens-val">{v:.2f}%</div>

        </div>

        """

 

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

        "📦",

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

        "ℹ️",

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
