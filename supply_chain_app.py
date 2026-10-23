import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path

# =====================================================
# 1. PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "supply_chain_transformer.pth"
PREPROCESS_PATH = BASE_DIR / "supply_chain_preprocessing.pkl"


# =====================================================
# 2. TRANSFORMER MODEL
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
# 3. PROFESSIONAL DARK THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg, #0b1120, #14243b
    );
    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.2rem !important;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #f5c76b !important;
}

p, label, .stMarkdown {
    color: #f1f5f9;
}

section[data-testid="stSidebar"] {
    background-color: #080e1c;
    border-right: 1px solid #293750;
}

div[data-testid="stForm"] {
    background-color: #111c30;
    padding: 22px;
    border: 1px solid #293750;
    border-radius: 14px;
}

div[data-testid="stMetric"] {
    background-color: #111c30;
    padding: 18px;
    border: 1px solid #293750;
    border-radius: 12px;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(
        90deg, #e8b85b, #f5d58b
    );
    color: #111827;
    font-weight: bold;
    border: none;
    border-radius: 9px;
    min-height: 46px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #ffd778;
    color: #111827;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

input, textarea {
    color: #f8fafc !important;
}

hr {
    border-color: #344155;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD MODEL AND PREPROCESSING
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
# 5. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        "<h2>📦 Supply Chain</h2>",
        unsafe_allow_html=True
    )

    st.caption("INTELLIGENT LOGISTICS")

    st.divider()

    st.markdown("### Dashboard")
    st.write("Late Delivery Risk Prediction")

    st.markdown("### Model")
    st.write("Transformer Neural Network")

    st.markdown("### Input Features")
    st.write("3 Categorical")
    st.write("4 Numerical")

    st.divider()

    st.caption("DataCo Supply Chain Project")


# =====================================================
# 6. MAIN HEADER
# =====================================================

st.markdown("""
<div style="
    background: linear-gradient(120deg, #111c30, #1b304b);
    border: 1px solid #344763;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
">

<p style="
    color: #e8b85b;
    letter-spacing: 2px;
    font-size: 12px;
">
SUPPLY CHAIN • RISK ANALYTICS
</p>

<h1 style="font-size: 34px;">
Supply Chain Intelligence
</h1>

<h3 style="color: #f8fafc !important;">
AI-Powered Late Delivery Risk Prediction
</h3>

<p style="color: #cbd5e1;">
Analyze order details and estimate delivery risk
using a trained Transformer model.
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. DASHBOARD OVERVIEW
# =====================================================

st.markdown("## Dashboard Overview")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div style="
        background:#111c30;
        border:1px solid #293750;
        padding:20px;
        border-radius:14px;
        min-height:135px;
    ">
        <p style="color:#94a3b8;">MODEL</p>
        <h3>Transformer</h3>
        <p>Deep learning prediction</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="
        background:#111c30;
        border:1px solid #293750;
        padding:20px;
        border-radius:14px;
        min-height:135px;
    ">
        <p style="color:#94a3b8;">INPUT FEATURES</p>
        <h3>7 Features</h3>
        <p>Order and shipment details</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="
        background:#111c30;
        border:1px solid #293750;
        padding:20px;
        border-radius:14px;
        min-height:135px;
    ">
        <p style="color:#94a3b8;">OBJECTIVE</p>
        <h3>Risk Analysis</h3>
        <p>Estimate late-delivery risk</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# =====================================================
# 8. PREDICTION INPUTS
# =====================================================

st.markdown("## Late Delivery Risk Analysis")

st.write("Enter the following seven details.")

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


# =====================================================
# 9. PREDICTION APPLICATION
# =====================================================

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
            "🔍 Predict Delivery Risk",
            use_container_width=True
        )


    # =================================================
    # 10. PREDICTION RESULT
    # =================================================

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

        st.divider()

        st.markdown("## Prediction Result")

        result_col, probability_col = st.columns(2)

        with result_col:

            if predicted_class == 1:
                st.error("⚠️ High Late-Delivery Risk")
            else:
                st.success("✅ Low Late-Delivery Risk")

        with probability_col:

            st.metric(
                "Late-Delivery Probability",
                f"{late_probability * 100:.2f}%"
            )

        st.progress(late_probability)

        st.caption(
            "This result is a model estimate, not a guarantee. "
            "Confirm that class 1 represents late delivery "
            "in your original training labels."
        )


        # =============================================
        # 11. INPUT SENSITIVITY
        # =============================================

        st.markdown("### Model Response to Input Changes")

        st.caption(
            "Shows how the predicted probability changes "
            "when individual visible inputs are varied. "
            "This does not establish causation."
        )

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
# 12. ERROR HANDLING
# =====================================================

except Exception as error:

    st.error("The model could not be loaded or used.")

    st.write("Check that these files are in the same folder:")

    st.code(
        "supply_chain_app.py\n"
        "supply_chain_transformer.pth\n"
        "supply_chain_preprocessing.pkl"
    )

    st.caption(f"Technical details: {error}")


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:22px;
    color:#94a3b8;
">

<p style="
    color:#e8b85b;
    letter-spacing:2px;
">
SUPPLY CHAIN INTELLIGENCE
</p>

Transformer-Based Late Delivery Risk Prediction

<p style="font-size:12px;">
DataCo Supply Chain Project
</p>

</div>
""", unsafe_allow_html=True)
```
