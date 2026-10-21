import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path

# ==================================================
# 1. PAGE SETTINGS
# ==================================================
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "supply_chain_transformer.pth"
PREPROCESS_PATH = BASE_DIR / "supply_chain_preprocessing.pkl"

# ==================================================
# 2. PROFESSIONAL NAVY & GOLD DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #0B1D35);
    color: #F3F6FC;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 0rem;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #061020;
    border-right: 1px solid #B89442;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F2CC70 !important;
}

h1, h2, h3 {
    color: #F2CC70 !important;
    letter-spacing: 0.2px;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #E5EAF3;
}

.hero {
    background: linear-gradient(115deg, #102A49, #16416A);
    border: 1px solid #B89442;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 0 0 20px 0;
}

.hero-kicker {
    color: #F2CC70;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: #FFFFFF;
    font-size: clamp(30px, 3.2vw, 46px);
    font-weight: 800;
    line-height: 1.15;
    margin: 12px 0;
}

.hero-subtitle {
    color: #D6E2F2;
    font-size: 17px;
    line-height: 1.6;
}

.info-card {
    background: #10243D;
    border: 1px solid #294764;
    border-radius: 13px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 12px;
}

.info-card h3 {
    color: #F2CC70 !important;
    font-size: 17px;
    margin: 0 0 10px 0;
}

.info-card p {
    color: #D5DEEB;
    font-size: 14px;
    margin-bottom: 0;
    line-height: 1.5;
}

[data-testid="stExpander"] {
    background: #0C1C30;
    border: 1px solid #294764;
    border-radius: 12px;
}

[data-testid="stForm"] {
    background: #0C1C30;
    border: 1px solid #294764;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stWidgetLabel"] p {
    color: #F1F5F9 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #D8AD4A, #F2CC70);
    color: #071426 !important;
    border: none;
    border-radius: 9px;
    min-height: 46px;
    font-weight: 800;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #FFE09A;
    color: #071426 !important;
}

[data-testid="stMetric"] {
    background: #10243D;
    border: 1px solid #B89442;
    border-radius: 12px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #D6E2F2 !important;
}

[data-testid="stMetricValue"] {
    color: #F2CC70 !important;
}

hr {
    border-color: #294764;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #A8B8CC;
    font-size: 12px;
    padding: 18px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. TRANSFORMER MODEL
# ==================================================
class SupplyChainTransformer(nn.Module):
    def __init__(
        self, category_counts, num_numerical,
        d_model=32, nhead=4
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(int(count), d_model)
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
            [cls, embedded, numerical_token], dim=1
        )

        encoded = self.transformer(tokens)
        return self.classifier(encoded[:, 0, :])


# ==================================================
# 4. LOAD MODEL AND PREPROCESSING
# ==================================================
@st.cache_resource
def load_model_and_preprocessing():
    with open(PREPROCESS_PATH, "rb") as file:
        prep = pickle.load(file)

    categorical_cols = prep["categorical_cols"]
    numerical_cols = prep["numerical_cols"]
    label_encoders = prep["label_encoders"]
    scaler = prep["scaler"]
    category_counts = prep["category_counts"]

    model = SupplyChainTransformer(
        category_counts=category_counts,
        num_numerical=len(numerical_cols)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return (
        model,
        categorical_cols,
        numerical_cols,
        label_encoders,
        scaler
    )


# ==================================================
# 5. SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("# Supply Chain")
    st.caption("INTELLIGENT LOGISTICS")
    st.divider()

    st.markdown("### Home")
    st.caption("Supply Chain Intelligence")

    st.markdown("### Prediction")
    st.caption("Late Delivery Risk Analysis")

    st.markdown("### About")
    st.write(
        "A Transformer-based machine learning application "
        "for estimating late-delivery risk from order data."
    )

    st.divider()
    st.markdown("### Design")
    st.caption("Navy Blue & Luxury Gold")

    st.divider()
    st.caption("Supply Chain Risk Analytics")
    st.caption("DataCo project")


# ==================================================
# 6. TOP HERO BANNER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">
        SMARTER DECISIONS · STRONGER SUPPLY CHAINS
    </div>
    <div class="hero-title">
        Supply Chain Intelligence
    </div>
    <div style="color:#F2CC70;font-size:21px;font-weight:700;">
        AI-Powered Late Delivery Risk Prediction
    </div>
    <div class="hero-subtitle">
        Analyze order details, estimate delivery risk, and support
        smarter logistics decisions with a trained Transformer model.
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 7. FEATURE CARDS
# ==================================================
st.markdown("## Supply Chain Overview")

c1, c2, c3, c4 = st.columns(4)

features = [
    (
        c1,
        "Order Visibility",
        "Review order-related information in one place."
    ),
    (
        c2,
        "Risk Assessment",
        "Estimate the possibility of late delivery."
    ),
    (
        c3,
        "Faster Decisions",
        "Use model output to support logistics planning."
    ),
    (
        c4,
        "Data Insights",
        "Turn order information into useful predictions."
    )
]

for col, heading, description in features:
    with col:
        st.markdown(f"""
        <div class="info-card">
            <h3>{heading}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ==================================================
# 8. PREDICTION SECTION
# ==================================================
st.markdown("## Late Delivery Risk Analysis")

st.write(
    "Choose the order categories and enter numerical details. "
    "The default numerical values are only starting points; "
    "replace them with realistic order information when available."
)

try:
    (
        model,
        categorical_cols,
        numerical_cols,
        label_encoders,
        scaler
    ) = load_model_and_preprocessing()

    st.success("Trained Transformer model loaded successfully.")

    with st.form("prediction_form"):
        categorical_values = {}
        numerical_values = {}

        with st.expander(
            "1. Categorical Order Features",
            expanded=True
        ):
            st.caption(
                "Select categories available in the training data."
            )

            cat_col1, cat_col2 = st.columns(2)

            for i, col in enumerate(categorical_cols):
                options = list(label_encoders[col].classes_)

                target_col = (
                    cat_col1 if i % 2 == 0 else cat_col2
                )

                with target_col:
                    categorical_values[col] = st.selectbox(
                        col,
                        options=options,
                        key=f"cat_{i}"
                    )

        with st.expander(
            "2. Numerical Order Features",
            expanded=False
        ):
            st.caption(
                "Defaults are training-set means, not actual order values."
            )

            num_col1, num_col2 = st.columns(2)

            integer_fields = {
                "Days for shipment (scheduled)",
                "Category Id",
                "Customer Zipcode",
                "Department Id",
                "Order Item Cardprod Id",
                "Order Item Quantity",
                "Product Card Id",
                "Product Category Id",
                "Product Status",
                "Order_Year",
                "Order_Month",
                "Order_Day",
                "Order_DayOfWeek"
            }

            for i, col in enumerate(numerical_cols):
                default_value = float(scaler.mean_[i])

                target_col = (
                    num_col1 if i % 2 == 0 else num_col2
                )

                with target_col:
                    if col in integer_fields:
                        numerical_values[col] = st.number_input(
                            col,
                            value=int(round(default_value)),
                            step=1,
                            key=f"num_{i}"
                        )
                    else:
                        numerical_values[col] = st.number_input(
                            col,
                            value=default_value,
                            format="%.4f",
                            key=f"num_{i}"
                        )

        submitted = st.form_submit_button(
            "Analyze Delivery Risk",
            use_container_width=True
        )

    # ==================================================
    # 9. PREDICTION RESULT
    # ==================================================
    if submitted:
        encoded_values = []

        for col in categorical_cols:
            encoded = label_encoders[col].transform(
                [categorical_values[col]]
            )[0]

            encoded_values.append(int(encoded))

        numerical_df = pd.DataFrame(
            [[numerical_values[col] for col in numerical_cols]],
            columns=numerical_cols
        )

        scaled_values = scaler.transform(numerical_df)

        categorical_tensor = torch.tensor(
            [encoded_values],
            dtype=torch.long
        )

        numerical_tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32
        )

        with torch.no_grad():
            outputs = model(
                categorical_tensor,
                numerical_tensor
            )

            probabilities = torch.softmax(outputs, dim=1)[0]

            predicted_class = int(
                torch.argmax(probabilities).item()
            )

            late_probability = float(
                probabilities[1].item()
            )

        st.divider()
        st.markdown("## Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if predicted_class == 1:
                st.error("⚠️ Late Delivery Risk: HIGH")
            else:
                st.success("✅ Late Delivery Risk: LOW")

        with result_col2:
            st.metric(
                "Estimated Late-Risk Probability",
                f"{late_probability * 100:.2f}%"
            )

        st.progress(late_probability)

        # ==================================================
        # MODEL EXPLANATION
        # ==================================================
        st.markdown("### 🔎 Model Explanation")

        st.caption(
            "These results show how the model responds to changes "
            "in individual inputs. They are not confirmed causes "
            "of delivery delays."
        )

        baseline_probability = late_probability
        feature_effects = []

        # Categorical feature sensitivity
        for col_index, col in enumerate(categorical_cols):
            probabilities_for_feature = []

            options = list(label_encoders[col].classes_)

            for option in options:
                option_encoded = int(
                    label_encoders[col].transform([option])[0]
                )

                changed_categorical = encoded_values.copy()
                changed_categorical[col_index] = option_encoded

                cat_tensor = torch.tensor(
                    [changed_categorical],
                    dtype=torch.long
                )

                with torch.no_grad():
                    changed_output = model(
                        cat_tensor,
                        numerical_tensor
                    )

                    changed_probability = float(
                        torch.softmax(
                            changed_output, dim=1
                        )[0, 1].item()
                    )

                probabilities_for_feature.append(
                    changed_probability
                )

            if probabilities_for_feature:
                effect = (
                    max(probabilities_for_feature)
                    - min(probabilities_for_feature)
                )

                feature_effects.append({
                    "Feature": col,
                    "Probability range": effect * 100
                })

        # Numerical feature sensitivity
        for col_index, col in enumerate(numerical_cols):
            changed_values = scaled_values.copy()
            changed_values[0, col_index] += 0.5

            changed_tensor = torch.tensor(
                changed_values,
                dtype=torch.float32
            )

            with torch.no_grad():
                changed_output = model(
                    categorical_tensor,
                    changed_tensor
                )

                changed_probability = float(
                    torch.softmax(
                        changed_output, dim=1
                    )[0, 1].item()
                )

            effect = abs(
                changed_probability - baseline_probability
            )

            feature_effects.append({
                "Feature": col,
                "Probability range": effect * 100
            })

        if feature_effects:
            effects_df = pd.DataFrame(feature_effects)

            effects_df = effects_df.sort_values(
                "Probability range",
                ascending=False
            )

            st.markdown(
                "**Inputs with the strongest model response**"
            )

            st.dataframe(
                effects_df.head(5).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "For categorical features, the range compares available "
            "categories. For numerical features, it measures the "
            "response to a small increase in the scaled input. "
            "These values indicate model sensitivity, not causation."
        )

except Exception as error:
    st.error("The model could not be loaded or used.")

    st.write(
        "Ensure these files are in the same folder as "
        "supply_chain_app.py:"
    )

    st.code(
        "supply_chain_app.py\n"
        "supply_chain_transformer.pth\n"
        "supply_chain_preprocessing.pkl"
    )

    st.caption(f"Technical details: {error}")


# ==================================================
# 10. HOW IT WORKS
# ==================================================
st.divider()
st.markdown("## How It Works")

s1, s2, s3, s4 = st.columns(4)

steps = [
    (
        s1,
        "01",
        "Enter Details",
        "Provide order features."
    ),
    (
        s2,
        "02",
        "Model Analysis",
        "The Transformer processes the input."
    ),
    (
        s3,
        "03",
        "Get Prediction",
        "Review the estimated risk."
    ),
    (
        s4,
        "04",
        "Plan Ahead",
        "Use the estimate to support decisions."
    )
]

for col, number, heading, description in steps:
    with col:
        st.markdown(f"""
        <div class="info-card">
            <div style="color:#F2CC70;font-size:26px;font-weight:800;">
                {number}
            </div>
            <h3>{heading}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Supply Chain Risk Analytics | Transformer-Based Prediction Project
</div>
""", unsafe_allow_html=True)
