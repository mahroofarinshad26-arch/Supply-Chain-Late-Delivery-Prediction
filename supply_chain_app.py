import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path

# 1. PAGE SETTINGS
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "supply_chain_transformer.pth"
PREPROCESS_PATH = BASE_DIR / "supply_chain_preprocessing.pkl"

# 2. DESIGN
st.markdown("""
<style>
.stApp { background-color: #0b1120; color: #f8fafc; }
h1, h2, h3 { color: #f5c76b !important; }
div[data-testid="stForm"] {
    background-color: #111c30; padding: 22px; border-radius: 14px;
    border: 1px solid #293750;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background-color: #e8b85b; color: #111827; font-weight: bold;
    border-radius: 8px; border: none; width: 100%;
}
div[data-testid="stMetric"] {
    background-color: #111c30; padding: 18px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# 3. TRANSFORMER MODEL — must match the training architecture
class SupplyChainTransformer(nn.Module):
    def __init__(self, category_counts, num_numerical, d_model=32, nhead=4):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(count), d_model) for count in category_counts
        ])
        self.num_projection = nn.Linear(num_numerical, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
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
        numerical_token = self.num_projection(numerical).unsqueeze(1)
        cls = self.cls_token.expand(categorical.size(0), -1, -1)
        tokens = torch.cat([cls, embedded, numerical_token], dim=1)
        encoded = self.transformer(tokens)
        return self.classifier(encoded[:, 0, :])

# 4. LOAD MODEL AND PREPROCESSING
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, categorical_cols, numerical_cols, label_encoders, scaler

# 5. SIDEBAR
with st.sidebar:
    st.markdown("# 📦 Supply Chain")
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

# 6. HERO BANNER WITH LOGISTICS IMAGE
hero_image_path = BASE_DIR / "supply_chain_banner.jpeg"

if hero_image_path.exists():
    st.image(str(hero_image_path), use_container_width=True)
else:
    # Fallback design: app still works if the optional picture is not uploaded.
    st.markdown("""
    <div style="background:linear-gradient(120deg,#06172d,#17385a);
    border:1px solid #293750;border-radius:18px;padding:30px;margin-bottom:20px;">
    <p style="color:#e8b85b;letter-spacing:2px;font-size:12px;">
    SMARTER DECISIONS · STRONGER SUPPLY CHAINS</p>
    <h1 style="color:#f5c76b;margin-bottom:8px;">Supply Chain Intelligence</h1>
    <h3 style="color:#f8fafc !important;">AI-Powered Late Delivery Risk Prediction</h3>
    <p style="color:#cbd5e1;">Analyze order details, estimate delivery risk, and support
    smarter logistics decisions with a trained Transformer model.</p>
    </div>
    """, unsafe_allow_html=True)

# 7. FEATURE CARDS
st.markdown("## Supply Chain Overview")
c1, c2, c3, c4 = st.columns(4)
features = [
    (c1, "Order Visibility", "Review order-related information in one place."),
    (c2, "Risk Assessment", "Estimate the possibility of late delivery."),
    (c3, "Faster Decisions", "Use model output to support logistics planning."),
    (c4, "Data Insights", "Turn order information into useful predictions.")
]
for col, heading, description in features:
    with col:
        st.markdown(f"""
        <div style="background:#111c30;border:1px solid #293750;
        border-radius:14px;padding:18px;min-height:145px;">
        <h3 style="color:#f5c76b;">{heading}</h3>
        <p style="color:#cbd5e1;">{description}</p></div>
        """, unsafe_allow_html=True)

st.divider()

# 8. PREDICTION FORM — only these 7 fields are shown
st.markdown("## Late Delivery Risk Analysis")
st.write(
    "Enter the details below. Other features required by the trained model "
    "will use default values based on the saved preprocessing data."
)

try:
    model, categorical_cols, numerical_cols, label_encoders, scaler = (
        load_model_and_preprocessing()
    )
    st.success("Trained Transformer model loaded successfully.")

    wanted_categorical = ["Shipping Mode", "Market", "Customer Segment"]
    wanted_numerical = [
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Product Price",
        "Order Item Discount"
    ]

    visible_categorical = [
        col for col in wanted_categorical
        if col in categorical_cols and col in label_encoders
    ]
    visible_numerical = [
        col for col in wanted_numerical if col in numerical_cols
    ]

    missing_fields = [
        col for col in wanted_categorical + wanted_numerical
        if col not in categorical_cols and col not in numerical_cols
    ]
    if not visible_categorical and not visible_numerical:
        st.error(
            "The selected field names do not match the saved preprocessing "
            "columns. Check the column names in supply_chain_preprocessing.pkl."
        )
        st.stop()

    if missing_fields:
        st.warning(
            "These fields are not in the saved preprocessing data and will "
            "not be shown: " + ", ".join(missing_fields)
        )

    with st.form("prediction_form"):
        categorical_values = {}
        numerical_values = {}

        st.markdown("### Categorical Details")
        cat_col1, cat_col2 = st.columns(2)
        for i, col in enumerate(visible_categorical):
            options = list(label_encoders[col].classes_)
            target_col = cat_col1 if i % 2 == 0 else cat_col2
            with target_col:
                categorical_values[col] = st.selectbox(
                    col, options=options, key=f"cat_{col}"
                )

        st.markdown("### Numerical Details")
        num_col1, num_col2 = st.columns(2)
        integer_fields = {
            "Days for shipment (scheduled)", "Order Item Quantity"
        }

        for i, col in enumerate(visible_numerical):
            scaler_index = numerical_cols.index(col)
            default_value = float(scaler.mean_[scaler_index])
            target_col = num_col1 if i % 2 == 0 else num_col2
            with target_col:
                if col in integer_fields:
                    numerical_values[col] = st.number_input(
                        col, value=max(0, int(round(default_value))),
                        min_value=0, step=1, key=f"num_{col}"
                    )
                else:
                    numerical_values[col] = st.number_input(
                        col, value=max(0.0, default_value),
                        min_value=0.0, format="%.4f", key=f"num_{col}"
                    )

        submitted = st.form_submit_button(
            "🔎 Analyze Delivery Risk", use_container_width=True
        )

    # 9. PREDICTION RESULT
    if submitted:
        # Initialize every categorical feature with its first known category.
        encoded_values = []
        for col in categorical_cols:
            options = list(label_encoders[col].classes_)
            encoded_values.append(
                int(label_encoders[col].transform([options[0]])[0])
            )

        # Replace defaults with the user's selected categories.
        for col, value in categorical_values.items():
            col_index = categorical_cols.index(col)
            encoded_values[col_index] = int(
                label_encoders[col].transform([value])[0]
            )

        # Initialize every numerical feature to the training mean.
        numerical_row = np.asarray(scaler.mean_, dtype=float).copy()
        for col, value in numerical_values.items():
            numerical_row[numerical_cols.index(col)] = float(value)

        numerical_df = pd.DataFrame([numerical_row], columns=numerical_cols)
        scaled_values = scaler.transform(numerical_df)

        categorical_tensor = torch.tensor([encoded_values], dtype=torch.long)
        numerical_tensor = torch.tensor(scaled_values, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(categorical_tensor, numerical_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            late_probability = float(probabilities[1].item())

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

        # 10. SENSITIVITY ANALYSIS — visible inputs only
        st.markdown("### 🔎 Model Explanation")
        st.caption(
            "These values show how the model responds to changes in inputs. "
            "They are not confirmed causes of delivery delays."
        )

        baseline_probability = late_probability
        feature_effects = []

        for col in visible_categorical:
            col_index = categorical_cols.index(col)
            probabilities_for_feature = []

            for option in label_encoders[col].classes_:
                changed_categorical = encoded_values.copy()
                changed_categorical[col_index] = int(
                    label_encoders[col].transform([option])[0]
                )
                cat_tensor = torch.tensor(
                    [changed_categorical], dtype=torch.long
                )
                with torch.no_grad():
                    changed_output = model(cat_tensor, numerical_tensor)
                    changed_probability = float(
                        torch.softmax(changed_output, dim=1)[0, 1].item()
                    )
                probabilities_for_feature.append(changed_probability)

            effect = max(probabilities_for_feature) - min(
                probabilities_for_feature
            )
            feature_effects.append({
                "Feature": col, "Probability range (%)": effect * 100
            })

        for col in visible_numerical:
            col_index = numerical_cols.index(col)
            changed_values = scaled_values.copy()
            changed_values[0, col_index] += 0.5
            changed_tensor = torch.tensor(
                changed_values, dtype=torch.float32
            )
            with torch.no_grad():
                changed_output = model(categorical_tensor, changed_tensor)
                changed_probability = float(
                    torch.softmax(changed_output, dim=1)[0, 1].item()
                )
            effect = abs(changed_probability - baseline_probability)
            feature_effects.append({
                "Feature": col, "Probability range (%)": effect * 100
            })

        if feature_effects:
            effects_df = pd.DataFrame(feature_effects).sort_values(
                "Probability range (%)", ascending=False
            )
            st.markdown("**Inputs with the strongest model response**")
            st.dataframe(
                effects_df.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Sensitivity indicates model response, not causation. Hidden "
            "features remain at default values, so this is an estimate."
        )

except Exception as error:
    st.error("The model could not be loaded or used.")
    st.write("Ensure these files are in the same folder as supply_chain_app.py:")
    st.code(
        "supply_chain_app.py\n"
        "supply_chain_transformer.pth\n"
        "supply_chain_preprocessing.pkl"
    )
    st.caption(f"Technical details: {error}")

# 11. HOW IT WORKS
st.divider()
st.markdown("## How It Works")
s1, s2, s3, s4 = st.columns(4)
steps = [
    (s1, "01", "Enter Details", "Provide order features."),
    (s2, "02", "Model Analysis", "The Transformer processes the input."),
    (s3, "03", "Get Prediction", "Review the estimated risk."),
    (s4, "04", "Plan Ahead", "Use the estimate to support decisions.")
]
for col, number, heading, description in steps:
    with col:
        st.markdown(f"""
        <div style="background:#111c30;border:1px solid #293750;
        border-radius:14px;padding:16px;min-height:145px;">
        <p style="color:#e8b85b;font-size:13px;">{number}</p>
        <h3 style="color:#f5c76b;">{heading}</h3>
        <p style="color:#cbd5e1;">{description}</p></div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:24px;color:#cbd5e1;">
Supply Chain Risk Analytics | Transformer-Based Prediction Project
</div>
""", unsafe_allow_html=True)
