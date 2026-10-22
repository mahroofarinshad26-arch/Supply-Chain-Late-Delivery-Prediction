import streamlit as st
import pandas as pd
import pickle
import torch
import torch.nn as nn
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "supply_chain_transformer.pth"
PREPROCESS_PATH = BASE_DIR / "supply_chain_preprocessing.pkl"
BANNER_PATH = BASE_DIR / "supply_chain_banner.jpeg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- TRANSFORMER MODEL ----------------
class SupplyChainTransformer(nn.Module):
    def __init__(self, category_counts, num_numerical,
                 d_model=32, nhead=4):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(int(count), d_model)
            for count in category_counts
        ])

        self.num_projection = nn.Linear(num_numerical, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

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


# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_preprocessing():
    with open(PREPROCESS_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_model():
    prep = load_preprocessing()

    model = SupplyChainTransformer(
        category_counts=prep["category_counts"],
        num_numerical=len(prep["numerical_cols"])
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# ---------------- NAVY + GOLD DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #10223a, #172940);
    color: #F4F6FA;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B192B, #1A3049);
    border-right: 1px solid #344B66;
}
h1, h2, h3 {
    color: #E7C568 !important;
}
p, label {
    color: #E1E8F0;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background: #E7C568;
    color: #071426;
    border-radius: 10px;
    border: 1px solid #F3D98C;
    font-weight: 700;
    min-height: 42px;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #F3D98C;
    color: #071426;
}
.feature-card {
    background: linear-gradient(145deg, #193451, #10243B);
    border: 1px solid #344B66;
    border-radius: 15px;
    padding: 22px;
    min-height: 145px;
}
.feature-card h3 {
    font-size: 20px;
}
.feature-card p {
    color: #D4DEEA;
}
.metric-card {
    background: #122842;
    border: 1px solid #344B66;
    border-radius: 13px;
    padding: 18px;
}
.metric-title {
    color: #BCCADA;
    font-size: 14px;
}
.metric-value {
    color: #E7C568;
    font-size: 20px;
    font-weight: bold;
}
div[data-testid="stForm"] {
    background: #10243B;
    border: 1px solid #344B66;
    border-radius: 15px;
    padding: 20px;
}
hr {
    border-color: #344B66;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("##  Supply Chain")
    st.caption("INTELLIGENT LOGISTICS")
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    if st.button(" Overview", use_container_width=True):
        st.session_state.page = "Overview"

    if st.button(" Risk Prediction", use_container_width=True):
        st.session_state.page = "Prediction"

    if st.button(" About Project", use_container_width=True):
        st.session_state.page = "About"

    st.markdown("---")
    st.caption("Late Delivery Risk Analysis")
    st.caption("Transformer-based Deep Learning")

page = st.session_state.page


# ---------------- OVERVIEW ----------------
if page == "Overview":

    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), use_container_width=True)

    st.markdown("# Supply Chain Overview")

    st.write(
        "A data-driven application for assessing late-delivery "
        "risk and supporting smarter logistics decisions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Prediction Target</div>
        <div class="metric-value">Late Delivery Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Deep Learning Model</div>
        <div class="metric-value">Transformer</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Interface</div>
        <div class="metric-value">Interactive Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Key Features")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="feature-card">
        <h3> Order Visibility</h3>
        <p>Review important order and shipping details.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="feature-card">
        <h3> Risk Assessment</h3>
        <p>Estimate late-delivery risk using the trained model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="feature-card">
        <h3>Decision Support</h3>
        <p>Present model predictions through a clear interface.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Supply Chain Risk Analytics | Transformer Model")


# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("Late Delivery Risk Prediction")
    st.write("Enter the required order information.")

    try:
        prep = load_preprocessing()
        categorical_cols = list(prep["categorical_cols"])
        numerical_cols = list(prep["numerical_cols"])
        label_encoders = prep["label_encoders"]
        scaler = prep["scaler"]
        category_counts = prep["category_counts"]

    except Exception as e:
        st.error(f"Could not load preprocessing file: {e}")
        st.stop()

    with st.form("prediction_form"):

        st.markdown("### Categorical Features")

        cat_values = {}
        left, right = st.columns(2)

        for i, col_name in enumerate(categorical_cols):
            encoder = label_encoders.get(col_name)

            with (left if i % 2 == 0 else right):
                if encoder is not None and hasattr(encoder, "classes_"):
                    options = [str(x) for x in encoder.classes_]
                    cat_values[col_name] = st.selectbox(
                        col_name, options
                    )
                else:
                    cat_values[col_name] = st.text_input(
                        col_name, value=""
                    )

        st.markdown("### Numerical Features")

        num_values = {}
        left, right = st.columns(2)

        for i, col_name in enumerate(numerical_cols):
            with (left if i % 2 == 0 else right):
                num_values[col_name] = st.number_input(
                    col_name,
                    value=0.0,
                    step=1.0
                )

        submitted = st.form_submit_button(
            "🔍 Predict Late Delivery Risk",
            use_container_width=True
        )

    if submitted:
        try:
            encoded_categories = []

            for i, col_name in enumerate(categorical_cols):
                encoder = label_encoders.get(col_name)
                raw_value = cat_values[col_name]

                if encoder is None:
                    raise ValueError(
                        f"No saved encoder found for {col_name}."
                    )

                encoded_value = int(
                    encoder.transform([raw_value])[0]
                )

                if encoded_value < 0 or encoded_value >= int(category_counts[i]):
                    raise ValueError(
                        f"Encoded category is outside the model range for {col_name}."
                    )

                encoded_categories.append(encoded_value)

            numerical_df = pd.DataFrame(
                [[num_values[col] for col in numerical_cols]],
                columns=numerical_cols
            )

            scaled_values = scaler.transform(numerical_df)

            categorical_tensor = torch.tensor(
                [encoded_categories],
                dtype=torch.long,
                device=DEVICE
            )

            numerical_tensor = torch.tensor(
                scaled_values,
                dtype=torch.float32,
                device=DEVICE
            )

            model = load_model()

            with torch.no_grad():
                output = model(
                    categorical_tensor,
                    numerical_tensor
                )
                probabilities = torch.softmax(
                    output, dim=1
                )[0].cpu().numpy()

            predicted_class = int(probabilities.argmax())

            st.markdown("---")
            st.markdown("### Prediction Result")

            if predicted_class == 1:
                st.error("⚠️ Late Delivery Risk Detected")
                st.metric(
                    "Class 1 Probability",
                    f"{probabilities[1] * 100:.2f}%"
                )
            else:
                st.success("✅ Class 0 Predicted")
                st.metric(
                    "Class 0 Probability",
                    f"{probabilities[0] * 100:.2f}%"
                )

            st.write("Class 0 probability:", f"{probabilities[0] * 100:.2f}%")
            st.write("Class 1 probability:", f"{probabilities[1] * 100:.2f}%")
            st.caption(
                "Class 0/1 meaning must match the target encoding used during training."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Check that the saved model, feature order, encoders, "
                "scaler and training architecture match."
            )


# ---------------- ABOUT ----------------
elif page == "About":

    st.title("About This Project")

    st.markdown("""
    <div class="feature-card">
    <h3>Supply Chain Late Delivery Prediction</h3>
    <p>
    This project uses order and shipping information to estimate
    late-delivery risk with a Transformer-based deep-learning model.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Main Components")
    st.markdown("""
    - Data preprocessing using saved label encoders and scaler
    - Transformer model for binary classification
    - Interactive Streamlit prediction interface
    - Navy Blue and Luxury Gold dashboard design
    """)

    st.markdown("### Technology Stack")
    st.write("Python · PyTorch · Pandas · Scikit-learn · Streamlit")

    st.markdown("---")
    st.caption("Supply Chain Risk Analytics")
