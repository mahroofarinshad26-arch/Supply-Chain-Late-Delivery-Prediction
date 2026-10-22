import streamlit as st
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# Image filename must match the file uploaded to GitHub
hero_image_path = BASE_DIR / "supply_chain_banner.jpeg"

# ---------------- NAVY BLUE + GOLD THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #101d32, #17243a);
    color: #F5F7FA;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101e33, #26364d);
    border-right: 1px solid #354966;
}
h1, h2, h3 {
    color: #F7D774 !important;
}
p, label, span {
    color: #F0F3F8;
}
div.stButton > button {
    background: #203958;
    color: #F7D774;
    border: 1px solid #D8B75A;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #D8B75A;
    color: #101e33;
    border: 1px solid #F7D774;
}
.card {
    background: linear-gradient(145deg, #203958, #14263f);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #405779;
    min-height: 155px;
    margin-bottom: 12px;
}
.card h3 {
    color: #F7D774;
    margin-top: 0;
}
.card p {
    color: #F0F3F8;
}
hr {
    border-color: #43536a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📦 Supply Chain")
    st.caption("INTELLIGENT LOGISTICS")
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.button("🏠  Home", use_container_width=True):
        st.session_state.page = "Home"

    st.caption("Supply Chain Intelligence")

    if st.button("📊  Prediction", use_container_width=True):
        st.session_state.page = "Prediction"

    st.caption("Late Delivery Risk Analysis")

    if st.button("ℹ️  About", use_container_width=True):
        st.session_state.page = "About"

    st.markdown("---")
    st.markdown("### Design")
    st.caption("Navy Blue & Luxury Gold")
    st.markdown("---")
    st.caption("Supply Chain Risk Analytics")

page = st.session_state.page

# ---------------- HOME PAGE ----------------
if page == "Home":
    if hero_image_path.exists():
        st.image(str(hero_image_path), use_container_width=True)
    else:
        st.warning(
            "Banner image not found. Upload the image to the same "
            "GitHub folder as supply_chain_app.py."
        )

    st.markdown("# Supply Chain Overview")
    st.write(
        "An intelligent logistics application designed to support "
        "late-delivery risk analysis using machine learning."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="card">
        <h3>Order Visibility</h3>
        <p>Review order-related information in one place.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <h3>Risk Assessment</h3>
        <p>Estimate the possibility of late delivery.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <h3>Faster Decisions</h3>
        <p>Use model output to support logistics planning.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">
        <h3>Data Insights</h3>
        <p>Turn order information into useful predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Supply Chain Risk Analytics | Transformer-based prediction")

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.markdown("# Late Delivery Risk Prediction")
    st.write(
        "Enter order details below to estimate late-delivery risk."
    )

    st.info(
        "Connect this form to your existing Transformer prediction "
        "function and preprocessing file before using it for predictions."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            days_shipping = st.number_input(
                "Days for shipping (scheduled)",
                min_value=0, value=4
            )
            days_real = st.number_input(
                "Days for shipping (real)",
                min_value=0, value=3
            )
            order_item_quantity = st.number_input(
                "Order item quantity",
                min_value=1, value=1
            )
            sales = st.number_input(
                "Sales",
                min_value=0.0, value=100.0
            )

        with col2:
            order_item_discount = st.number_input(
                "Order item discount",
                min_value=0.0, value=0.0
            )
            order_item_product_price = st.number_input(
                "Product price",
                min_value=0.0, value=100.0
            )
            benefit_per_order = st.number_input(
                "Benefit per order",
                value=10.0
            )

        submitted = st.form_submit_button(
            "🔍 Predict Late Delivery Risk",
            use_container_width=True
        )

    if submitted:
        st.warning(
            "The input form is ready, but the prediction is not connected "
            "in this template. Your original model inference code must be "
            "inserted here to display a real prediction."
        )

# ---------------- ABOUT PAGE ----------------
elif page == "About":
    st.markdown("# About This Project")

    st.markdown("""
    <div class="card">
    <h3>Supply Chain Late Delivery Prediction</h3>
    <p>
    This project focuses on estimating late-delivery risk from
    supply-chain order information using a Transformer-based
    machine-learning model.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Features")
    st.markdown("""
    - Order information input
    - Late-delivery risk analysis
    - Transformer-based model integration
    - Interactive Streamlit interface
    - Navy Blue and Luxury Gold design
    """)

    st.markdown("---")
    st.caption("Supply Chain Risk Analytics")
