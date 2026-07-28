import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ─── Page configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title='SuperKart Sales Forecaster',
    page_icon='🛒',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*='css'] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1040 50%, #0d1f3c 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid='stSidebar'] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
}

[data-testid='stSidebar'] * {
    color: #e2e8f0 !important;
}

/* Sidebar header */
.sidebar-header {
    background: linear-gradient(135deg, #6c63ff, #3ecfff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    animation: slideDown 0.6s ease-out;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #3ecfff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.55);
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #6c63ff, #3ecfff, transparent);
    margin: 1.5rem 0;
    border-radius: 2px;
}

/* Info cards row */
.info-grid {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
    justify-content: center;
}

.info-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    flex: 1;
    min-width: 140px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.info-card:hover {
    transform: translateY(-3px);
    border-color: rgba(108, 99, 255, 0.4);
}

.info-card-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #3ecfff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.info-card-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Prediction card */
.prediction-card {
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 600px;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(255,255,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}

.prediction-card.high {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,95,70,0.4));
    border: 1px solid rgba(16,185,129,0.35);
}

.prediction-card.medium {
    background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(120,53,15,0.4));
    border: 1px solid rgba(245,158,11,0.35);
}

.prediction-card.low {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(127,29,29,0.4));
    border: 1px solid rgba(239,68,68,0.35);
}

.prediction-amount {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.5rem 0;
    color: #ffffff;
}

.prediction-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}

.badge-high   { background: rgba(16,185,129,0.25); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
.badge-medium { background: rgba(245,158,11,0.25); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
.badge-low    { background: rgba(239,68,68,0.25);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.4);  }

.prediction-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c63ff, #3ecfff);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35);
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.5);
}

/* Slider and select styling */
.stSlider > div, .stSelectbox > div {
    color: #e2e8f0;
}

/* Animations */
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.3);
    margin: 1.4rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.waiting-card {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 500px;
    color: rgba(255,255,255,0.4);
}
</style>
""", unsafe_allow_html=True)

# ─── Load model (cached so it only downloads once) ────────────────────────
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id='Sadhana3105/superkart-sales-model', filename='best_superkart_model_v1.joblib')
    return joblib.load(path)

model = load_model()

# ─── Hero section ─────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>🛒 SuperKart Sales Forecaster</div>
    <div class='hero-subtitle'>AI-powered product sales prediction — enter store & product details to forecast revenue</div>
</div>
<div class='gradient-divider'></div>
""", unsafe_allow_html=True)

# Info stats row
st.markdown("""
<div class='info-grid'>
    <div class='info-card'><div class='info-card-value'>8,764</div><div class='info-card-label'>Records Trained On</div></div>
    <div class='info-card'><div class='info-card-value'>XGBoost</div><div class='info-card-label'>Model Architecture</div></div>
    <div class='info-card'><div class='info-card-value'>9</div><div class='info-card-label'>Input Features</div></div>
    <div class='info-card'><div class='info-card-value'>₹ Sales</div><div class='info-card-label'>Prediction Target</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar inputs ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-header'>⚙️ Product & Store Details</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Product Attributes</div>", unsafe_allow_html=True)
    Product_Weight = st.slider('Product Weight (kg)', 1.0, 25.0, 12.0, 0.1)
    Product_Allocated_Area = st.slider('Allocated Display Area Ratio', 0.00, 0.35, 0.05, 0.01)
    Product_MRP = st.slider('Maximum Retail Price (₹)', 50, 300, 150)
    Product_Sugar_Content = st.selectbox('Sugar Content', ['Low Sugar', 'Regular', 'No Sugar'])
    Product_Type = st.selectbox('Product Category', [
        'Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods',
        'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene',
        'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',
        'Others', 'Starchy Foods', 'Breakfast', 'Seafood'
    ])

    st.markdown("<div class='section-header'>Store Attributes</div>", unsafe_allow_html=True)
    Store_Age = st.slider('Years Since Store Opened', 1, 40, 15)
    Store_Size = st.selectbox('Store Size', ['High', 'Medium', 'Small'])
    Store_Location_City_Type = st.selectbox('City Tier', ['Tier 1', 'Tier 2', 'Tier 3'])
    Store_Type = st.selectbox('Store Type', [
        'Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'
    ])

    predict_btn = st.button('🔮  Forecast Sales Revenue', use_container_width=True)

# ─── Prediction output ────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([{
        'Product_Weight':           Product_Weight,
        'Product_Allocated_Area':   Product_Allocated_Area,
        'Product_MRP':              Product_MRP,
        'Store_Age':                Store_Age,
        'Product_Sugar_Content':    Product_Sugar_Content,
        'Product_Type':             Product_Type,
        'Store_Size':               Store_Size,
        'Store_Location_City_Type': Store_Location_City_Type,
        'Store_Type':               Store_Type,
    }])

    prediction = model.predict(input_df)[0]

    # Classify into sales tiers
    if prediction >= 4000:
        tier, tier_label, badge_cls = 'high', 'High Revenue Product', 'badge-high'
    elif prediction >= 2500:
        tier, tier_label, badge_cls = 'medium', 'Moderate Revenue Product', 'badge-medium'
    else:
        tier, tier_label, badge_cls = 'low', 'Low Revenue Product', 'badge-low'

    st.markdown(f"""
    <div class='prediction-card {tier}'>
        <div class='prediction-badge {badge_cls}'>{tier_label}</div>
        <div class='prediction-amount'>₹ {prediction:,.0f}</div>
        <div class='prediction-label'>Estimated Product Store Sales Total</div>
    </div>
    """, unsafe_allow_html=True)

    # Mini summary of inputs
    with st.expander('📋 View Input Summary'):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

else:
    st.markdown("""
    <div class='waiting-card'>
        <div style='font-size:3rem;margin-bottom:1rem;'>📊</div>
        <div style='font-size:1.1rem;font-weight:600;color:rgba(255,255,255,0.55);'>Awaiting Prediction</div>
        <div style='font-size:0.85rem;margin-top:0.5rem;'>Fill in the product and store details in the sidebar, then click <strong>Forecast Sales Revenue</strong>.</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────
st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:rgba(255,255,255,0.25);font-size:0.78rem;padding:1rem 0 2rem;'>
    SuperKart Sales Forecaster &nbsp;·&nbsp; Powered by XGBoost &nbsp;·&nbsp; Deployed on Hugging Face Spaces
</div>
""", unsafe_allow_html=True)
