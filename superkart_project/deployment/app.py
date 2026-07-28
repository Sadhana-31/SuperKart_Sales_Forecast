import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model
model_path = hf_hub_download(
    repo_id="Sadhana3105/superkart-sales-model",
    filename="best_superkart_model_v1.joblib"
)
model = joblib.load(model_path)

def predict_sales(
    product_weight, product_allocated_area, product_mrp, store_age,
    product_sugar_content, product_type, store_size,
    store_location_city_type, store_type
):
    input_df = pd.DataFrame([{
        "Product_Weight":           product_weight,
        "Product_Allocated_Area":   product_allocated_area,
        "Product_MRP":              product_mrp,
        "Store_Age":                store_age,
        "Product_Sugar_Content":    product_sugar_content,
        "Product_Type":             product_type,
        "Store_Size":               store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type":               store_type,
    }])

    prediction = model.predict(input_df)[0]

    if prediction >= 4000:
        tier     = "HIGH REVENUE"
        color    = "#10b981"
        bg_color = "rgba(16,185,129,0.15)"
        border   = "rgba(16,185,129,0.4)"
        emoji    = "&#x1F7E2;"
    elif prediction >= 2500:
        tier     = "MODERATE REVENUE"
        color    = "#f59e0b"
        bg_color = "rgba(245,158,11,0.15)"
        border   = "rgba(245,158,11,0.4)"
        emoji    = "&#x1F7E1;"
    else:
        tier     = "LOW REVENUE"
        color    = "#ef4444"
        bg_color = "rgba(239,68,68,0.15)"
        border   = "rgba(239,68,68,0.4)"
        emoji    = "&#x1F534;"

    html = f"""
    <div style="background:{bg_color};border:1.5px solid {border};border-radius:20px;
                padding:2.5rem 2rem;text-align:center;font-family:Inter,sans-serif;">
        <div style="display:inline-block;background:{bg_color};border:1px solid {border};
                    border-radius:50px;padding:0.3rem 1.2rem;font-size:0.75rem;font-weight:700;
                    letter-spacing:0.1em;color:{color};margin-bottom:1rem;text-transform:uppercase;">
            {emoji} &nbsp; {tier}
        </div>
        <div style="font-size:3.5rem;font-weight:800;color:#ffffff;letter-spacing:-1px;
                    line-height:1;margin:0.4rem 0;">
            Rs. {prediction:,.0f}
        </div>
        <div style="color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:0.6rem;">
            Estimated Product Store Sales Total
        </div>
    </div>
    """
    return html

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
body, .gradio-container {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1040 50%, #0d1f3c 100%) !important;
    min-height: 100vh;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
label { color: #cbd5e1 !important; font-size: 0.88rem !important; }
button.primary {
    background: linear-gradient(135deg, #6c63ff, #3ecfff) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important; color: white !important;
}
button.primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(108,99,255,0.5) !important; }
"""

product_types = [
    "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
    "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
    "Soft Drinks", "Meat", "Breads", "Hard Drinks",
    "Others", "Starchy Foods", "Breakfast", "Seafood"
]

with gr.Blocks(css=css, title="SuperKart Sales Forecaster") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:2.5rem 1rem 1rem;font-family:Inter,sans-serif;">
        <div style="font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#6c63ff,#3ecfff,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
                    margin-bottom:0.4rem;">
            &#x1F6D2; SuperKart Sales Forecaster
        </div>
        <div style="color:rgba(255,255,255,0.45);font-size:1rem;">
            AI-powered product sales prediction &mdash; enter store and product details to forecast revenue
        </div>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,transparent,#6c63ff,#3ecfff,transparent);
                border-radius:2px;margin:1rem 0;"></div>
    <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.5rem;">
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.5rem;text-align:center;min-width:130px;">
            <div style="font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#6c63ff,#3ecfff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">8,764</div>
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;">Records Trained</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.5rem;text-align:center;min-width:130px;">
            <div style="font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#6c63ff,#3ecfff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">XGBoost</div>
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;">Model</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.5rem;text-align:center;min-width:130px;">
            <div style="font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#6c63ff,#3ecfff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">9</div>
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;">Input Features</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.5rem;text-align:center;min-width:130px;">
            <div style="font-size:1.5rem;font-weight:700;background:linear-gradient(135deg,#6c63ff,#3ecfff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Rs.</div>
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;">Sales Target</div>
        </div>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,transparent,#6c63ff,#3ecfff,transparent);border-radius:2px;margin:0 0 1.5rem;"></div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.3);padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:0.5rem;'>Product Attributes</div>")
            product_weight         = gr.Slider(1.0, 25.0, value=12.0, step=0.1,  label="Product Weight (kg)")
            product_allocated_area = gr.Slider(0.00, 0.35, value=0.05, step=0.01, label="Allocated Display Area Ratio")
            product_mrp            = gr.Slider(50, 300, value=150, step=1,        label="Maximum Retail Price (Rs.)")
            product_sugar_content  = gr.Dropdown(["Low Sugar", "Regular", "No Sugar"], value="Low Sugar", label="Sugar Content")
            product_type           = gr.Dropdown(product_types, value="Snack Foods", label="Product Category")
            gr.HTML("<div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.3);padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.07);margin:1rem 0 0.5rem;'>Store Attributes</div>")
            store_age              = gr.Slider(1, 40, value=15, step=1,           label="Years Since Store Opened")
            store_size             = gr.Dropdown(["High", "Medium", "Small"], value="Medium", label="Store Size")
            store_location         = gr.Dropdown(["Tier 1", "Tier 2", "Tier 3"], value="Tier 2", label="City Tier")
            store_type             = gr.Dropdown(
                ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
                value="Supermarket Type1", label="Store Type"
            )
            predict_btn = gr.Button("Forecast Sales Revenue", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.HTML("<div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.3);padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:0.5rem;'>Prediction Result</div>")
            output_html = gr.HTML("""
            <div style="background:rgba(255,255,255,0.03);border:1.5px dashed rgba(255,255,255,0.1);
                        border-radius:20px;padding:3rem 2rem;text-align:center;color:rgba(255,255,255,0.35);
                        font-family:Inter,sans-serif;">
                <div style="font-size:3rem;margin-bottom:0.8rem;">&#x1F4CA;</div>
                <div style="font-size:1rem;font-weight:600;">Awaiting Prediction</div>
                <div style="font-size:0.82rem;margin-top:0.4rem;">Fill in the details on the left and click Forecast Sales Revenue</div>
            </div>
            """)

    predict_btn.click(
        fn=predict_sales,
        inputs=[product_weight, product_allocated_area, product_mrp, store_age,
                product_sugar_content, product_type, store_size, store_location, store_type],
        outputs=output_html
    )

    gr.HTML("""
    <div style="height:2px;background:linear-gradient(90deg,transparent,#6c63ff,#3ecfff,transparent);border-radius:2px;margin:1.5rem 0 0;"></div>
    <div style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;padding:1rem 0 0.5rem;">
        SuperKart Sales Forecaster &nbsp;&middot;&nbsp; Powered by XGBoost &nbsp;&middot;&nbsp; Deployed on Hugging Face Spaces
    </div>
    """)

demo.launch()
