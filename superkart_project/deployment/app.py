import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import spaces

# Load model
model_path = hf_hub_download(
 repo_id="Sadhana3105/superkart-sales-model",
 filename="best_superkart_model_v1.joblib"
)
model = joblib.load(model_path)

@spaces.GPU
def _dummy():
 pass

def predict_sales(
 product_weight, product_allocated_area, product_mrp, store_age,
 product_sugar_content, product_type, store_size,
 store_location_city_type, store_type
):
 input_df = pd.DataFrame([{
 "Product_Weight": product_weight,
 "Product_Allocated_Area": product_allocated_area,
 "Product_MRP": product_mrp,
 "Store_Age": store_age,
 "Product_Sugar_Content": product_sugar_content,
 "Product_Type": product_type,
 "Store_Size": store_size,
 "Store_Location_City_Type": store_location_city_type,
 "Store_Type": store_type,
 }])

 prediction = model.predict(input_df)[0]

 if prediction >= 4000:
 tier = "HIGH REVENUE"
 color = "#10b981" # Emerald
 elif prediction >= 2500:
 tier = "MODERATE REVENUE"
 color = "#f59e0b" # Amber
 else:
 tier = "LOW REVENUE"
 color = "#ef4444" # Red

 html = f"""
 <div style="border:1px solid var(--border-color-primary);border-radius:12px;padding:2.5rem 2rem;text-align:center;font-family:Inter,sans-serif;background:var(--background-fill-secondary);">
 <div style="display:inline-block;border:1px solid {color};border-radius:4px;padding:0.3rem 0.8rem;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;color:{color};margin-bottom:1rem;text-transform:uppercase;background:transparent;">
 {tier}
 </div>
 <div style="font-size:3.2rem;font-weight:700;color:var(--body-text-color);letter-spacing:-1px;line-height:1;margin:0.4rem 0;">
 Rs. {prediction:,.0f}
 </div>
 <div style="color:var(--body-text-color-subdued);font-size:0.9rem;margin-top:0.8rem;">
 Estimated Product Store Sales Total
 </div>
 </div>
 """
 return html

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
button.primary {
 background-color: #0f172a !important; /* Dark slate */
 border: none !important; 
 border-radius: 6px !important;
 font-weight: 600 !important; 
 font-size: 1rem !important;
 color: white !important;
 transition: all 0.2s ease !important;
}
button.primary:hover { 
 background-color: #334155 !important; 
}
"""

product_types = [
 "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
 "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
 "Soft Drinks", "Meat", "Breads", "Hard Drinks",
 "Others", "Starchy Foods", "Breakfast", "Seafood"
]

with gr.Blocks(title="SuperKart Sales Forecaster") as demo:

 gr.HTML("""
 <div style="text-align:center;padding:2rem 1rem 1rem;font-family:Inter,sans-serif;">
 <div style="font-size:2.2rem;font-weight:700;color:var(--body-text-color);margin-bottom:0.3rem;letter-spacing:-0.5px;">
 &#x1F6D2; SuperKart Sales Forecaster
 </div>
 <div style="color:var(--body-text-color-subdued);font-size:1rem;">
 Enter store and product details to forecast sales revenue
 </div>
 </div>
 <hr style="border:0;height:1px;background:var(--border-color-primary);margin:1rem 0;">
 <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.5rem;">
 <div style="border:1px solid var(--border-color-primary);border-radius:8px;padding:1rem 1.5rem;text-align:center;min-width:130px;background:var(--background-fill-secondary);">
 <div style="font-size:1.4rem;font-weight:600;color:var(--body-text-color);">8,764</div>
 <div style="font-size:0.75rem;color:var(--body-text-color-subdued);text-transform:uppercase;letter-spacing:0.05em;margin-top:0.3rem;">Records Trained</div>
 </div>
 <div style="border:1px solid var(--border-color-primary);border-radius:8px;padding:1rem 1.5rem;text-align:center;min-width:130px;background:var(--background-fill-secondary);">
 <div style="font-size:1.4rem;font-weight:600;color:var(--body-text-color);">XGBoost</div>
 <div style="font-size:0.75rem;color:var(--body-text-color-subdued);text-transform:uppercase;letter-spacing:0.05em;margin-top:0.3rem;">Model</div>
 </div>
 <div style="border:1px solid var(--border-color-primary);border-radius:8px;padding:1rem 1.5rem;text-align:center;min-width:130px;background:var(--background-fill-secondary);">
 <div style="font-size:1.4rem;font-weight:600;color:var(--body-text-color);">9</div>
 <div style="font-size:0.75rem;color:var(--body-text-color-subdued);text-transform:uppercase;letter-spacing:0.05em;margin-top:0.3rem;">Input Features</div>
 </div>
 <div style="border:1px solid var(--border-color-primary);border-radius:8px;padding:1rem 1.5rem;text-align:center;min-width:130px;background:var(--background-fill-secondary);">
 <div style="font-size:1.4rem;font-weight:600;color:var(--body-text-color);">INR</div>
 <div style="font-size:0.75rem;color:var(--body-text-color-subdued);text-transform:uppercase;letter-spacing:0.05em;margin-top:0.3rem;">Sales Target</div>
 </div>
 </div>
 <hr style="border:0;height:1px;background:var(--border-color-primary);margin:0 0 1.5rem;">
 """)

 with gr.Row():
 with gr.Column(scale=1):
 gr.HTML("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--body-text-color-subdued);padding-bottom:0.3rem;border-bottom:1px solid var(--border-color-primary);margin-bottom:0.8rem;'>Product Attributes</div>")
 product_weight = gr.Slider(1.0, 25.0, value=12.0, step=0.1, label="Product Weight (kg)")
 product_allocated_area = gr.Slider(0.00, 0.35, value=0.05, step=0.01, label="Allocated Display Area Ratio")
 product_mrp = gr.Slider(50, 300, value=150, step=1, label="Maximum Retail Price (Rs.)")
 product_sugar_content = gr.Dropdown(["Low Sugar", "Regular", "No Sugar"], value="Low Sugar", label="Sugar Content")
 product_type = gr.Dropdown(product_types, value="Snack Foods", label="Product Category")
 
 gr.HTML("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--body-text-color-subdued);padding-bottom:0.3rem;border-bottom:1px solid var(--border-color-primary);margin:1.2rem 0 0.8rem;'>Store Attributes</div>")
 store_age = gr.Slider(1, 40, value=15, step=1, label="Years Since Store Opened")
 store_size = gr.Dropdown(["High", "Medium", "Small"], value="Medium", label="Store Size")
 store_location = gr.Dropdown(["Tier 1", "Tier 2", "Tier 3"], value="Tier 2", label="City Tier")
 store_type = gr.Dropdown(
 ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
 value="Supermarket Type1", label="Store Type"
 )
 predict_btn = gr.Button("Forecast Sales Revenue", variant="primary", size="lg")

 with gr.Column(scale=1):
 gr.HTML("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--body-text-color-subdued);padding-bottom:0.3rem;border-bottom:1px solid var(--border-color-primary);margin-bottom:0.8rem;'>Prediction Result</div>")
 output_html = gr.HTML("""
 <div style="border:1px dashed var(--border-color-primary);border-radius:12px;padding:3rem 2rem;text-align:center;color:var(--body-text-color-subdued);font-family:Inter,sans-serif;background:var(--background-fill-secondary);">
 <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.5;">&#x1F4CA;</div>
 <div style="font-size:1rem;font-weight:500;">Ready for Prediction</div>
 <div style="font-size:0.85rem;margin-top:0.4rem;">Fill in the details on the left and click Forecast Sales Revenue</div>
 </div>
 """)

 predict_btn.click(
 fn=predict_sales,
 inputs=[product_weight, product_allocated_area, product_mrp, store_age,
 product_sugar_content, product_type, store_size, store_location, store_type],
 outputs=output_html
 )

 gr.HTML("""
 <hr style="border:0;height:1px;background:var(--border-color-primary);margin:2rem 0 1rem;">
 <div style="text-align:center;color:var(--body-text-color-subdued);font-size:0.75rem;padding-bottom:1rem;">
 SuperKart Sales Forecaster &nbsp;&middot;&nbsp; Powered by XGBoost &nbsp;&middot;&nbsp; Deployed on Hugging Face Spaces
 </div>
 """)

demo.launch(css=css, ssr_mode=False)
