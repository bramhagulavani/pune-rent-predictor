import gradio as gr
import pickle
import numpy as np
import pandas as pd

# ── Load model and encoders ──────────────────────────────────────────────────
print("Loading model...")

with open("C:/Projects/pune-rent-predictor/models/pune_rent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:/Projects/pune-rent-predictor/models/location_encoder.pkl", "rb") as f:
    le_location = pickle.load(f)

with open("C:/Projects/pune-rent-predictor/models/furnishing_encoder.pkl", "rb") as f:
    le_furnishing = pickle.load(f)

with open("C:/Projects/pune-rent-predictor/models/locations_list.pkl", "rb") as f:
    locations_list = pickle.load(f)

print("Model loaded!")

# ── Prediction function ──────────────────────────────────────────────────────
def predict_rent(location, bedroom, bathroom, area, 
                 furnishing, floor_number, parking, 
                 powerbackup, gate_community):

    # Encode categorical inputs
    location_encoded = le_location.transform([location])[0]
    furnishing_encoded = le_furnishing.transform([furnishing])[0]

    # Build input array
    features = np.array([[
        bedroom, bathroom, area,
        furnishing_encoded, floor_number,
        int(parking), int(powerbackup), int(gate_community),
        location_encoded
    ]])

    # Predict
    predicted_rent = model.predict(features)[0]

    # Add confidence range ±15%
    low = predicted_rent * 0.85
    high = predicted_rent * 1.15

    return (
        f"₹{int(predicted_rent):,} / month",
        f"₹{int(low):,} — ₹{int(high):,} / month"
    )

# ── Custom CSS ───────────────────────────────────────────────────────────────
custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
        min-height: 100vh;
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-align: center;
    }
    .block {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
    }
    label span {
        color: #a8d5a2 !important;
        font-weight: 600 !important;
    }
    button.primary {
        background: linear-gradient(90deg, #56ab2f, #a8e063) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #0f2027 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    footer { display: none !important; }
"""

# ── Banner ───────────────────────────────────────────────────────────────────
banner = """
<div style="text-align:center; padding:10px 0 18px;">
    <p style="color:#a8d5a2; font-size:1rem; margin:0;">
        🧠 Powered by <b>XGBoost</b> &nbsp;|&nbsp;
        🏠 Trained on <b>3,201</b> Pune listings &nbsp;|&nbsp;
        🎯 <b>74.79%</b> R² Accuracy
    </p>
    <p style="color:#7fb3a0; font-size:0.85rem; margin-top:6px;">
        Enter property details below and get instant rent prediction!
    </p>
</div>
"""

footer_html = """
<div style="text-align:center; margin-top:20px; color:#7fb3a0; font-size:0.82rem;">
    Built by <b style="color:#a8d5a2;">Bramha Vinayak Gulavani</b> &nbsp;·&nbsp;
    AI & ML Student, VIT Pune &nbsp;·&nbsp;
    <a href="https://github.com/bramhagulavani/pune-rent-predictor"
       style="color:#56ab2f;" target="_blank">GitHub →</a>
</div>
"""

# ── Build Gradio app ─────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="🏠 Pune Rent Predictor") as app:

    gr.HTML("<h1>🏠 Pune Rent Predictor AI</h1>")
    gr.HTML(banner)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            location = gr.Dropdown(
                choices=sorted(locations_list),
                label="📍 Location",
                value=locations_list[0]
            )
            with gr.Row():
                bedroom = gr.Slider(1, 6, value=2, step=1, label="🛏️ Bedrooms (BHK)")
                bathroom = gr.Slider(1, 6, value=2, step=1, label="🚿 Bathrooms")

            area = gr.Slider(200, 5000, value=900, step=50, label="📐 Area (sqft)")
            furnishing = gr.Radio(
                choices=["Unfurnished", "Semifurnished", "Furnished"],
                value="Semifurnished",
                label="🪑 Furnishing"
            )
            floor_number = gr.Slider(0, 30, value=3, step=1, label="🏢 Floor Number")

            with gr.Row():
                parking = gr.Checkbox(label="🚗 Parking")
                powerbackup = gr.Checkbox(label="⚡ Power Backup")
                gate_community = gr.Checkbox(label="🔐 Gated Community")

            predict_btn = gr.Button("🔍 Predict Rent", variant="primary")

        # Right column — output
        with gr.Column(scale=1):
            predicted = gr.Textbox(
                label="💰 Predicted Rent",
                interactive=False
            )
            rent_range = gr.Textbox(
                label="📊 Expected Range (±15%)",
                interactive=False
            )
            gr.HTML("""
                <div style="margin-top:14px; padding:14px 16px;
                            background:rgba(255,255,255,0.05);
                            border:1px solid rgba(255,255,255,0.1);
                            border-radius:12px; color:#a8d5a2;
                            font-size:0.84rem; line-height:1.8;">
                    <b style="color:#fff;">ℹ️ How to use:</b><br>
                    1. Select your preferred location<br>
                    2. Set bedrooms, bathrooms and area<br>
                    3. Choose furnishing type<br>
                    4. Select amenities<br>
                    5. Click <b>Predict Rent</b>!<br><br>
                    <b style="color:#fff;">📍 Coverage:</b><br>
                    343 Pune locations including Kharadi,
                    Hinjewadi, Hadapsar, Wakad, Baner,
                    Kalyani Nagar, Viman Nagar & more!
                </div>
            """)

    predict_btn.click(
        fn=predict_rent,
        inputs=[location, bedroom, bathroom, area,
                furnishing, floor_number, parking,
                powerbackup, gate_community],
        outputs=[predicted, rent_range]
    )

    gr.HTML(footer_html)

# ── Launch ───────────────────────────────────────────────────────────────────
print("Starting app...")
app.launch()