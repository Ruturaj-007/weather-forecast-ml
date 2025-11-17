import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# ---------- Load Model ----------
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return f"Cannot load model {path}: {e}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "rf_model_compressed.pkl"))


# ---------- Prediction Function ----------
def predict_weather(avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM):

    if isinstance(model, str):
        return "", pd.DataFrame(), model, ""

    # Prepare table
    df = pd.DataFrame({
        "Feature": ["Humidity (%)", "Cloud Cover (%)", "Wind Speed (Kmph)", "Precipitation (mm)"],
        "Value": [avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]
    })

    # Prediction
    X = pd.DataFrame([[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]],
                     columns=['avg_humidity', 'avg_cloudcover', 'avg_windspeedKmph', 'total_precipMM'])

    pred = model.predict(X)[0]

    # ------------------ Dynamic Color Box with Animation ------------------
    if pred > 35:
        gradient = "linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%)"
        shadow = "0 8px 32px rgba(255, 107, 107, 0.3)"
        text_msg = "Hot day predicted — Stay hydrated!"
        icon_bg = "#ff6b6b"
    elif pred < 20:
        gradient = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
        shadow = "0 8px 32px rgba(79, 172, 254, 0.3)"
        text_msg = "Cool day predicted — You may need a jacket."
        icon_bg = "#4facfe"
    else:
        gradient = "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
        shadow = "0 8px 32px rgba(67, 233, 123, 0.3)"
        text_msg = "Moderate temperature predicted."
        icon_bg = "#43e97b"

    msg = f"""
    <div class='prediction-card animate-in' style="
        background: {gradient};
        padding: 30px;
        border-radius: 20px;
        box-shadow: {shadow};
        color: white;
        position: relative;
        overflow: hidden;">
        
        <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; 
                    background: rgba(255,255,255,0.1); border-radius: 50%; filter: blur(40px);"></div>
        
        <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 1;">
            <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(10px);
                        width: 80px; height: 80px; border-radius: 20px; display: flex;
                        align-items: center; justify-content: center; font-size: 40px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); font-weight: 900;">
                {pred:.0f}°
            </div>
            <div style="flex: 1;">
                <div style="font-size: 16px; opacity: 0.9; font-weight: 500; margin-bottom: 5px;">
                    Predicted Temperature
                </div>
                <div style="font-size: 48px; font-weight: 900; letter-spacing: -2px;">
                    {pred:.1f}°C
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.15);
                    backdrop-filter: blur(10px); border-radius: 12px; font-size: 16px;
                    font-weight: 500; position: relative; z-index: 1;">
            {text_msg}
        </div>
    </div>
    """

    return "", df, msg, pred


# ---------- Enhanced CSS ----------
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* Glass card effect */
#main-card {
    max-width: 1000px;
    margin: 40px auto;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    padding: 45px;
    border-radius: 30px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Header with gradient text */
h1 {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}

/* Subtitle */
.gradio-container .prose {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section headers */
h3 {
    color: #1e293b;
    font-weight: 700;
    font-size: 20px;
    margin-top: 30px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Enhanced sliders */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(to right, #667eea, #764ba2);
    outline: none;
    transition: all 0.3s ease;
}

input[type="range"]:hover {
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: white;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Labels */
label {
    font-weight: 600 !important;
    color: #475569 !important;
    font-size: 14px !important;
}

/* Number input */
input[type="number"] {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

input[type="number"]:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Button styling */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    padding: 16px 32px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin: 25px 0 !important;
}

.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5) !important;
}

.primary:active {
    transform: translateY(0);
}

/* Table styling */
.dataframe {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 2px solid #e2e8f0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
}

.dataframe thead {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.dataframe thead th {
    color: white !important;
    font-weight: 700 !important;
    padding: 15px !important;
    font-size: 14px !important;
}

.dataframe tbody tr {
    transition: all 0.2s ease !important;
}

.dataframe tbody tr:hover {
    background: #f8fafc !important;
}

.dataframe tbody td {
    padding: 12px 15px !important;
    color: #475569 !important;
}

/* Animation for prediction card */
@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: slideUp 0.6s ease-out;
}

/* Footer */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #e2e8f0, transparent);
    margin: 40px 0 20px 0;
}

/* Responsive design */
@media (max-width: 768px) {
    #main-card {
        margin: 20px;
        padding: 25px;
        border-radius: 20px;
    }
    
    h1 {
        font-size: 36px;
    }
    
    .prediction-card {
        padding: 20px !important;
    }
    
    .prediction-card > div:first-child {
        flex-direction: column !important;
        text-align: center;
    }
}

/* Input containers */
.input-container {
    background: #f8fafc;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.input-container:hover {
    border-color: #667eea;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
}
"""


# ---------- UI Layout ----------
with gr.Blocks(css=css, title="SmartWeather - ML Temperature Forecasting") as demo:

    with gr.Column(elem_id="main-card"):

        gr.Markdown("""
        <div style='text-align:center; margin-bottom:35px;'>
            <div style='font-size:56px; margin-bottom:15px;'>⛅</div>
            <h1 style='margin-bottom:12px;'>SmartWeather</h1>
            <div style='font-size:15px; color:#1e293b; font-weight:500; line-height:1.5;'>
                Accurate temperature forecasting powered by machine learning
            </div>
        </div>
        """)

        gr.Markdown("### Input Weather Parameters")

        with gr.Row():
            avg_humidity = gr.Slider(0, 100, value=60, label="Humidity (%)", 
                                    info="Atmospheric moisture level")
            avg_cloudcover = gr.Slider(0, 100, value=40, label="Cloud Cover (%)", 
                                      info="Sky coverage percentage")

        with gr.Row():
            avg_windspeedKmph = gr.Slider(0, 200, value=10, label="Wind Speed (Kmph)", 
                                         info="Wind velocity in kilometers per hour")
            total_precipMM = gr.Number(value=5.0, label="Precipitation (mm)", 
                                      info="Rainfall in millimeters")

        predict_btn = gr.Button("Generate Forecast", variant="primary")

        gr.Markdown("### Parameter Summary")
        input_table = gr.Dataframe(headers=["Feature", "Value"], 
                                  datatype=["str", "number"], 
                                  interactive=False)

        gr.Markdown("### Prediction Result")
        output = gr.HTML()

        hidden = gr.Number(visible=False)

        predict_btn.click(
            predict_weather,
            inputs=[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM],
            outputs=[gr.Textbox(visible=False), input_table, output, hidden]
        )

        gr.Markdown("<hr>")
        gr.Markdown("<div style='text-align:center; color:#94a3b8; font-size:14px;'>Powered by Random Forest ML Model & Gradio</div>")

demo.launch()