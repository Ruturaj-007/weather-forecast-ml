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
        return model, pd.DataFrame(), ""

    # Prepare table data
    df = pd.DataFrame({
        "Feature": ["avg_humidity", "avg_cloudcover", "avg_windspeedKmph", "total_precipMM"],
        "Value": [avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]
    })

    # Predict
    X = pd.DataFrame([[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]],
                     columns=['avg_humidity', 'avg_cloudcover', 'avg_windspeedKmph', 'total_precipMM'])

    pred = model.predict(X)[0]

    # Pretty Output Box
    msg = f"""
<div class='success-box'>
    🌡 <b>Predicted Temperature: {pred:.2f} °C</b><br>
"""

    if pred > 35:
        msg += "☀ Hot day predicted — Stay hydrated!"
    elif pred < 20:
        msg += "🧥 Cool day predicted — You may need a jacket."
    else:
        msg += "🌤 Moderate temperature predicted."

    msg += "</div>"

    return "", df, msg


# ---------- CSS (white, clean, responsive) ----------
css = """
.gradio-container {
    background: #f5f7fb !important;
    font-family: 'Segoe UI', sans-serif;
}

#main-card {
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    font-size: 34px;
    color: #6a1bb0;
    font-weight: 800;
}

.success-box {
    background: #d9f7e8;
    padding: 18px 20px;
    border-radius: 12px;
    border-left: 6px solid #0f9d58;
    font-size: 17px;
}

.table-container {
    margin-top: 15px;
}

@media (max-width: 600px){
    #main-card{
        padding: 20px;
        border-radius: 15px;
    }
    h1{
        font-size: 28px;
    }
}
"""

# ---------- UI Layout ----------
with gr.Blocks(css=css, title="Weather Predictor") as demo:

    with gr.Column(elem_id="main-card"):

        gr.Markdown("<h1>🌦 Weather Prediction Dashboard</h1>")
        gr.Markdown("<div style='text-align:center;'>Enter weather parameters to predict the temperature.</div>")

        gr.Markdown("### **Input Weather Parameters**")

        with gr.Row():
            avg_humidity = gr.Slider(0, 100, value=60, label="Humidity (%)")
            avg_cloudcover = gr.Slider(0, 100, value=40, label="Cloud Cover (%)")

        with gr.Row():
            avg_windspeedKmph = gr.Slider(0, 200, value=10, label="Wind Speed (Kmph)")
            total_precipMM = gr.Number(value=5.0, label="Precipitation (mm)")

        predict_btn = gr.Button("🌤 Predict Temperature", variant="primary")

        gr.Markdown("### **Input Summary**")
        input_table = gr.Dataframe(headers=["Feature", "Value"], datatype=["str", "number"], interactive=False)

        output = gr.HTML()

        predict_btn.click(
            predict_weather,
            inputs=[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM],
            outputs=[gr.Textbox(visible=False), input_table, output]
        )

        gr.Markdown("<hr>")
        gr.Markdown("<div style='text-align:center; opacity:0.7;'>Built using Machine Learning & Gradio.</div>")

demo.launch()
