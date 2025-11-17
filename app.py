import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# ---------- Helper to load model safely ----------
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return f"Cannot load model {path}: {e}"

# ---------- Load Model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "rf_model_compressed.pkl"))


# ---------- Prediction Function ----------
def predict_weather(avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM):
    if isinstance(model, str):
        return model  # error string
    
    X_input = pd.DataFrame([[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]],
                           columns=['avg_humidity', 'avg_cloudcover', 'avg_windspeedKmph', 'total_precipMM'])
    
    pred = model.predict(X_input)[0]

    message = f"🌡 Predicted Temperature: {pred:.2f} °C\n\n"

    if pred > 35:
        message += "☀ Hot day predicted — stay hydrated!"
    elif pred < 20:
        message += "🧥 Cool day predicted — you may need a jacket."
    else:
        message += "🌤 Moderate temperature predicted."

    return message


# ---------- UI ----------
with gr.Blocks(title="Weather Prediction", theme="soft") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#7728b0;'>🌦 Weather Prediction Dashboard</h1>")
    gr.Markdown("Enter weather parameters to predict the temperature.")

    with gr.Row():
        avg_humidity = gr.Slider(0, 100, value=60, label="Humidity (%)")
        avg_cloudcover = gr.Slider(0, 100, value=40, label="Cloud Cover (%)")
    
    with gr.Row():
        avg_windspeedKmph = gr.Slider(0, 200, value=10, label="Wind Speed (Kmph)")
        total_precipMM = gr.Number(value=5.0, label="Precipitation (mm)")

    predict_btn = gr.Button("Predict 🌤")
    output = gr.Textbox(label="Prediction Output")

    predict_btn.click(
        predict_weather,
        inputs=[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM],
        outputs=output
    )

    gr.Markdown("---")
    gr.Markdown("This weather prediction tool uses a machine learning model trained in Google Colab.")

demo.launch()
