import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ---------- Helper to load model safely ----------
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Cannot load model {path}: {e}")
        return None

# ---------- Load Model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "rf_model_compressed.pkl"))


# ---------- Page Configuration ----------
st.set_page_config(page_title="Weather Prediction", page_icon="🌦", layout="centered")

# ---------- Title ----------
st.markdown("<h1 style='text-align:center; color:#7728b0;'>🌦 Weather Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("Enter weather parameters on the left to predict the temperature.")

# ---------- Sidebar Inputs ----------
st.sidebar.header("Input Weather Parameters")

avg_humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
avg_cloudcover = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 40.0)
avg_windspeedKmph = st.sidebar.slider("Wind Speed (Kmph)", 0.0, 200.0, 10.0)
total_precipMM = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, max_value=1000.0, value=5.0)

# ---------- Input Summary ----------
st.subheader("Input Summary")
st.write(pd.DataFrame({
    "Feature": ["avg_humidity", "avg_cloudcover", "avg_windspeedKmph", "total_precipMM"],
    "Value": [avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]
}))

# ---------- Prediction Section ----------
if st.button("Predict 🌤"):
    if model is None:
        st.error("❌ Model not loaded. Please ensure the file is in the same folder.")
    else:
        # Prepare input
        X_input = pd.DataFrame([[avg_humidity, avg_cloudcover, avg_windspeedKmph, total_precipMM]],
                               columns=['avg_humidity', 'avg_cloudcover', 'avg_windspeedKmph', 'total_precipMM'])
        
        # Display processing message
        with st.spinner('Running prediction...'):
            pred = model.predict(X_input)[0]

        # Display results
        st.success(f"🌡 Predicted Temperature: *{pred:.2f} °C*")

        # Feedback message
        if pred > 35:
            st.warning("☀ Hot day predicted — stay hydrated!")
        elif pred < 20:
            st.info("🧥 Cool day predicted — you may need a jacket.")
        else:
            st.success("🌤 Moderate temperature predicted.")

# ---------- Footer ----------
st.markdown("---")
st.caption("This weather prediction tool uses data-driven algorithms trained in Google Colab.")