import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌤️ Next-Hour Temperature Predictor")
st.markdown("Enter current weather data below to get the predicted temperature for the next hour.")

# Input features
temperature = st.number_input("Temperature (°C)", value=22.0)
apparent_temp = st.number_input("Apparent Temperature (°C)", value=22.5)
humidity = st.number_input("Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.75)
wind_speed = st.number_input("Wind Speed (km/h)", value=13.0)
pressure = st.number_input("Pressure (millibars)", value=1012)

rolling_3_temp = st.number_input("3-Hour Rolling Temp (°C)", value=21.9)
rolling_14_temp = st.number_input("14-Hour Rolling Temp (°C)", value=21.5)
rolling_3_hum = st.number_input("3-Hour Rolling Humidity", min_value=0.0, max_value=1.0, value=0.73)
rolling_14_hum = st.number_input("14-Hour Rolling Humidity", min_value=0.0, max_value=1.0, value=0.70)

# Prediction logic
if st.button("Predict Temperature"):
    input_features = np.array([
        temperature, apparent_temp, humidity, wind_speed, pressure,
        rolling_3_temp, rolling_14_temp, rolling_3_hum, rolling_14_hum
    ]).reshape(1, -1)

    # Use the pre-trained scaler
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)

    st.success(f"🌡️ Predicted Temperature for Next Hour: {round(prediction[0], 2)} °C")
