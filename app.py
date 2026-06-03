# =========================
# 1. IMPORT LIBRARIES
# =========================
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# =========================
# 2. LOAD MODEL
# =========================
model = tf.keras.models.load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")

# =========================
# 3. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="centered"
)

# =========================
# 4. DARK THEME UI
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    width: 100%;
    height: 45px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 5. TITLE
# =========================
st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below:")

# =========================
# 6. INPUT FIELDS
# =========================
age = st.number_input("Age")
sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("ECG Result", [0,1,2])
thalachh = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [1,2,3])

# =========================
# 7. PREDICTION
# =========================
if st.button("Predict ❤️"):

    data = np.array([[age, sex, cp, trestbps, chol,
                      fbs, restecg, thalachh, exang,
                      oldpeak, slope, ca, thal]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.subheader("Result:")

    if prediction[0][0] > 0.5:
        st.error("⚠ HIGH RISK OF HEART DISEASE")
        st.write("Consult a doctor immediately.")
    else:
        st.success("✅ LOW RISK OF HEART DISEASE")
        st.write("Maintain healthy lifestyle ❤️")