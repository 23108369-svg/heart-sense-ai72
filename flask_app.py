# =========================
# 1. IMPORT LIBRARIES
# =========================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# =========================
# 2. INIT APP
# =========================
app = Flask(__name__)

# =========================
# 3. LOAD MODEL + SCALER
# =========================
try:
    model = tf.keras.models.load_model("ann_model.h5")
    scaler = joblib.load("scaler.pkl")

    print("✅ Model and scaler loaded successfully!")

except Exception as e:
    print("❌ Error loading model/scaler:", e)

# =========================
# 4. HOME ROUTE




# =========================
@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Heart Disease Prediction API Running"
    })

# =========================
# 5. PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # get json data
        json_data = request.get_json()

        # validate input
        if "data" not in json_data:
            return jsonify({
                "error": "Missing 'data' key"
            }), 400

        data = json_data["data"]

        # convert to numpy array
        input_data = np.array(data).reshape(1, -1)

        # scale input
        input_data = scaler.transform(input_data)

        # prediction
        prediction = model.predict(input_data)[0][0]

        probability = float(prediction)

        result = (
            "HIGH RISK"
            if probability > 0.5
            else "LOW RISK"
        )

        return jsonify({
            "prediction_score": round(probability, 4),
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# =========================
# 6. RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )