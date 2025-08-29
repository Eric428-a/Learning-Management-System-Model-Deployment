# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load model + feature schema
MODEL_PATH = "taxi_fare_model.joblib"
FEATURES_PATH = "features.json"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

with open(FEATURES_PATH) as f:
    feature_names = json.load(f)["features"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    # Convert types
    for k in input_data:
        input_data[k] = float(input_data[k])
    df = pd.DataFrame([input_data], columns=feature_names)
    fare = round(float(model.predict(df)[0]), 2)
    return render_template('index.html', prediction=fare)

if __name__ == '__main__':
    app.run(debug=True)
