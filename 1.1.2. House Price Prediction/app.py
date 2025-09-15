from flask import Flask, request, render_template
import joblib
import numpy as np
import json
import pandas as pd       # <-- add this

app = Flask(__name__)

# Load model and feature order
model = joblib.load("house_price_model.joblib")
with open("features.json") as f:
    feature_names = json.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        row = {}
        for feat in feature_names:
            raw = request.form.get(feat, "")
            try:
                val = float(raw)
            except ValueError:
                val = raw
            row[feat] = val

        # create a DataFrame with a single row
        X = pd.DataFrame([row], columns=feature_names)

        log_pred = model.predict(X)[0]
        prediction = np.expm1(log_pred)

    return render_template("index.html",
                           prediction=prediction,
                           features=feature_names)

if __name__ == "__main__":
    app.run(debug=True)