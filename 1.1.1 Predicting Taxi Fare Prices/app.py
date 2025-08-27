from flask import Flask, request, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("model", "taxi_fare_model_small.pkl")
model = joblib.load(MODEL_PATH)

# Features expected by the model
FEATURES = ["distance_miles", "passenger_count", "hour_of_day", "day_of_week", "month"]

def prepare_input(form_data):
    """
    Converts HTML form input into a DataFrame with correct column names.
    """
    values = {f: [float(form_data[f])] for f in FEATURES}
    return pd.DataFrame(values)

@app.route("/", methods=["GET", "POST"])
def index():
    fare_prediction = None
    if request.method == "POST":
        input_df = prepare_input(request.form)
        fare_prediction = round(model.predict(input_df)[0], 2)
    return render_template("index.html", prediction=fare_prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
