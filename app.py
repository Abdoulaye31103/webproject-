from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path
import os

app = Flask(__name__)

# ---------- Load model -------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "rf_pipeline.joblib"
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    pipe = None
    print(f"❌ Failed to load model: {e}")

# Optional: Load reference dataset (not used for prediction)
DATASET_PATH = Path(__file__).parent / "model" / "processed_titanic.csv"
if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
else:
    df = None

# ---------- Routes -----------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not pipe:
        return "Model not loaded. Please check server logs.", 500

    try:
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Age": float(request.form["Age"]),
            "SibSp": int(request.form["SibSp"]),
            "Parch": int(request.form["Parch"]),
            "Fare": float(request.form["Fare"]),
        }
        X = pd.DataFrame([data])
        pred = pipe.predict(X)[0]
        prob = pipe.predict_proba(X)[0][1]

        return render_template(
            "result.html",
            survived=bool(pred),
            probability=f"{prob*100:.1f}%",
            data=data
        )
    except Exception as e:
        return f"Error making prediction: {e}", 400

# ---------- Entry point ------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT env var
    app.run(debug=False, host="0.0.0.0", port=port)
