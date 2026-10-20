import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.preprocess import RAW_FEATURES, REFERENCE_YEAR, normalize_name
from src.train import METRICS_PATH, MODEL_PATH, train_and_save

app = Flask(__name__)


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(METRICS_PATH.read_text())
    except Exception as exc:                      # noqa: BLE001
        print(f"[app] could not load model ({exc}); retraining...")
        meta = train_and_save(verbose=False)
        model = joblib.load(MODEL_PATH)
    return model, meta


MODEL, META = load_artifacts()
KNOWN_NAMES = set(META["car_names"])
CURRENT_YEAR = date.today().year

CHOICES = {
    "Fuel_Type": {"Petrol", "Diesel", "Cng"},
    "Selling_type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}


def parse_input(data: dict) -> dict:
    """Validate raw form/JSON values. Raises ValueError with a friendly message."""
    try:
        name = normalize_name(data.get("Car_Name", ""))
        if not name:
            raise ValueError("Please enter the car / bike name.")
        year = int(float(data.get("Year", "")))
        present = float(data.get("Present_Price", ""))
        kms = float(data.get("Driven_kms", ""))
        owner = int(float(data.get("Owner", 0)))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Please"):
            raise
        raise ValueError("Please fill every field with valid numbers.") from None

    fuel = str(data.get("Fuel_Type", "")).strip().title()
    seller = str(data.get("Selling_type", "")).strip().title()
    trans = str(data.get("Transmission", "")).strip().title()
    for key, val in (("Fuel_Type", fuel), ("Selling_type", seller), ("Transmission", trans)):
        if val not in CHOICES[key]:
            raise ValueError(f"Invalid value for {key.replace('_', ' ')}.")

    if not 1995 <= year <= CURRENT_YEAR:
        raise ValueError(f"Year must be between 1995 and {CURRENT_YEAR}.")
    if not 0.1 <= present <= 500:
        raise ValueError("Showroom price must be between 0.1 and 500 lakh.")
    if not 0 <= kms <= 1_000_000:
        raise ValueError("Kilometres driven must be between 0 and 1,000,000.")
    if owner < 0 or owner > 5:
        raise ValueError("Previous owners must be between 0 and 5.")

    return {"Car_Name": name, "Year": year, "Present_Price": present,
            "Driven_kms": kms, "Fuel_Type": fuel, "Selling_type": seller,
            "Transmission": trans, "Owner": owner}


def predict_price(row: dict) -> dict:
    X = pd.DataFrame([row])[RAW_FEATURES]
    price = float(MODEL.predict(X)[0])
    price = float(np.clip(price, 0.05, row["Present_Price"]))   # can't beat new price

    lo = price * float(np.exp(META["oof_residual_log_q10"]))
    hi = price * float(np.exp(META["oof_residual_log_q90"]))
    lo, hi = max(0.05, min(lo, price)), min(row["Present_Price"], max(hi, price))

    notes = []
    if row["Car_Name"] not in KNOWN_NAMES:
        notes.append("This model name wasn't in the training data, so the estimate "
                     "relies mostly on price, age, mileage and fuel/transmission.")
    if row["Year"] > REFERENCE_YEAR:
        notes.append(f"The model was trained on listings up to {REFERENCE_YEAR}; "
                     "newer vehicles are treated as the newest age bracket.")
    return {
        "price_lakh": round(price, 2),
        "low_lakh": round(lo, 2),
        "high_lakh": round(hi, 2),
        "price_inr": int(round(price * 100_000, -2)),
        "low_inr": int(round(lo * 100_000, -3)),
        "high_inr": int(round(hi * 100_000, -3)),
        "notes": notes,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        car_names=sorted(KNOWN_NAMES),
        current_year=CURRENT_YEAR,
        metrics=META["test"],
        model_name=META["best_model"],
        n_rows=META["n_rows"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify({"ok": True, **predict_price(parse_input(data))})
    except ValueError as err:
        return jsonify({"ok": False, "error": str(err)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": META["best_model"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)