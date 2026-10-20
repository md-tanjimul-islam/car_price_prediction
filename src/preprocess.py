"""
preprocess.py
-------------
Data loading, cleaning, feature engineering and the sklearn preprocessing
step used by BOTH training (src/train.py) and the web app (app.py).

The feature engineering lives inside the saved pipeline, so the website only
needs to send the raw form values (Car_Name, Year, Present_Price, ...).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "car data.csv"

TARGET = "Selling_Price"          # in lakhs (1 lakh = 100,000 INR)
REFERENCE_YEAR = 2018             # newest year in the dataset (data snapshot year)

RAW_FEATURES = [
    "Car_Name", "Year", "Present_Price", "Driven_kms",
    "Fuel_Type", "Selling_type", "Transmission", "Owner",
]

# Two-wheelers contained in the dataset (used to create a Vehicle_Type feature)
BIKE_KEYWORDS = (
    "bajaj", "hero", "honda cb", "honda activa", "honda dream", "honda karizma",
    "activa", "hyosung", "ktm", "mahindra mojo", "royal enfield", "suzuki access",
    "tvs", "um renegade", "yamaha",
)

# Car model -> brand ("brand goodwill" feature from the task description)
CAR_BRANDS = {
    "maruti": ["800", "alto 800", "alto k10", "baleno", "ciaz", "dzire", "ertiga",
               "ignis", "omni", "ritz", "s cross", "swift", "sx4", "vitara brezza",
               "wagon r"],
    "honda": ["amaze", "brio", "city", "jazz"],
    "toyota": ["camry", "corolla", "corolla altis", "etios cross", "etios g",
               "etios gd", "etios liva", "fortuner", "innova", "land cruiser"],
    "hyundai": ["creta", "eon", "elantra", "grand i10", "i10", "i20", "verna", "xcent"],
}
CAR_TO_BRAND = {m: b for b, models in CAR_BRANDS.items() for m in models}

BIKE_BRANDS = ["hero honda", "royal enfield", "bajaj", "hero", "honda", "activa",
               "hyosung", "ktm", "mahindra", "suzuki", "tvs", "um", "yamaha"]

NUMERIC_FEATURES = ["Car_Age", "log_Present_Price", "log_Driven_kms",
                    "Kms_per_year", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission",
                        "Vehicle_Type", "Brand", "Model"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalize_name(name: object) -> str:
    """lower-case, strip and collapse repeated spaces ('Bajaj  ct 100' -> 'bajaj ct 100')."""
    return " ".join(str(name).lower().split())


def is_bike(name: object) -> bool:
    return normalize_name(name).startswith(BIKE_KEYWORDS)


def get_brand(name: object) -> str:
    n = normalize_name(name)
    if n in CAR_TO_BRAND:
        return CAR_TO_BRAND[n]
    for b in BIKE_BRANDS:
        if n.startswith(b):
            return "honda" if b == "activa" else b
    return "other"


# --------------------------------------------------------------------------
# 1. Loading + cleaning (training only)
# --------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Standardise text, fix types, drop duplicates and impossible rows."""
    df = df.copy()
    n0 = len(df)

    df.columns = [c.strip() for c in df.columns]
    df["Car_Name"] = df["Car_Name"].map(normalize_name)
    for col in ["Fuel_Type", "Selling_type", "Transmission"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    for col in ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates()
    df = df[(df["Selling_Price"] > 0) & (df["Present_Price"] > 0)
            & (df["Driven_kms"] >= 0) & (df["Year"] <= REFERENCE_YEAR)]
    df = df.reset_index(drop=True)

    if verbose:
        print(f"[clean] {n0} rows -> {len(df)} rows "
              f"({n0 - len(df)} duplicate/invalid rows removed)")
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering (runs inside the pipeline)
# --------------------------------------------------------------------------
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the raw columns."""
    X = X.copy()
    name = X["Car_Name"].map(normalize_name)

    X["Model"] = name
    X["Brand"] = name.map(get_brand)
    X["Vehicle_Type"] = name.map(lambda n: "Bike" if is_bike(n) else "Car")

    for col in ["Fuel_Type", "Selling_type", "Transmission"]:
        X[col] = X[col].astype(str).str.strip().str.title()

    age = (REFERENCE_YEAR - pd.to_numeric(X["Year"])).clip(lower=0)
    X["Car_Age"] = age
    X["log_Present_Price"] = np.log1p(pd.to_numeric(X["Present_Price"]))
    kms = pd.to_numeric(X["Driven_kms"]).clip(lower=0)
    X["log_Driven_kms"] = np.log1p(kms)
    X["Kms_per_year"] = np.log1p(kms / (age + 1))
    X["Owner"] = pd.to_numeric(X["Owner"])
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor(scale_numeric: bool = False) -> Pipeline:
    """Feature engineering + encoding. Scaling only matters for linear models."""
    num_steps = StandardScaler() if scale_numeric else "passthrough"
    encoder = ColumnTransformer(
        transformers=[
            ("num", num_steps, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2,
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, validate=False)),
        ("encode", encoder),
    ])


def get_xy(df: pd.DataFrame):
    return df[RAW_FEATURES], df[TARGET]