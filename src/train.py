"""
train.py
--------
Trains several regression models on the FULL cleaned dataset, compares them with
repeated cross-validation, tunes the best candidates, evaluates on a held-out
test set, then refits the winner on all data and saves it.

Run from the project root:   python -m src.train
"""
import json
import warnings

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     RepeatedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

from src.preprocess import (ROOT, build_preprocessor, clean_data, get_xy,
                            load_data, REFERENCE_YEAR)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "car_price_pipeline_v1.pkl"
METRICS_PATH = MODEL_DIR / "metrics_v1.json"


def make_pipeline(estimator, scale: bool = False) -> TransformedTargetRegressor:
    """preprocessing -> estimator, predicting log(price) and converting back."""
    pipe = Pipeline([("prep", build_preprocessor(scale_numeric=scale)),
                     ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "R2": round(float(r2_score(y, pred)), 4),
        "MAE": round(float(mean_absolute_error(y, pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
    }


def train_and_save(verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    df = clean_data(load_data(), verbose=verbose)
    X, y = get_xy(df)
    log(f"[data] {X.shape[0]} rows, {X.shape[1]} raw features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # ---- 1. compare candidate models (repeated 5-fold CV on the training split)
    candidates = {
        "Ridge": make_pipeline(Ridge(alpha=1.0), scale=True),
        "RandomForest": make_pipeline(RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "ExtraTrees": make_pipeline(ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "GradientBoosting": make_pipeline(GradientBoostingRegressor(
            random_state=RANDOM_STATE)),
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cv_results = {}
    log("\n[cv] model comparison (5-fold x 3 repeats, R2 on price in lakhs)")
    for name, mdl in candidates.items():
        scores = cross_val_score(mdl, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_results[name] = float(scores.mean())
        log(f"     {name:<18} R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ---- 2. tune every candidate
    search_spaces = {
        "Ridge": {"regressor__model__alpha": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        "RandomForest": {
            "regressor__model__n_estimators": [200, 400, 600],
            "regressor__model__max_depth": [None, 8, 12, 16],
            "regressor__model__min_samples_leaf": [1, 2, 3],
            "regressor__model__max_features": [0.5, 0.8, 1.0],
        },
        "ExtraTrees": {
            "regressor__model__n_estimators": [200, 400, 600],
            "regressor__model__max_depth": [None, 8, 12, 16],
            "regressor__model__min_samples_leaf": [1, 2, 3],
            "regressor__model__max_features": [0.5, 0.8, 1.0],
        },
        "GradientBoosting": {
            "regressor__model__n_estimators": [200, 400, 600],
            "regressor__model__learning_rate": [0.02, 0.05, 0.1],
            "regressor__model__max_depth": [2, 3, 4],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
            "regressor__model__min_samples_leaf": [1, 3, 5],
        },
    }
    tuned = {}
    log("\n[tune] hyper-parameter search for every candidate")
    inner_cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    for name, space in search_spaces.items():
        if name == "Ridge":   # tiny grid -> exhaustive search
            rs = GridSearchCV(candidates[name], space, cv=inner_cv, scoring="r2", n_jobs=-1)
        else:
            rs = RandomizedSearchCV(candidates[name], space, n_iter=15, cv=inner_cv,
                                    scoring="r2", random_state=RANDOM_STATE, n_jobs=-1)
        rs.fit(X_train, y_train)
        tuned[name] = rs
        log(f"       {name:<18} best CV R2 = {rs.best_score_:.4f}")

    best_name = max(tuned, key=lambda n: tuned[n].best_score_)
    best_params = tuned[best_name].best_params_
    log(f"\n[best] {best_name}  {best_params}")

    # ---- 3. honest evaluation on the untouched test split
    best = tuned[best_name].best_estimator_
    test_metrics = evaluate(best, X_test, y_test)
    train_metrics = evaluate(best, X_train, y_train)
    log(f"[test]  R2={test_metrics['R2']}  MAE={test_metrics['MAE']} lakh  "
        f"RMSE={test_metrics['RMSE']} lakh")
    log(f"[train] R2={train_metrics['R2']}  MAE={train_metrics['MAE']} lakh")

    # ---- 4. uncertainty band from out-of-fold residuals (log scale)
    oof = cross_val_predict(best, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    log_resid = np.log1p(y.values) - np.log1p(np.clip(oof, 0, None))
    lo, hi = np.quantile(log_resid, [0.10, 0.90])

    # ---- 5. refit on ALL data (use every row) and save
    final = clone(best)   # same tuned hyper-parameters, refit on every row
    final.fit(X, y)
    final_metrics = evaluate(final, X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "best_params": {k.split("__")[-1]: v for k, v in best_params.items()},
        "cv_r2": {k: round(v, 4) for k, v in cv_results.items()},
        "tuned_cv_r2": round(float(tuned[best_name].best_score_), 4),
        "test": test_metrics,
        "train": train_metrics,
        "oof_residual_log_q10": float(lo),
        "oof_residual_log_q90": float(hi),
        "n_rows": int(len(df)),
        "reference_year": REFERENCE_YEAR,
        "car_names": sorted(df["Car_Name"].unique().tolist()),
        "price_range_lakh": [float(df["Present_Price"].min()), float(df["Present_Price"].max())],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"\n[saved] {MODEL_PATH}\n[saved] {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_and_save()