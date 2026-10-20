# 🚗 Car Price Prediction — CodeAlpha Data Science Internship (Task 3)

Predicts the resale (selling) price of a used car/bike from its showroom price, age,
kilometres driven, fuel, transmission, seller type, owners and brand/model.
Includes a Flask web app (form + chat assistant) ready to deploy on **Render**.

## Project structure
```
car_price_prediction/
├── data/car data.csv                  # dataset (301 rows)
├── models/car_price_pipeline_v1.pkl   # saved pipeline (preprocessing + model)
├── models/metrics_v1.json             # metrics + settings used by the web app
├── reports/figures/                   # EDA charts
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # cleaning + feature engineering + encoders
│   ├── eda.py                         # exploratory data analysis
│   └── train.py                       # model comparison, tuning, evaluation, saving
├── templates/index.html               # website (form + assistant)
├── app.py                             # Flask app
├── requirements.txt
├── render.yaml                        # Render deployment config
└── README.md
```

## Workflow
1. **Cleaning** – normalise names (`"Bajaj  ct 100"` → `"bajaj ct 100"`), fix types, drop duplicates / invalid rows.
2. **Feature engineering** – `Car_Age`, `log(Present_Price)`, `log(Driven_kms)`, `Kms_per_year`,
   `Brand` (brand goodwill), `Vehicle_Type` (car/bike), model name.
3. **EDA** – distributions, outliers, price vs. every feature, correlations (`python -m src.eda`).
4. **Modelling** – Ridge, Random Forest, Extra Trees, Gradient Boosting on a **log-price target**;
   repeated 5-fold CV, hyper-parameter search, 80/20 hold-out test, then refit on **all rows**.
5. **Deployment** – Flask API + web UI.

## Results (hold-out test set)
| Model | CV R² |
|---|---|
| **Ridge (chosen)** | **0.956** |
| Gradient Boosting | 0.906 |
| Random Forest | 0.890 |
| Extra Trees | 0.877 |

Test R² ≈ 0.94, MAE ≈ ₹77,000. Prices in the dataset are in **lakh ₹** (1 lakh = ₹100,000).
The app also shows a likely price range built from out-of-fold residuals.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.eda                 # optional: EDA charts
python -m src.train               # trains and saves models/car_price_pipeline_v1.pkl
python app.py                     # open http://127.0.0.1:5000
```
> Always run `src` modules from the project root with `python -m ...`.

## Deploy on Render
1. Push the project to GitHub (repo name: `CodeAlpha_CarPricePrediction`) — **commit the `models/` folder**.
2. Render → **New → Web Service** → connect the repo (it will pick up `render.yaml`), or set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
3. Deploy. Health check endpoint: `/health`.

If the `.pkl` was built with a different scikit-learn version, the app automatically retrains on startup.
To avoid that, pin your local version in `requirements.txt` (`pip freeze | findstr scikit-learn`).

## Limitations
The data is a small snapshot (up to 2018, India). Predictions for much newer vehicles or
unseen models are extrapolations, so treat results as estimates, not valuations.