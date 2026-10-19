# 🚗 End-to-End Used Car Price Prediction

A production-grade machine learning project predicting secondhand vehicle market prices[cite: 1, 2]. Built with **Python**, **Scikit-Learn**, **Pandas**, and deployed via an interactive **Streamlit** application[cite: 1, 2].

## 🛠️ Tech Stack & Concepts
* **Data Engineering:** Automated feature engineering (extracting vehicle age and brand metadata)[cite: 1, 2].
* **Pipelines & Robustness:** Implemented `ColumnTransformer` and `Pipeline` architectures to prevent data leakage during scaling and encoding[cite: 1].
* **Model Selection:** Benchmarked linear models against ensemble architectures (Random Forest & Gradient Boosting) using 5-fold cross-validation[cite: 1, 2].
* **MLOps:** Artifact serialization via `joblib` for plug-and-play production inference[cite: 1].

## 🚀 Getting Started Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/car-price-prediction.git](https://github.com/your-username/car-price-prediction.git)
   cd car-price-prediction