import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import engineer_features

def main():
    # 1. Load local dataset
    csv_file = "data/car data.csv"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_file}'. Please place your CSV file inside the 'data/' folder."
        )
    
    print(f"Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 2. Process data using EDA and feature engineering (utilizing every column)
    df_processed = engineer_features(df)
    
    TARGET = 'Selling_Price'
    if TARGET not in df_processed.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset columns.")
        
    X = df_processed.drop(columns=[TARGET])
    y = df_processed[TARGET]
    
    # Explicitly separate column types ensuring Owner, Present_Price, and Driven_kms are treated numerically
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Build Preprocessing Pipelines
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # 5. Model Pipeline Construction & Hyperparameter Tuning
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    param_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__learning_rate': [0.03, 0.08],
        'regressor__max_depth': [4, 6]
    }
    
    print("\nTraining comprehensive model with 5-Fold Cross-Validation...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters found:", grid_search.best_params_)
    
    # 6. Evaluate Model on Unseen Test Data
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n--- Final Test Evaluation ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    
    # 7. Save Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/car_price_pipeline_v1.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nModel artifact successfully saved to: {model_path}")

if __name__ == "__main__":
    main()