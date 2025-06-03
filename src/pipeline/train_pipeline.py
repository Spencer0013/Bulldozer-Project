# src/utils.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def train_pipeline(train_csv_path, model_output_path):
    """
    Train a RandomForestRegressor pipeline on bulldozer data.
    
    Args:
        train_csv_path (str): Path to the CSV file containing training data.
        model_output_path (str): Path to save the trained pipeline (.pkl).
    """
    # Load training data
    df = pd.read_csv(train_csv_path, parse_dates=["saledate"])
    if "SalePrice" not in df.columns:
        raise ValueError("Target column 'SalePrice' not found in training data.")
    
    y = df["SalePrice"]
    X = df.drop("SalePrice", axis=1)
    
    # Feature Engineering: Extract date features
    X["saleYear"] = X["saledate"].dt.year
    X["saleMonth"] = X["saledate"].dt.month
    X["saleDay"] = X["saledate"].dt.day
    X["saleDayOfWeek"] = X["saledate"].dt.dayofweek
    X["saleDayOfYear"] = X["saledate"].dt.dayofyear
    X = X.drop("saledate", axis=1)
    
    # Select columns
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Build Preprocessor
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ("num", SimpleImputer(strategy="median"), numerical_cols)
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X, y)
    
    # Save pipeline
    joblib.dump(pipeline, model_output_path)
    print(f"✅ Pipeline trained and saved to {model_output_path}")

