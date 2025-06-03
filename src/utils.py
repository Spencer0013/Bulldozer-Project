import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the 'saledate' column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'saledate' column.
    
    Returns:
        pd.DataFrame: DataFrame with added date-related columns.
    """
    df = df.copy()

    if "saledate" not in df.columns:
        raise KeyError("The column 'saledate' is missing from the input DataFrame.")
    
    if not np.issubdtype(df["saledate"].dtype, np.datetime64):
        raise TypeError("The 'saledate' column must be of datetime type. Use `parse_dates=['saledate']` when reading the CSV.")

    df["saleYear"] = df["saledate"].dt.year
    df["saleMonth"] = df["saledate"].dt.month
    df["saleDay"] = df["saledate"].dt.day
    df["saleDayOfWeek"] = df["saledate"].dt.dayofweek
    df["saleDayOfYear"] = df["saledate"].dt.dayofyear

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that preprocesses numerical and categorical features.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame used to infer column types.
    
    Returns:
        ColumnTransformer: A transformer object for preprocessing.
    """
    df = df.copy()

    # Drop the target if present
    df.drop(columns=["SalePrice"], inplace=True, errors="ignore")

    # Select column types
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Define transformers
    num_transformer = SimpleImputer(strategy="median")

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Combine in a ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    return preprocessor






