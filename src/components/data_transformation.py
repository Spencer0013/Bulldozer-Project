# src/components/data_transformation.py

import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        # 1) Explicit lists of numeric / categorical columns
        self.num_cols = [
            "SalesID", "MachineID", "ModelID", "datasource", "auctioneerID",
            "YearMade", "MachineHoursCurrentMeter",
            "saleYear", "saleMonth", "saleDay", "saleDayOfWeek", "saleDayOfYear",
        ]
        self.cat_cols = [
            "UsageBand", "fiModelDesc", "fiBaseModel", "fiSecondaryDesc",
            "fiModelSeries", "fiModelDescriptor", "ProductSize", "fiProductClassDesc",
            "state", "ProductGroup", "ProductGroupDesc", "Drive_System", "Enclosure",
            "Forks", "Pad_Type", "Ride_Control", "Stick", "Transmission", "Turbocharged",
            "Blade_Extension", "Blade_Width", "Enclosure_Type", "Engine_Horsepower",
            "Hydraulics", "Pushblock", "Ripper", "Scarifier", "Tip_Control", "Tire_Size",
            "Coupler", "Coupler_System", "Grouser_Tracks", "Hydraulics_Flow", "Track_Type",
            "Undercarriage_Pad_Width", "Stick_Length", "Thumb", "Pattern_Changer",
            "Grouser_Type", "Backhoe_Mounting", "Blade_Type", "Travel_Controls",
            "Differential_Type", "Steering_Controls",
        ]

        # 2) Define numeric pipeline (median → scale)
        self.num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # 3) Define categorical pipeline (most_frequent → one-hot → keep sparse)
        self.cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            # --- 1) Read only the smaller CSVs under artifacts/ ---
            train_df = pd.read_csv(train_path)
            val_df   = pd.read_csv(val_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Loaded train/val/test data from artifacts/")

            target = "SalePrice"

            # --- 2) Split X / y for train & val ---
            X_train = train_df[self.num_cols + self.cat_cols]
            y_train = train_df[target].to_numpy()

            X_val   = val_df[self.num_cols + self.cat_cols]
            y_val   = val_df[target].to_numpy()

            # --- 3) For test: if SalePrice exists, separate; else treat all as X_test ---
            if target in test_df.columns:
                X_test = test_df[self.num_cols + self.cat_cols]
                y_test = test_df[target].to_numpy()
            else:
                X_test = test_df[self.num_cols + self.cat_cols].copy()
                y_test = None

            # --- 4) Fit numeric & categorical pipelines on training set ---
            logging.info("Fitting numeric pipeline on train data")
            X_train_num = self.num_pipeline.fit_transform(X_train[self.num_cols])
            logging.info("Fitting categorical pipeline on train data")
            # Check cardinality of each categorical column
            for col in self.cat_cols:
                 unique_vals = X_train[col].nunique()
                 print(f"{col}: {unique_vals} unique values")
            X_train_cat = self.cat_pipeline.fit_transform(X_train[self.cat_cols])

            # Stack numeric (dense) + categorical (sparse) → sparse matrix
            X_train_sp = sparse.hstack([X_train_num, X_train_cat], format="csr")

            # --- 5) Transform validation & test with the already-fitted pipelines ---
            logging.info("Transforming validation data")
            X_val_num = self.num_pipeline.transform(X_val[self.num_cols])
            X_val_cat = self.cat_pipeline.transform(X_val[self.cat_cols])
            X_val_sp = sparse.hstack([X_val_num, X_val_cat], format="csr")

            logging.info("Transforming test data")
            X_test_num = self.num_pipeline.transform(X_test[self.num_cols])
            X_test_cat = self.cat_pipeline.transform(X_test[self.cat_cols])
            X_test_sp = sparse.hstack([X_test_num, X_test_cat], format="csr")

            # --- 6) Save fitted pipelines together as a dict for later inference ---
            # We’ll store {"num_pipeline":..., "cat_pipeline":..., "num_cols":..., "cat_cols":...}
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj={
                    "num_pipeline": self.num_pipeline,
                    "cat_pipeline": self.cat_pipeline,
                    "num_cols": self.num_cols,
                    "cat_cols": self.cat_cols,
                },
            )
            logging.info("Preprocessing pipelines saved to %s", self.config.preprocessor_obj_file_path)

            # --- 7) Return sparse matrices + targets separately ---
            return (
                X_train_sp, y_train,
                X_val_sp,   y_val,
                X_test_sp,  y_test,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------
# Driver block: calls transformation only on existing
# artifacts/train.csv, val.csv, test.csv
# -------------------------------------------------------
if __name__ == "__main__":
    train_path = os.path.join("artifacts", "train.csv")
    val_path   = os.path.join("artifacts", "val.csv")
    test_path  = os.path.join("artifacts", "test.csv")

    transformer = DataTransformation()
    (
        X_train_sp, y_train,
        X_val_sp,   y_val,
        X_test_sp,  y_test,
        preprocessor_path
    ) = transformer.initiate_data_transformation(
        train_path, val_path, test_path
    )

    print("Transformation complete.")
    print("Preprocessor saved to:", preprocessor_path)
    print("X_train shape (sparse):", X_train_sp.shape)
    print("y_train shape:", y_train.shape)
    print("X_val   shape (sparse):", X_val_sp.shape)
    print("y_val   shape:", y_val.shape)
    if y_test is not None:
        print("X_test  shape (sparse):", X_test_sp.shape)
        print("y_test  shape:", y_test.shape)
    else:
        print("X_test  shape (sparse):", X_test_sp.shape)
        print("y_test is None (no 'SalePrice' in test set).")



