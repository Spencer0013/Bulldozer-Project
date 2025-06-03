import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        try:
            logging.info("Starting model training...")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report, fitted_models = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_val, y_test=y_val,
                models=models, param=params
            )

            best_model_name = min(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = fitted_models[best_model_name]

            logging.info(f"Best model: {best_model_name} with RMSLE: {best_model_score:.4f}")

            if best_model_score > 1.0:
                raise CustomException("No suitable model found (RMSLE too high).")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
             
            predicted = best_model.predict(X_val)

            Rmsle = np.sqrt(mean_squared_log_error(y_val, predicted))


           

            return best_model, Rmsle

        except Exception as e:
            raise CustomException(e, sys)
