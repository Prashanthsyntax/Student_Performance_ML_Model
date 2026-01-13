# =========================
# Import required libraries
# =========================

import os                                  # Used for file path handling
import sys                                 # Used for system-level exception handling

from dataclasses import dataclass          # Used to create configuration class

# =========================
# Machine Learning Models
# =========================

from catboost import CatBoostRegressor     # Gradient boosting model (handles categorical well)

from sklearn.ensemble import (
    AdaBoostRegressor,                    # Boosting-based ensemble model
    GradientBoostingRegressor,             # Gradient boosting regressor
    RandomForestRegressor,                 # Bagging-based ensemble model
)

from sklearn.linear_model import LinearRegression   # Linear regression model
from sklearn.metrics import r2_score                 # Evaluation metric
from sklearn.neighbors import KNeighborsRegressor    # KNN regressor (not used but imported)
from sklearn.tree import DecisionTreeRegressor       # Decision tree model

from xgboost import XGBRegressor            # Extreme Gradient Boosting model

# =========================
# Custom utilities
# =========================

from src.exception import CustomException   # Custom exception handling
from src.logger import logging              # Custom logging utility
from src.utils import save_object, evaluate_models  # Helper functions


# =========================================================
# ModelTrainerConfig
# Stores path for saving trained model
# =========================================================

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for Model Trainer
    """
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


# =========================================================
# ModelTrainer class
# Responsible for training and selecting best model
# =========================================================

class ModelTrainer:
    def __init__(self):
        """
        Constructor initializes configuration
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        This method:
        1. Splits features and target
        2. Trains multiple ML models
        3. Performs hyperparameter tuning
        4. Selects the best model based on R² score
        5. Saves the trained model
        """

        try:
            # =========================
            # Split input features and target
            # =========================

            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # All columns except last as features
                train_array[:, -1],    # Last column as target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # =========================
            # Define models
            # =========================

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # =========================
            # Hyperparameter grid
            # =========================

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson"
                    ],
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # "CatBoosting Regressor": {
                #     "depth": [6, 8, 10],
                #     "learning_rate": [0.01, 0.05, 0.1],
                #     "iterations": [30, 50, 100]
                # },

                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            # =========================
            # Train and evaluate models
            # =========================

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # =========================
            # Select best model
            # =========================

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # =========================
            # Check minimum performance threshold
            # =========================

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")

            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            # =========================
            # Save trained model
            # =========================

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # =========================
            # Evaluate final model
            # =========================

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Wrap and raise exception
            raise CustomException(e, sys)
