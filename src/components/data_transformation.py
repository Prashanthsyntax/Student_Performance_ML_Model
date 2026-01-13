# =========================
# Import required libraries
# =========================

import sys                                  # Used for system-level exception handling
import os                                   # Used for file path operations

from dataclasses import dataclass            # Used to create config classes

import numpy as np                           # Used for numerical operations
import pandas as pd                          # Used to read CSV files

# Scikit-learn tools for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom exception and logging
from src.exception import CustomException
from src.logger import logging

# Utility function to save objects (pickle)
from src.utils import save_object


# =========================================================
# DataTransformationConfig
# Stores path to save preprocessing object
# =========================================================

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation
    """
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl'
    )


# =========================================================
# DataTransformation class
# Responsible for feature engineering & preprocessing
# =========================================================

class DataTransformation:
    def __init__(self):
        """
        Constructor initializes configuration object
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method creates and returns a preprocessing object
        that handles:
        - Numerical feature scaling and imputation
        - Categorical feature encoding and imputation
        """

        try:
            # =========================
            # Define column names
            # =========================

            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # =========================
            # Numerical pipeline
            # =========================
            # Steps:
            # 1. Replace missing values with median
            # 2. Standardize numerical features

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # =========================
            # Categorical pipeline
            # =========================
            # Steps:
            # 1. Replace missing values with most frequent value
            # 2. Convert categories to one-hot encoded values
            # 3. Scale features (without centering due to sparsity)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Log column information
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # =========================
            # Combine pipelines using ColumnTransformer
            # =========================

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Return preprocessing object
            return preprocessor

        except Exception as e:
            # Raise custom exception if error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This method:
        1. Reads train and test data
        2. Applies preprocessing
        3. Separates input features and target
        4. Saves preprocessing object
        5. Returns transformed arrays
        """

        try:
            # =========================
            # Read train and test datasets
            # =========================

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded successfully")

            # =========================
            # Get preprocessing object
            # =========================

            logging.info("Creating preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # =========================
            # Define target and numerical columns
            # =========================

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # =========================
            # Split input features and target
            # =========================

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1
            )
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing to train and test data")

            # =========================
            # Apply transformations
            # =========================

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # =========================
            # Combine input features with target column
            # =========================

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # =========================
            # Save preprocessing object
            # =========================

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # =========================
            # Return transformed data
            # =========================

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Handle errors using custom exception
            raise CustomException(e, sys)
