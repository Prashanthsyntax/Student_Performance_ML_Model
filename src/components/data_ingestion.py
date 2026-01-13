# =========================
# Import required libraries
# =========================

import os                      # Used for file and directory operations
import sys                     # Used to get system-specific information (for exception handling)

import pandas as pd            # Used for data manipulation and analysis

from sklearn.model_selection import train_test_split  # Used to split dataset into train and test

from dataclasses import dataclass  # Used to create configuration classes easily

# Custom exception class for better error handling
from src.exception import CustomException

# Custom logging module to track execution logs
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# =========================================================
# DataIngestionConfig class
# Holds paths for saving ingested data
# =========================================================

@dataclass
class DataIngestionConfig:
    """
    This class stores file paths for:
    - Training data
    - Testing data
    - Raw data
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# =========================================================
# DataIngestion class
# Responsible for reading data and splitting it
# =========================================================

class DataIngestion:
    def __init__(self):
        """
        Constructor initializes the DataIngestionConfig object
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This method:
        1. Reads the dataset
        2. Saves raw data
        3. Splits data into train and test sets
        4. Saves train and test CSV files
        """

        # Log the start of data ingestion
        logging.info("Entered the Data Ingestion component")

        try:
            # =========================
            # Read dataset into DataFrame
            # =========================

            # NOTE: Use raw string (r'') to avoid Windows path issues
            df = pd.read_csv(r'notebook\data\stud.csv')

            logging.info("Dataset successfully read into DataFrame")

            # =========================
            # Create artifacts directory
            # =========================

            # os.path.dirname extracts the folder name ("artifacts")
            # exist_ok=True prevents error if folder already exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # =========================
            # Save raw dataset
            # =========================

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Raw data saved successfully")

            # =========================
            # Train-test split
            # =========================

            logging.info("Initiating train-test split")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,      # 20% data for testing
                random_state=42     # Ensures reproducibility
            )

            # =========================
            # Save train dataset
            # =========================

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # =========================
            # Save test dataset
            # =========================

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            # =========================
            # Return file paths
            # =========================

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Wrap any exception into CustomException
            raise CustomException(e, sys)


# =========================================================
# Main execution block
# =========================================================

if __name__ == "__main__":
    """
    This block runs only when this file is executed directly
    """

    # Create DataIngestion object
    obj = DataIngestion()

    # Start data ingestion process
    train_data, test_data = obj.initiate_data_ingestion()

    # =========================
    # Future pipeline steps
    # =========================

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
