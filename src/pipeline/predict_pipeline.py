# Import sys module to access system-specific parameters and exception details
import sys

# Import os module to work with file paths (used for model/preprocessor paths)
import os

# Import pandas for creating and handling DataFrames
import pandas as pd

# Import custom exception class for consistent error handling across the project
from src.exception import CustomException

# Import utility function to load serialized objects (model & preprocessor)
from src.utils import load_object


# ===========================
# Prediction Pipeline Class
# ===========================
class PredictPipeline:
    def __init__(self):
        # Constructor method
        # Currently no initialization is required, so we use pass
        pass

    def predict(self, features):
        """
        This method takes input features,
        loads the trained model and preprocessor,
        transforms the input data,
        and returns the model predictions.
        """
        try:
            # Define the file path for the trained model
            model_path = os.path.join("artifacts", "model.pkl")

            # Define the file path for the trained preprocessor
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Debug message before loading model and preprocessor
            print("Before Loading")

            # Load the trained model object from disk
            model = load_object(file_path=model_path)

            # Load the trained preprocessor object from disk
            preprocessor = load_object(file_path=preprocessor_path)

            # Debug message after successful loading
            print("After Loading")

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Generate predictions using the trained model
            preds = model.predict(data_scaled)

            # Return the prediction results
            return preds

        except Exception as e:
            # Raise a custom exception with system-level details
            raise CustomException(e, sys)


# ===========================
# Custom Input Data Class
# ===========================
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        """
        This constructor initializes all input features
        required for prediction.
        """

        # Store gender value (e.g., 'male', 'female')
        self.gender = gender

        # Store race/ethnicity category
        self.race_ethnicity = race_ethnicity

        # Store parent's education level
        self.parental_level_of_education = parental_level_of_education

        # Store lunch type (standard / free/reduced)
        self.lunch = lunch

        # Store test preparation course status
        self.test_preparation_course = test_preparation_course

        # Store reading score
        self.reading_score = reading_score

        # Store writing score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the input data into a Pandas DataFrame
        so it can be passed into the preprocessor and model.
        """
        try:
            # Create a dictionary where keys match training feature names
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to a Pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise custom exception if DataFrame creation fails
            raise CustomException(e, sys)
        

        