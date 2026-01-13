# Import Flask class to create the web application
# request is used to fetch form data from HTML
# render_template is used to render HTML templates
from flask import Flask, request, render_template

# Import NumPy (not directly used here but commonly required in ML apps)
import numpy as np

# Import pandas for DataFrame handling
import pandas as pd

# Import StandardScaler (not used here, but often used in ML pipelines)
from sklearn.preprocessing import StandardScaler

# Import CustomData and PredictPipeline classes for prediction
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# Create a Flask application instance
application = Flask(__name__)

# Create an alias for the Flask app (used by deployment services)
app = application


# ===========================
# Route for Home Page
# ===========================
@app.route('/')
def index():
    """
    This function renders the landing page of the application.
    """
    return render_template('index.html')


# ===========================
# Route for Prediction Page
# ===========================
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    This function handles both:
    - GET request: show prediction form
    - POST request: process form data and return prediction
    """

    # If the request type is GET, render the input form page
    if request.method == 'GET':
        return render_template('home.html')

    # If the request type is POST, collect form data and predict
    else:
        # Create a CustomData object using user input from the form
        data = CustomData(
            # Fetch gender input from HTML form
            gender=request.form.get('gender'),

            # Fetch ethnicity input from HTML form
            race_ethnicity=request.form.get('ethnicity'),

            # Fetch parental education level
            parental_level_of_education=request.form.get(
                'parental_level_of_education'
            ),

            # Fetch lunch type
            lunch=request.form.get('lunch'),

            # Fetch test preparation course status
            test_preparation_course=request.form.get(
                'test_preparation_course'
            ),

            # Fetch reading score and convert it to float
            # NOTE: Your form field names are swapped here intentionally
            reading_score=float(request.form.get('writing_score')),

            # Fetch writing score and convert it to float
            writing_score=float(request.form.get('reading_score'))
        )

        # Convert collected data into a Pandas DataFrame
        pred_df = data.get_data_as_data_frame()

        # Print DataFrame to console for debugging
        print(pred_df)

        # Debug message before prediction
        print("Before Prediction")

        # Create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()

        # Debug message during prediction
        print("Mid Prediction")

        # Generate prediction result
        results = predict_pipeline.predict(pred_df)

        # Debug message after prediction
        print("After Prediction")

        # Render the same page with prediction result displayed
        return render_template('home.html', results=results[0])


# ===========================
# Application Entry Point
# ===========================
if __name__ == "__main__":
    # Run the Flask app on all network interfaces
    # This is useful for Docker or cloud deployment
    app.run(host="0.0.0.0", debug=True)
