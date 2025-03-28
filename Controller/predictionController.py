"""
Prediction Controller Module
This module handles tea price prediction requests and responses.
It provides endpoints for fetching ML model predictions and manages the prediction workflow.
"""

from flask import Blueprint, jsonify
from Service.predictionService import get_prediction
import logging

# Create Blueprint for the prediction route
# This blueprint will handle all prediction-related endpoints
prediction_blueprint = Blueprint('prediction', __name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

@prediction_blueprint.route('/predict', methods=['GET'])
def predict():
    """
    Handles the GET request to fetch tea price predictions.
    
    Returns:
        JSON response with:
        - Success (200): {"prediction": [predicted_values]}
        - Bad Request (400): {"error": "Invalid input error message"}
        - Server Error (500): {"error": "Internal server error message"}
    """
    try:
        # Get the prediction result from the service layer
        # This calls the ML models and processes the results
        prediction = get_prediction()

        # Log the prediction for traceability and monitoring
        logging.debug(f'Prediction successfully generated: {prediction}')

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    except ValueError as ve:
        # Handle invalid input errors (e.g., missing required parameters)
        logging.error(f'Invalid input error: {ve}')
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        # Handle unexpected errors gracefully
        # This includes ML model errors, data processing errors, etc.
        logging.error(f'Internal server error: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
