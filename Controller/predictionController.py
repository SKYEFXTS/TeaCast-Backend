"""
Prediction Controller Module
This module handles tea price prediction requests and responses.
It provides endpoints for fetching ML model predictions and manages the prediction workflow.
"""

from typing import Dict, List, Any, Tuple, Union
from flask import Blueprint, jsonify, Response
from Service.predictionService import get_prediction
import logging

# Create Blueprint for the prediction route
# This blueprint will handle all prediction-related endpoints
prediction_blueprint = Blueprint('prediction', __name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

@prediction_blueprint.route('/predict', methods=['GET'])
def predict() -> Union[Response, Tuple[Response, int]]:
    """Handles the GET request to fetch tea price predictions.
    
    This endpoint processes requests for tea price forecasts by calling
    the prediction service and returning formatted results. No input
    parameters are required as predictions are based on historical data
    already loaded by the service.
    
    Returns:
        Union[Response, Tuple[Response, int]]: One of:
            - Success (200): JSON response with predictions
            - Bad Request (400): Error message for invalid inputs 
            - Server Error (500): Error message for processing failures
    
    Example response:
        {
            "prediction": [
                {"Auction_Number": 1, "Final_Prediction": 1250},
                {"Auction_Number": 2, "Final_Prediction": 1240},
                ...
            ]
        }
    """
    try:
        # Get the prediction result from the service layer
        prediction: List[Dict[str, Any]] = get_prediction()

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
