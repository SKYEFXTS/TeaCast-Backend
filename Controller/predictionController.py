"""
Prediction Controller Module
This module handles tea price prediction requests and responses.
It provides endpoints for fetching ML model predictions and manages the prediction workflow.
"""

from typing import Dict, List, Any, Tuple, Union
from flask import Blueprint, jsonify, Response, request
import time
import logging
from Service.predictionService import get_prediction

# Create Blueprint for the prediction route
# This blueprint will handle all prediction-related endpoints
prediction_blueprint = Blueprint('prediction', __name__)

# Get a logger specific to this module
logger = logging.getLogger(__name__)

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
    start_time = time.time()
    request_id = request.headers.get('X-Request-ID', 'unknown')
    client_ip = request.remote_addr
    
    logger.info(f"Tea price prediction request received - ID: {request_id}, IP: {client_ip}")
    
    try:
        # Get the prediction result from the service layer
        prediction: List[Dict[str, Any]] = get_prediction()

        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        # Log the prediction result summary
        prediction_count = len(prediction) if prediction else 0
        logger.info(f"Prediction completed successfully - Request ID: {request_id}, Processing time: {elapsed_time:.3f}s, Items: {prediction_count}")
        logger.debug(f"Detailed prediction result: {prediction}")

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    except ValueError as ve:
        # Handle invalid input errors (e.g., missing required parameters)
        elapsed_time = time.time() - start_time
        logger.error(f"Invalid input error - Request ID: {request_id}, Time: {elapsed_time:.3f}s, Error: {str(ve)}")
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        # Handle unexpected errors gracefully
        # This includes ML model errors, data processing errors, etc.
        elapsed_time = time.time() - start_time
        logger.error(f"Internal server error - Request ID: {request_id}, Time: {elapsed_time:.3f}s, Error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
