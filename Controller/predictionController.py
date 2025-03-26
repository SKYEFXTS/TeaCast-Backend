from flask import Blueprint, jsonify
from Service.predictionService import get_prediction
import logging

# Create Blueprint for the prediction route
prediction_blueprint = Blueprint('prediction', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@prediction_blueprint.route('/', methods=['GET'])
def predict():
    """Handles the GET request to fetch predictions."""
    try:
        # Get the prediction result from the service layer
        prediction = get_prediction()

        # Log the prediction for traceability
        logging.debug(f'Prediction successfully generated: {prediction}')

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except ValueError as ve:
        # Handle invalid input errors
        logging.error(f'Invalid input error: {ve}')
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        # Handle unexpected errors gracefully
        logging.error(f'Internal server error: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
