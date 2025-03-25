from flask import Blueprint, request, jsonify
from Service.predictionService import get_prediction
import logging

# Create Blueprint for the prediction route
prediction_blueprint = Blueprint('prediction', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@prediction_blueprint.route('/', methods=['GET'])
def predict():
    try:
        # Get the prediction result from the service layer
        prediction = get_prediction()

        # Log the prediction
        logging.debug(f'Prediction: {prediction}')

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except ValueError as ve:
        logging.error(f'Invalid input: {ve}')
        return jsonify({'error': 'Invalid input: ' + str(ve)}), 400
    except Exception as e:
        logging.error(f'Internal server error: {e}')
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500
