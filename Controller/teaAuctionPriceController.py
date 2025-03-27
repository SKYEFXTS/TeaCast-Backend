from flask import Blueprint, jsonify
from Service.teaAuctionPriceService import get_last_auctions_average_prices
import logging

# Create Blueprint for the auction route
tea_auction_price_blueprint = Blueprint('auction', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@tea_auction_price_blueprint.route('/tea-auction-price', methods=['GET'])
def average_prices():
    """Handles the GET request to fetch average prices for the last auctions."""
    try:
        # Get the average prices from the service layer
        average_prices = get_last_auctions_average_prices()

        # Log the successful fetch of average prices
        logging.debug(f'Average prices successfully fetched: {average_prices}')

        # Return the average prices as a JSON response
        return jsonify({'average_prices': average_prices})

    except Exception as e:
        # Log and return an error response if something goes wrong
        logging.error(f'Internal server error: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
