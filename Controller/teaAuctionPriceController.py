"""
Tea Auction Price Controller Module
This module handles requests for tea auction price data and statistics.
It provides endpoints for fetching historical auction prices and related analytics.
"""

from flask import Blueprint, jsonify
from Service.teaAuctionPriceService import get_last_auctions_average_prices
import logging

# Create Blueprint for the auction route
# This blueprint will handle all tea auction price-related endpoints
tea_auction_price_blueprint = Blueprint('auction', __name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

@tea_auction_price_blueprint.route('/tea-auction-price', methods=['GET'])
def average_prices():
    """
    Handles the GET request to fetch average prices for the last auctions.
    
    Returns:
        JSON response with:
        - Success (200): {"average_prices": [price_data]}
        - Server Error (500): {"error": "Internal server error message"}
        
    The price_data includes historical auction prices and related statistics.
    """
    try:
        # Get the average prices from the service layer
        # This retrieves and processes historical auction data
        average_prices = get_last_auctions_average_prices()

        # Log the successful fetch of average prices for monitoring
        logging.debug(f'Average prices successfully fetched: {average_prices}')

        # Return the average prices as a JSON response
        return jsonify({'average_prices': average_prices})

    except Exception as e:
        # Log and return an error response if something goes wrong
        # This handles any errors during data retrieval or processing
        logging.error(f'Internal server error: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
