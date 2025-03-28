"""
Tea Dashboard Controller Module
This module handles requests for the tea dashboard analytics and visualizations.
It provides endpoints for fetching comprehensive tea market data and statistics.
"""

from flask import Blueprint, jsonify
from Service.teaDashboardService import get_tea_price_over_time, get_all_average_prices
import logging

# Create Blueprint for the dashboard route
# This blueprint will handle all dashboard-related endpoints
tea_dashboard_blueprint = Blueprint('dashboard', __name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

@tea_dashboard_blueprint.route('/dashboard', methods=['GET'])
def tea_price_time_series():
    """
    Handles GET request to fetch historical data for tea prices, USD buying rate, crude oil price,
    and average prices for all categories.
    
    Returns:
        JSON response with:
        - Success (200): {
            "tea_price_data": [historical_price_data],
            "all_average_price_data": [category_average_data]
          }
        - Server Error (500): {"error": "Error message"}
        
    The response includes comprehensive market data for dashboard visualizations.
    """
    try:
        # Get the historical tea price data over time
        # This includes time series data for prices and related market indicators
        tea_price_data = get_tea_price_over_time()

        # Get the average price data for all categories
        # This processes the raw dataset to calculate category-wise averages
        file_path = 'Data/TeaCast Dataset.csv'
        all_average_price_data = get_all_average_prices(file_path)

        # Combine both tea price data and all category average price data
        # This creates a comprehensive response for the dashboard
        response_data = {
            'tea_price_data': tea_price_data,
            'all_average_price_data': all_average_price_data
        }

        # Return the combined data as JSON
        return jsonify(response_data)

    except Exception as e:
        # Log and handle any errors during data processing or retrieval
        logging.error(f'Error occurred while fetching tea price time series: {e}')
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500
