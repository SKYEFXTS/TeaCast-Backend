from flask import Blueprint, jsonify
from Service.teaDashboardService import get_tea_price_over_time, get_all_average_prices
import logging

# Create Blueprint for the dashboard route
tea_dashboard_blueprint = Blueprint('dashboard', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@tea_dashboard_blueprint.route('/dashboard', methods=['GET'])
def tea_price_time_series():
    """Handles GET request to fetch historical data for tea prices, USD buying rate, crude oil price,
       and average prices for all categories."""
    try:
        # Get the historical tea price data over time
        tea_price_data = get_tea_price_over_time()

        # Get the average price data for all categories
        file_path = 'Data/TeaCast Dataset.csv'
        all_average_price_data = get_all_average_prices(file_path)

        # Combine both tea price data and all category average price data
        response_data = {
            'tea_price_data': tea_price_data,
            'all_average_price_data': all_average_price_data
        }

        # Return the combined data as JSON
        return jsonify(response_data)

    except Exception as e:
        logging.error(f'Error occurred while fetching tea price time series: {e}')
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500
