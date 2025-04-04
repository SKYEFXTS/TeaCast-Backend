"""
TeaCast API - Main Application Entry Point
This module initializes and configures the Flask application, sets up CORS,
and registers all the blueprints for different API endpoints.
"""

import subprocess
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Import blueprints from different controllers
from Controller.authController import auth_blueprint
from Controller.predictionController import prediction_blueprint
from Controller.teaAuctionPriceController import tea_auction_price_blueprint
from Controller.teaDashboardController import tea_dashboard_blueprint

# Configure logging with debug level for development
logging.basicConfig(level=logging.DEBUG)

def install_requirements():
    """
    Installs the dependencies listed in the 'requirements.txt' file.
    This function is typically used during development or deployment setup.
    
    Raises:
        subprocess.CalledProcessError: If pip install fails
    """
    try:
        logging.debug("Installing dependencies from requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        raise

# Install dependencies only if necessary (uncomment for development environments)
install_requirements()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for only the '/auth' and '/data' route prefixes
# This allows cross-origin requests from the frontend development server
CORS(app, resources={
    r"/auth/*": {"origins": "http://localhost:3000"},
    r"/data/*": {"origins": "http://localhost:3000"}
})

# Register the authentication and prediction routes
# Each blueprint handles a specific set of related endpoints
app.register_blueprint(auth_blueprint, url_prefix='/auth')
app.register_blueprint(prediction_blueprint, url_prefix='/data')
app.register_blueprint(tea_auction_price_blueprint, url_prefix='/data')
app.register_blueprint(tea_dashboard_blueprint, url_prefix='/data')

# Run the Flask app
if __name__ == '__main__':
    try:
        logging.debug("Starting Flask app...")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error occurred while starting Flask app: {e}")
        raise