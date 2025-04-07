"""
TeaCast API - Main Application Entry Point
This module initializes and configures the Flask application, sets up CORS,
and registers all the blueprints for different API endpoints.
"""

import subprocess
import sys
import logging
from typing import Dict, List, Set, Optional
from flask import Flask
from flask_cors import CORS

# Import blueprints from different controllers
from Controller.authController import auth_blueprint
from Controller.predictionController import prediction_blueprint
from Controller.teaAuctionPriceController import tea_auction_price_blueprint
from Controller.teaDashboardController import tea_dashboard_blueprint

# Configure logging with debug level for development
logging.basicConfig(level=logging.DEBUG)

def install_requirements() -> None:
    """Installs the dependencies listed in the 'requirements.txt' file.
    
    This function is typically used during development or deployment setup.
    In production environments, dependencies should be installed during the
    build/deployment process instead of at runtime.
    
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

def create_app(install_deps: bool = True) -> Flask:
    """Creates and configures the Flask application.
    
    This factory function creates the Flask app, configures CORS settings,
    and registers all blueprints. Having the app creation in a factory function
    makes it easier to test and to create multiple instances if needed.
    
    Args:
        install_deps: Whether to install dependencies (default: True)
            Should be set to False in production environments
            
    Returns:
        Flask: Configured Flask application
    """
    # Install dependencies if requested (development only)
    if install_deps:
        install_requirements()
        
    # Initialize Flask app
    flask_app = Flask(__name__)
    
    # Configure CORS - restrict allowed origins to specific routes
    # In production, these origins should come from environment variables
    CORS(flask_app, resources={
        r"/auth/*": {"origins": "http://localhost:3000"},
        r"/data/*": {"origins": "http://localhost:3000"}
    })
    
    # Register all blueprints
    register_blueprints(flask_app)
    
    return flask_app

def register_blueprints(app: Flask) -> None:
    """Registers all blueprints with the Flask application.
    
    Args:
        app: Flask application instance
    """
    # Register the authentication and prediction routes
    # Each blueprint handles a specific set of related endpoints
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(prediction_blueprint, url_prefix='/data')
    app.register_blueprint(tea_auction_price_blueprint, url_prefix='/data')
    app.register_blueprint(tea_dashboard_blueprint, url_prefix='/data')
    
    logging.debug("All blueprints registered successfully")

# Create the Flask app
# In production, set install_deps=False
app = create_app(install_deps=True)

# Run the Flask app
if __name__ == '__main__':
    try:
        logging.debug("Starting Flask app...")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error occurred while starting Flask app: {e}")
        raise