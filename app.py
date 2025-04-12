"""
TeaCast API - Main Application Entry Point
This module initializes and configures the Flask application, sets up CORS,
and registers all the blueprints for different API endpoints.
"""

import subprocess
import sys
import logging
import os
from typing import Dict, List, Set, Optional
from flask import Flask
from flask_cors import CORS

# Import blueprints from different controllers
from Controller.authController import auth_blueprint
from Controller.predictionController import prediction_blueprint
from Controller.teaAuctionPriceController import tea_auction_price_blueprint
from Controller.teaDashboardController import tea_dashboard_blueprint

# Import the setup_logging function from the dedicated logging config module
from Utilities.loggingConfig import setup_logging

# Environment configuration with defaults
class Config:
    """Configuration settings for the application.
    
    This class centralizes all configuration settings and loads them from
    environment variables when available, falling back to sensible defaults.
    """
    # Server settings
    DEBUG = os.getenv('TEACAST_DEBUG', 'False').lower() in ('true', '1', 't')
    PORT = int(os.getenv('PORT', os.getenv('TEACAST_PORT', '5001')))  # Use Render's PORT env var
    HOST = os.getenv('TEACAST_HOST', '0.0.0.0')
    
    # Logging settings
    LOG_LEVEL = os.getenv('TEACAST_LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('TEACAST_LOG_DIR', 'logs')
    LOG_FILE = os.path.join(LOG_DIR, os.getenv('TEACAST_LOG_FILE', 'teacast.log'))
    
    # CORS settings - allowing the deployed Netlify frontend
    CORS_ORIGINS = os.getenv('TEACAST_CORS_ORIGINS', 'https://teacast.netlify.app,http://localhost:3000').split(',')

# Create logs directory if it doesn't exist
if not os.path.exists(Config.LOG_DIR):
    os.makedirs(Config.LOG_DIR)

# Configure logging level based on environment setting
log_level_name = Config.LOG_LEVEL.upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Configure centralized logging for the entire application
setup_logging(log_level=log_level, log_file=Config.LOG_FILE)

# Get the root logger after it's been configured
logger = logging.getLogger(__name__)
logger.info(f"TeaCast API initializing with log level: {log_level_name}")

def install_requirements() -> None:
    """Installs the dependencies listed in the 'requirements.txt' file.
    
    This function is typically used during development or deployment setup.
    In production environments, dependencies should be installed during the
    build/deployment process instead of at runtime.
    
    Raises:
        subprocess.CalledProcessError: If pip install fails
    """
    try:
        logger.debug("Installing dependencies from requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise

def create_app(install_deps: bool = False) -> Flask:  # Default to False for production
    """Creates and configures the Flask application.
    
    This factory function creates the Flask app, configures CORS settings,
    and registers all blueprints. Having the app creation in a factory function
    makes it easier to test and to create multiple instances if needed.
    
    Args:
        install_deps: Whether to install dependencies (default: False)
            Should be set to False in production environments
            
    Returns:
        Flask: Configured Flask application
    """
    # Install dependencies if requested (development only)
    if install_deps:
        install_requirements()
        
    # Initialize Flask app
    flask_app = Flask(__name__)
    
    # Configure CORS - restrict allowed origins based on configuration
    CORS(flask_app, resources={
        r"/auth/*": {"origins": Config.CORS_ORIGINS},
        r"/data/*": {"origins": Config.CORS_ORIGINS}
    })
    
    # Register all blueprints
    register_blueprints(flask_app)
    
    # Log application startup
    logger.info("TeaCast API initialized successfully")
    
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
    
    logger.debug("All blueprints registered successfully")

# Create the Flask app
# In production, set install_deps=False
app = create_app(install_deps=False)  # Set to False for production deployment

# Run the Flask app
if __name__ == '__main__':
    try:
        logger.info("Starting TeaCast API server...")
        app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
    except Exception as e:
        logger.error(f"Error occurred while starting Flask app: {e}", exc_info=True)
        raise