import subprocess
import sys
import logging
from flask import Flask

from Controller.authController import auth_blueprint
from Controller.predictionController import prediction_blueprint
from Controller.teaAuctionPriceController import tea_auction_price_blueprint
from Controller.teaDashboardController import tea_dashboard_blueprint

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def install_requirements():
    """
    Installs the dependencies listed in the 'requirements.txt' file.
    """
    try:
        logging.debug("Installing dependencies from requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        raise

# Install dependencies only if necessary (uncomment for development environments)
# install_requirements()

# Initialize Flask app
app = Flask(__name__)

# Register the authentication and prediction routes
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
