import subprocess
import sys
import logging
from flask import Flask
from Controller.predictionController import prediction_blueprint

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

# Register the prediction route
app.register_blueprint(prediction_blueprint, url_prefix='/predict')

# Run the Flask app
if __name__ == '__main__':
    try:
        logging.debug("Starting Flask app...")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error occurred while starting Flask app: {e}")
        raise
