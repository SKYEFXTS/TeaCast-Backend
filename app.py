import subprocess
import sys
from flask import Flask
from Controller.predictionController import prediction_blueprint

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Install dependencies
install_requirements()

# Initialize Flask app
app = Flask(__name__)

# Register the prediction route
app.register_blueprint(prediction_blueprint, url_prefix='/predict')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)