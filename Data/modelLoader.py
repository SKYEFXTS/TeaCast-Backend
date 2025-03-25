import tensorflow as tf
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Function to load the SARIMAX model
def load_sarimax_model(model_path='Model/SARIMAX_Model.pkl'):
    """Load the SARIMAX model."""
    sarimax_model = joblib.load(model_path)
    return sarimax_model

# Function to load the BLSTM model
def load_blstm_model(model_path='Model/BLSTM_Model_3.0_v2.keras'):
    """Load the BLSTM model."""
    blstm_model = tf.keras.models.load_model(model_path)
    return blstm_model

# Function to load the X scaler
def load_X_scaler(scaler_path='Model/scaler_X.pkl'):
    """Load the X scaler."""
    X_scaler = joblib.load(scaler_path)
    return X_scaler

# Function to load the y scaler
def load_y_scaler(scaler_path='Model/scaler_y.pkl'):
    """Load the y scaler."""
    y_scaler = joblib.load(scaler_path)
    return y_scaler

# Main function to load all components
def load_all_models():
    try:
        logging.debug("Loading SARIMAX model")
        sarimax_model = load_sarimax_model()
        logging.debug("SARIMAX model loaded")

        logging.debug("Loading BLSTM model")
        blstm_model = load_blstm_model()
        logging.debug("BLSTM model loaded")

        logging.debug("Loading X_scaler")
        X_scaler = load_X_scaler()
        logging.debug("X_scaler loaded")

        logging.debug("Loading y_scaler")
        y_scaler = load_y_scaler()
        logging.debug("y_scaler loaded")

        return sarimax_model, blstm_model, X_scaler, y_scaler
    except Exception as e:
        logging.error(f"Error loading models or scalers: {e}")
        raise