import tensorflow as tf
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_model(model_type, model_path):
    """
    General function to load a model or scaler.

    :param model_type: The type of model (e.g., 'SARIMAX', 'BLSTM', 'scaler_X', 'scaler_y').
    :param model_path: Path to the model/scaler file.
    :return: The loaded model/scaler.
    """
    try:
        if model_type == 'SARIMAX' or model_type == 'scaler_X' or model_type == 'scaler_y':
            return joblib.load(model_path)
        elif model_type == 'BLSTM':
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logging.error(f"Error loading {model_type} model/scaler from {model_path}: {e}")
        raise

def load_sarimax_model(model_path='Model/SARIMAX_Model.pkl'):
    """Load SARIMAX model."""
    return load_model('SARIMAX', model_path)

def load_blstm_model(model_path='Model/BLSTM_Model_3.0_v2.keras'):
    """Load BLSTM model."""
    return load_model('BLSTM', model_path)

def load_X_scaler(scaler_path='Model/scaler_X.pkl'):
    """Load X scaler."""
    return load_model('scaler_X', scaler_path)

def load_y_scaler(scaler_path='Model/scaler_y.pkl'):
    """Load y scaler."""
    return load_model('scaler_y', scaler_path)

def load_all_models():
    """Load all models and scalers."""
    try:
        logging.debug("Loading all models and scalers")

        # Load each model and scaler using the utility function
        sarimax_model = load_sarimax_model()
        blstm_model = load_blstm_model()
        X_scaler = load_X_scaler()
        y_scaler = load_y_scaler()

        return sarimax_model, blstm_model, X_scaler, y_scaler
    except Exception as e:
        logging.error(f"Error loading all models or scalers: {e}")
        raise
