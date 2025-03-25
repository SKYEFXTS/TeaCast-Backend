import tensorflow as tf
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

def load_sarimax_model(model_path='Model/SARIMAX_Model.pkl'):
    try:
        return joblib.load(model_path)
    except Exception as e:
        logging.error(f"Error loading SARIMAX model: {e}")
        raise

def load_blstm_model(model_path='Model/BLSTM_Model_3.0_v2.keras'):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        logging.error(f"Error loading BLSTM model: {e}")
        raise

def load_X_scaler(scaler_path='Model/scaler_X.pkl'):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        logging.error(f"Error loading X scaler: {e}")
        raise

def load_y_scaler(scaler_path='Model/scaler_y.pkl'):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        logging.error(f"Error loading y scaler: {e}")
        raise

def load_all_models():
    try:
        logging.debug("Loading all models and scalers")
        sarimax_model = load_sarimax_model()
        blstm_model = load_blstm_model()
        X_scaler = load_X_scaler()
        y_scaler = load_y_scaler()
        return sarimax_model, blstm_model, X_scaler, y_scaler
    except Exception as e:
        logging.error(f"Error loading models or scalers: {e}")
        raise
