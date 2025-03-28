"""
Model Loader Module
This module handles the loading of machine learning models and data scalers.
It provides functionality to load SARIMAX and BLSTM models, as well as their associated
feature and target scalers.
"""

import tensorflow as tf
import joblib
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

def load_model(model_type, model_path):
    """
    General function to load a model or scaler.
    
    Args:
        model_type (str): The type of model to load:
            - 'SARIMAX': For SARIMAX time series model
            - 'BLSTM': For Bidirectional LSTM model
            - 'scaler_X': For input feature scaler
            - 'scaler_y': For target value scaler
        model_path (str): Path to the model/scaler file
        
    Returns:
        object: The loaded model or scaler
        
    Raises:
        ValueError: If model_type is not recognized
        Exception: If there's an error loading the model/scaler
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
    """
    Load the SARIMAX time series model.
    
    Args:
        model_path (str): Path to the SARIMAX model file (default: 'Model/SARIMAX_Model.pkl')
        
    Returns:
        object: The loaded SARIMAX model
    """
    return load_model('SARIMAX', model_path)

def load_blstm_model(model_path='Model/BLSTM_Model_3.0_v2.keras'):
    """
    Load the Bidirectional LSTM model.
    
    Args:
        model_path (str): Path to the BLSTM model file (default: 'Model/BLSTM_Model_3.0_v2.keras')
        
    Returns:
        tf.keras.Model: The loaded BLSTM model
    """
    return load_model('BLSTM', model_path)

def load_X_scaler(scaler_path='Model/scaler_X.pkl'):
    """
    Load the input feature scaler.
    
    Args:
        scaler_path (str): Path to the X scaler file (default: 'Model/scaler_X.pkl')
        
    Returns:
        object: The loaded feature scaler
    """
    return load_model('scaler_X', scaler_path)

def load_y_scaler(scaler_path='Model/scaler_y.pkl'):
    """
    Load the target value scaler.
    
    Args:
        scaler_path (str): Path to the y scaler file (default: 'Model/scaler_y.pkl')
        
    Returns:
        object: The loaded target scaler
    """
    return load_model('scaler_y', scaler_path)

def load_all_models():
    """
    Load all models and scalers required for prediction.
    
    Returns:
        tuple: (sarimax_model, blstm_model, X_scaler, y_scaler)
            - sarimax_model: The loaded SARIMAX model
            - blstm_model: The loaded BLSTM model
            - X_scaler: The loaded input feature scaler
            - y_scaler: The loaded target value scaler
            
    Raises:
        Exception: If there's an error loading any of the models or scalers
    """
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
