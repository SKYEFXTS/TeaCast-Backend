"""
Model Loader Module
This module handles the loading of machine learning models and data scalers.
It provides functionality to load SARIMAX and BLSTM models, as well as their associated
feature and target scalers.
"""

from typing import Any, Tuple, Union, Optional
import tensorflow as tf
from tensorflow.keras.models import Model
import joblib
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from sklearn.preprocessing import StandardScaler

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

def load_model(model_type: str, model_path: str) -> Any:
    """General function to load a model or scaler.
    
    Args:
        model_type: The type of model to load:
            - 'SARIMAX': For SARIMAX time series model
            - 'BLSTM': For Bidirectional LSTM model
            - 'scaler_X': For input feature scaler
            - 'scaler_y': For target value scaler
        model_path: Path to the model/scaler file
        
    Returns:
        Any: The loaded model or scaler object
        
    Raises:
        ValueError: If model_type is not recognized
        Exception: If there's an error loading the model/scaler
    """
    try:
        if model_type in ('SARIMAX', 'scaler_X', 'scaler_y'):
            return joblib.load(model_path)
        elif model_type == 'BLSTM':
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logging.error(f"Error loading {model_type} model/scaler from {model_path}: {e}")
        raise

def load_sarimax_model(model_path: str = 'Model/SARIMAX_Model.pkl') -> SARIMAXResultsWrapper:
    """Load the SARIMAX time series model.
    
    Args:
        model_path: Path to the SARIMAX model file (default: 'Model/SARIMAX_Model.pkl')
        
    Returns:
        SARIMAXResultsWrapper: The loaded SARIMAX model
        
    Raises:
        Exception: If there's an error loading the model
    """
    return load_model('SARIMAX', model_path)

def load_blstm_model(model_path: str = 'Model/BLSTM_Model_3.0_v2.keras') -> Model:
    """Load the Bidirectional LSTM model.
    
    Args:
        model_path: Path to the BLSTM model file (default: 'Model/BLSTM_Model_3.0_v2.keras')
        
    Returns:
        Model: The loaded BLSTM model
        
    Raises:
        Exception: If there's an error loading the model
    """
    return load_model('BLSTM', model_path)

def load_X_scaler(scaler_path: str = 'Model/scaler_X.pkl') -> StandardScaler:
    """Load the input feature scaler.
    
    Args:
        scaler_path: Path to the X scaler file (default: 'Model/scaler_X.pkl')
        
    Returns:
        StandardScaler: The loaded feature scaler
        
    Raises:
        Exception: If there's an error loading the scaler
    """
    return load_model('scaler_X', scaler_path)

def load_y_scaler(scaler_path: str = 'Model/scaler_y.pkl') -> StandardScaler:
    """Load the target value scaler.
    
    Args:
        scaler_path: Path to the y scaler file (default: 'Model/scaler_y.pkl')
        
    Returns:
        StandardScaler: The loaded target scaler
        
    Raises:
        Exception: If there's an error loading the scaler
    """
    return load_model('scaler_y', scaler_path)

def load_all_models() -> Tuple[SARIMAXResultsWrapper, Model, StandardScaler, StandardScaler]:
    """Load all models and scalers required for prediction.
    
    This function loads all required models and scalers in a single call for convenience.
    It's typically used when all components are needed together for making predictions.
    
    Returns:
        Tuple containing:
            SARIMAXResultsWrapper: The loaded SARIMAX model
            Model: The loaded BLSTM model
            StandardScaler: The loaded input feature scaler
            StandardScaler: The loaded target value scaler
            
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
