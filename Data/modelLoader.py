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
import os
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from sklearn.preprocessing import StandardScaler

# Create a custom logger for this module
logger = logging.getLogger(__name__)

# Constants for model paths - centralized for easier management
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
SARIMAX_MODEL_PATH = os.path.join(MODEL_DIR, 'SARIMAX_Model.pkl')
BLSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'BLSTM_Model_3.0_v2.keras')
X_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
Y_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_y.pkl')

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
        logger.info(f"Loading {model_type} from {model_path}")
        
        if model_type in ('SARIMAX', 'scaler_X', 'scaler_y'):
            return joblib.load(model_path)
        elif model_type == 'BLSTM':
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading {model_type} model/scaler from {model_path}: {str(e)}", exc_info=True)
        raise

def load_sarimax_model(model_path: str = SARIMAX_MODEL_PATH) -> SARIMAXResultsWrapper:
    """Load the SARIMAX time series model.
    
    Args:
        model_path: Path to the SARIMAX model file (default: uses the predefined constant)
        
    Returns:
        SARIMAXResultsWrapper: The loaded SARIMAX model
        
    Raises:
        Exception: If there's an error loading the model
    """
    return load_model('SARIMAX', model_path)

def load_blstm_model(model_path: str = BLSTM_MODEL_PATH) -> Model:
    """Load the Bidirectional LSTM model.
    
    Args:
        model_path: Path to the BLSTM model file (default: uses the predefined constant)
        
    Returns:
        Model: The loaded BLSTM model
        
    Raises:
        Exception: If there's an error loading the model
    """
    return load_model('BLSTM', model_path)

def load_X_scaler(scaler_path: str = X_SCALER_PATH) -> StandardScaler:
    """Load the input feature scaler.
    
    Args:
        scaler_path: Path to the X scaler file (default: uses the predefined constant)
        
    Returns:
        StandardScaler: The loaded feature scaler
        
    Raises:
        Exception: If there's an error loading the scaler
    """
    return load_model('scaler_X', scaler_path)

def load_y_scaler(scaler_path: str = Y_SCALER_PATH) -> StandardScaler:
    """Load the target value scaler.
    
    Args:
        scaler_path: Path to the y scaler file (default: uses the predefined constant)
        
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
        logger.info("Loading all models and scalers for prediction")

        # Load each model and scaler using the utility function
        sarimax_model = load_sarimax_model()
        blstm_model = load_blstm_model()
        X_scaler = load_X_scaler()
        y_scaler = load_y_scaler()

        logger.info("Successfully loaded all models and scalers")
        return sarimax_model, blstm_model, X_scaler, y_scaler
    except Exception as e:
        logger.error(f"Failed to load all models or scalers: {str(e)}", exc_info=True)
        raise
