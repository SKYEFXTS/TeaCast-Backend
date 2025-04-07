"""
Prediction Service Module
This module handles the tea price prediction logic using a hybrid approach
combining SARIMAX and BLSTM models. It processes input data, generates predictions,
and combines results from both models for final price forecasting.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import logging
import os
import time
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

from Data.modelLoader import load_sarimax_model, load_blstm_model, load_y_scaler, load_all_models
from Data.dataPreProcessor import scale_input, inverse_scale_output, prepare_data_for_blstm
from Data.datasetLoader import load_and_filter_data

# Create a custom logger for this module
logger = logging.getLogger(__name__)

# Constants for prediction configuration
FORECAST_HORIZON = 10  # Number of auctions to forecast
SEQUENCE_LENGTH = 10   # Length of sequences for BLSTM
TEA_GRADE = "WESTERN HIGH"
TEA_CATEGORY = "BOPF/BOPFSp"
OUTPUT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'Final_Prediction.csv'))

def get_prediction() -> List[Dict[str, Any]]:
    """Gets the final prediction by combining SARIMAX and BLSTM model outputs.
    
    This function orchestrates the entire prediction workflow including data loading,
    model preparation, generating predictions from both models, and combining them.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing auction numbers and 
            final predictions for the next 10 auctions.
              
    Raises:
        Exception: If there's an error in model loading, data processing, or prediction
    """
    try:
        start_time = time.time()
        logger.info("Starting tea price prediction process")
        
        # Load models and data
        logger.info("Loading models and data")
        models_data = load_models_and_data()
        sarimax_model, blstm_model, X_scaler, y_scaler, dataset_df = models_data
        
        # Generate SARIMAX predictions
        logger.info("Generating SARIMAX predictions")
        sarimax_results = generate_sarimax_predictions(dataset_df, sarimax_model)
        dataset_df_sarimax, sarimax_predictions = sarimax_results
        
        # Generate BLSTM predictions and combine with SARIMAX
        logger.info("Generating and combining BLSTM predictions with SARIMAX")
        combined_results = generate_and_combine_predictions(
            dataset_df, dataset_df_sarimax, blstm_model, X_scaler, y_scaler, sarimax_predictions
        )
        final_prediction_df, final_prediction = combined_results
        
        # Save and return the final predictions
        logger.info("Formatting and saving final predictions")
        result = format_and_save_predictions(final_prediction_df, final_prediction)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction process completed in {elapsed_time:.2f} seconds")
        
        return result
    except Exception as e:
        logger.error(f"Error in getting prediction: {e}", exc_info=True)
        raise

def load_models_and_data() -> Tuple[SARIMAXResultsWrapper, Model, StandardScaler, StandardScaler, pd.DataFrame]:
    """Loads all required models, scalers and dataset.
    
    Returns:
        Tuple containing:
            SARIMAXResultsWrapper: Loaded SARIMAX model
            Model: Loaded BLSTM model
            StandardScaler: X data scaler
            StandardScaler: y data scaler
            pd.DataFrame: Loaded and filtered dataset
    """
    try:
        # Load all required models and data scalers
        logger.info("Loading prediction models and scalers")
        sarimax_model, blstm_model, X_scaler, y_scaler = load_all_models()
        logger.debug("Models and scalers loaded successfully")

        # Load and filter data for specific tea grade and category
        logger.info(f"Loading and filtering dataset for {TEA_GRADE}, {TEA_CATEGORY}")
        dataset_df = load_and_filter_data(TEA_GRADE, TEA_CATEGORY, sarimax_model)
        logger.debug(f"Dataset loaded with {len(dataset_df)} records")
        
        if dataset_df.empty:
            logger.error("Loaded dataset is empty")
            raise ValueError("Dataset is empty after filtering")
            
        return sarimax_model, blstm_model, X_scaler, y_scaler, dataset_df
    except Exception as e:
        logger.error(f"Error loading models and data: {e}", exc_info=True)
        raise

def generate_sarimax_predictions(
    dataset_df: pd.DataFrame, 
    sarimax_model: SARIMAXResultsWrapper
) -> Tuple[Union[pd.Series, pd.DataFrame], np.ndarray]:
    """Generates predictions using the SARIMAX model.
    
    Args:
        dataset_df: DataFrame containing the tea price data
        sarimax_model: Trained SARIMAX model
        
    Returns:
        Tuple containing:
            Union[pd.Series, pd.DataFrame]: SARIMAX model outputs
            np.ndarray: Extracted SARIMAX predictions as numpy array
    """
    try:
        # Generate predictions using SARIMAX model (10 is the default forecast horizon)
        logger.info("Running SARIMAX model to generate forecasts")
        dataset_df_sarimax = get_sarimax_prediction(dataset_df, 10, sarimax_model)
        
        logger.info("Extracting SARIMAX predictions")
        sarimax_predictions = extract_sarimax_predictions(dataset_df_sarimax)
        
        logger.debug(f"SARIMAX predictions summary: min={np.min(sarimax_predictions)}, max={np.max(sarimax_predictions)}, mean={np.mean(sarimax_predictions)}")
        
        return dataset_df_sarimax, sarimax_predictions
    except Exception as e:
        logger.error(f"Error generating SARIMAX predictions: {e}", exc_info=True)
        raise

def generate_and_combine_predictions(
    dataset_df: pd.DataFrame,
    dataset_df_sarimax: Union[pd.Series, pd.DataFrame],
    blstm_model: Model,
    X_scaler: StandardScaler,
    y_scaler: StandardScaler,
    sarimax_predictions: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generates BLSTM residual predictions and combines them with SARIMAX predictions.
    
    Args:
        dataset_df: Original dataset DataFrame
        dataset_df_sarimax: SARIMAX predictions
        blstm_model: Trained BLSTM model
        X_scaler: Scaler for input features
        y_scaler: Scaler for output targets
        sarimax_predictions: Array of SARIMAX predictions
        
    Returns:
        Tuple containing:
            pd.DataFrame: DataFrame with all prediction data
            np.ndarray: Final combined predictions
            
    Raises:
        ValueError: If there's a length mismatch between residuals and SARIMAX predictions
    """
    try:
        # Prepare data for BLSTM model
        logger.info("Preparing data sequences for BLSTM model")
        X_test, final_prediction_df = prepare_data_for_blstm(dataset_df, dataset_df_sarimax, 10, 10, X_scaler)
        logger.debug(f"Prepared BLSTM input shape: {X_test.shape}")

        # Get residual predictions from BLSTM model
        logger.info("Generating residual predictions with BLSTM model")
        predicted_residuals = get_blstm_residual_prediction(X_test, blstm_model, y_scaler)

        # Flatten predicted residuals for easier processing
        predicted_residuals = predicted_residuals.flatten()
        logger.debug(f"BLSTM residuals summary: min={np.min(predicted_residuals)}, max={np.max(predicted_residuals)}, mean={np.mean(predicted_residuals)}")

        # Validate prediction lengths
        if len(predicted_residuals) != len(sarimax_predictions):
            error_msg = f"Length mismatch: Residuals ({len(predicted_residuals)}) and SARIMAX predictions ({len(sarimax_predictions)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Combine predictions from both models
        logger.info("Combining SARIMAX and BLSTM predictions")
        final_prediction = sarimax_predictions + predicted_residuals

        # Round predictions to whole numbers
        final_prediction = np.round(final_prediction).astype(int)
        logger.debug(f"Final prediction summary: min={np.min(final_prediction)}, max={np.max(final_prediction)}, mean={np.mean(final_prediction)}")

        # Update DataFrame with predictions
        final_prediction_df["Predicted_Residuals"] = np.nan
        final_prediction_df["Final_Prediction"] = np.nan

        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Predicted_Residuals")] = predicted_residuals
        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Final_Prediction")] = final_prediction

        return final_prediction_df, final_prediction
    except Exception as e:
        logger.error(f"Error in BLSTM prediction and combination: {e}", exc_info=True)
        raise

def format_and_save_predictions(
    final_prediction_df: pd.DataFrame, 
    final_prediction: np.ndarray
) -> List[Dict[str, Any]]:
    """Formats and saves the final predictions to a CSV file.
    
    Args:
        final_prediction_df: DataFrame containing all prediction data
        final_prediction: Array of final predictions
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries with auction numbers and predictions
    """
    # Save predictions to CSV for persistence
    final_prediction_df.to_csv(OUTPUT_CSV_PATH, index=False)

    # Log the final predictions
    logger.debug(f"Final prediction DataFrame after prediction:\n{final_prediction_df.tail()}")

    # Return formatted predictions
    return final_prediction_df[["Auction_Number", "Final_Prediction"]].tail(10).to_dict(orient='records')

def extract_sarimax_predictions(dataset_df_sarimax: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
    """Extracts SARIMAX predictions from the model output.
    
    Args:
        dataset_df_sarimax: Either a pandas Series or DataFrame containing SARIMAX predictions
        
    Returns:
        np.ndarray: Array of the last 10 SARIMAX predictions
    """
    if isinstance(dataset_df_sarimax, pd.Series):
        return dataset_df_sarimax.values  # Extract values if it's a Series
    else:
        return dataset_df_sarimax["SARIMAX_Predicted"].iloc[-10:].values  # Extract last 10 values if it's a DataFrame

def get_sarimax_prediction(
    data_df: pd.DataFrame, 
    forecast_auctions: int, 
    sarimax_model: SARIMAXResultsWrapper
) -> Union[pd.Series, pd.DataFrame]:
    """Generates SARIMAX predictions using the provided model and exogenous data.
    
    Args:
        data_df: DataFrame containing required features:
            - USD_Buying
            - Crude_Oil_Price_LKR
            - Week
            - Auction_Number
        forecast_auctions: Number of auctions to forecast ahead
        sarimax_model: Pre-loaded SARIMAX model
        
    Returns:
        Union[pd.Series, pd.DataFrame]: Forecasted SARIMAX predicted values
    """
    logger.debug("Generating SARIMAX prediction")

    # Prepare exogenous features for SARIMAX
    exog_features = data_df[["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number"]]
    exog_forecast = exog_features.iloc[-forecast_auctions:]

    # Generate SARIMAX forecast
    sarimax_forecast = sarimax_model.forecast(steps=forecast_auctions, exog=exog_forecast)
    logger.debug(f"SARIMAX forecast: {sarimax_forecast}")

    return sarimax_forecast

def get_blstm_residual_prediction(
    X_test: np.ndarray, 
    blstm_model: Model, 
    y_scaler: StandardScaler
) -> np.ndarray:
    """Predicts residuals using the BLSTM model.
    
    Args:
        X_test: Input data for BLSTM (sequences of past auctions and SARIMAX predictions)
        blstm_model: Pre-trained BLSTM model
        y_scaler: Scaler used for the residuals (target)
        
    Returns:
        np.ndarray: Predicted residuals after inverse scaling
    """
    logger.debug("Predicting BLSTM residuals")
    predicted_residuals_scaled = blstm_model.predict(X_test)
    predicted_residuals = y_scaler.inverse_transform(predicted_residuals_scaled)
    logger.debug(f"Predicted BLSTM residuals: {predicted_residuals}")
    return predicted_residuals
