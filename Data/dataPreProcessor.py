"""
Data Preprocessor Module
This module handles the preprocessing of data for machine learning models.
It provides functionality for scaling input features, inverse scaling predictions,
and preparing sequential data for the BLSTM model.
"""

from typing import Tuple, Optional, List, Dict, Union, Any
import numpy as np
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler

from Data.modelLoader import load_X_scaler, load_y_scaler, setup_logging
from Data.datasetSaver import save_dataframe_as_csv

# Create a custom logger for this module
logger = logging.getLogger(__name__)

def scale_input(data_df: pd.DataFrame) -> np.ndarray:
    """Scales the input features using the pre-loaded X scaler.
    
    Args:
        data_df: DataFrame containing input features:
            - USD_Buying
            - Crude_Oil_Price_LKR
            - Week
            - Auction_Number
            - SARIMAX_Predicted
            
    Returns:
        np.ndarray: Scaled input features
        
    Raises:
        Exception: If there's an error in scaling the data
    """
    try:
        logger.info("Scaling input features for prediction")
        
        # Load the X scaler
        X_scaler = load_X_scaler()
        logger.debug(f"Input data shape before scaling: {data_df.shape}")

        # Prepare the input data
        input_data = np.array([data_df['USD_Buying'], data_df['Crude_Oil_Price_LKR'],
                               data_df['Week'], data_df['Auction_Number'], data_df['SARIMAX_Predicted']])

        # Reshape to match scaler input shape (1 row, multiple columns)
        input_data = X_scaler.transform(input_data.reshape(1, -1))
        logger.debug(f"Input data shape after scaling: {input_data.shape}")

        return input_data
    except Exception as e:
        logger.error(f"Error in scaling input data: {e}", exc_info=True)
        raise

def inverse_scale_output(prediction: np.ndarray) -> np.ndarray:
    """Inverse scales the prediction output using the pre-loaded y scaler.
    
    Args:
        prediction: Scaled prediction values
        
    Returns:
        np.ndarray: Inverse scaled prediction values
        
    Raises:
        Exception: If there's an error in inverse scaling
    """
    try:
        logger.info("Inverse scaling prediction output")
        
        # Load the y scaler
        y_scaler = load_y_scaler()

        # Reshape prediction for inverse transformation
        prediction_reshaped = prediction.reshape(-1, 1)
        logger.debug(f"Prediction shape before inverse scaling: {prediction_reshaped.shape}")

        # Inverse scale
        result = y_scaler.inverse_transform(prediction_reshaped).flatten()
        logger.debug(f"Prediction shape after inverse scaling: {result.shape}")
        logger.debug(f"Inverse scaled prediction values: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Error in inverse scaling output: {e}", exc_info=True)
        raise

def prepare_data_for_blstm(
    original_df: pd.DataFrame, 
    sarimax_predictions: Union[pd.Series, np.ndarray], 
    forecast_auctions: int = 10, 
    seq_length: int = 10, 
    X_scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepares the data for BLSTM by creating sequences of past auctions' features
    along with future SARIMAX predictions.
    
    This function performs several key steps:
    1. Extracts last known economic indicators
    2. Creates future data points with SARIMAX predictions
    3. Combines historical and future data
    4. Scales the combined data
    5. Creates sequences for BLSTM input
    
    Args:
        original_df: Original dataset with historical data
        sarimax_predictions: SARIMAX model predictions for future auctions
        forecast_auctions: Number of future auctions to forecast (default: 10)
        seq_length: Length of sequences for BLSTM (default: 10)
        X_scaler: Pre-loaded scaler for input features
        
    Returns:
        Tuple containing:
            np.ndarray: Last 10 sequences of scaled data for BLSTM
            pd.DataFrame: DataFrame with original and future data
            
    Raises:
        Exception: If there's an error in data preparation
        ValueError: If X_scaler is not provided
    """
    features = ["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number", "SARIMAX_Predicted"]

    try:
        logger.info(f"Preparing data for BLSTM model with forecast horizon: {forecast_auctions} auctions")
        
        # Prepare historical data and extract last known values
        processed_df = _prepare_historical_data(original_df)
        last_data = _extract_last_known_data(processed_df)
        last_known_data, last_week, last_auction_number = last_data
        
        # Convert SARIMAX predictions to numpy array if needed
        sarimax_predictions_array = _convert_predictions_to_array(sarimax_predictions, forecast_auctions)
        
        # Create future data points and combine with historical data
        extended_data = _create_extended_dataset(
            processed_df, 
            sarimax_predictions_array, 
            last_known_data, 
            last_week, 
            last_auction_number, 
            forecast_auctions
        )
        
        # Scale data and create sequences if scaler is provided
        if X_scaler:
            logger.info("Using provided scaler to scale and create sequences")
            return _scale_and_create_sequences(extended_data, features, X_scaler, seq_length)
        else:
            error_msg = "X_scaler is required for scaling data."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Error in preparing data for BLSTM: {e}", exc_info=True)
        raise

def _prepare_historical_data(original_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the historical data by removing unnecessary columns.
    
    Args:
        original_df: Original DataFrame with historical data
        
    Returns:
        pd.DataFrame: Processed DataFrame without the Price column
    """
    # Make a copy to avoid modifying the original DataFrame
    processed_df = original_df.copy()
    
    # Drop the 'Price' column because it's not needed for the prediction
    if 'Price' in processed_df.columns:
        processed_df.drop("Price", axis=1, inplace=True)
        
    return processed_df

def _extract_last_known_data(df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
    """Extracts the last known economic data, week, and auction number.
    
    Args:
        df: DataFrame containing the historical data
        
    Returns:
        Tuple containing:
            np.ndarray: Last known economic indicators (USD buying rate and crude oil price)
            int: Last week number
            int: Last auction number
    """
    # Get the last known data for economic indicators
    last_known_data = df.iloc[-1][["USD_Buying", "Crude_Oil_Price_LKR"]].values
    last_week = df["Week"].iloc[-1]
    last_auction_number = df["Auction_Number"].iloc[-1]

    logger.debug(
        f"Last known data: USD_Buying: {last_known_data[0]}, "
        f"Crude_Oil_Price_LKR: {last_known_data[1]}, "
        f"Week: {last_week}, Auction_Number: {last_auction_number}"
    )
    
    return last_known_data, last_week, last_auction_number

def _convert_predictions_to_array(
    sarimax_predictions: Union[pd.Series, np.ndarray], 
    forecast_auctions: int
) -> np.ndarray:
    """Converts SARIMAX predictions to a numpy array.
    
    Args:
        sarimax_predictions: SARIMAX predictions as Series or array
        forecast_auctions: Number of auctions to forecast
        
    Returns:
        np.ndarray: SARIMAX predictions as numpy array
    """
    # Ensure that sarimax_predictions is a numpy array
    if isinstance(sarimax_predictions, pd.Series):
        sarimax_predictions = sarimax_predictions.values

    logger.debug(f"SARIMAX predictions received: {sarimax_predictions[:forecast_auctions]}")
    return sarimax_predictions

def _create_extended_dataset(
    original_df: pd.DataFrame, 
    sarimax_predictions: np.ndarray,
    last_known_data: np.ndarray,
    last_week: int,
    last_auction_number: int,
    forecast_auctions: int
) -> pd.DataFrame:
    """Creates an extended dataset with future data points.
    
    Args:
        original_df: Original dataset with historical data
        sarimax_predictions: Array of SARIMAX predictions
        last_known_data: Last known economic indicators
        last_week: Last week number
        last_auction_number: Last auction number
        forecast_auctions: Number of future auctions to forecast
        
    Returns:
        pd.DataFrame: Extended dataset with historical and future data
    """
    # Prepare the data for the future auctions (forecasted values)
    future_data = []

    for i in range(forecast_auctions):
        # Increment auction number and week, reset to 1 if >= 50
        future_auction_number = (last_auction_number + i) % 50 + 1
        future_week = (last_week + i) % 50 + 1

        # Create a dictionary for future data
        future_data.append({
            "USD_Buying": last_known_data[0],
            "Crude_Oil_Price_LKR": last_known_data[1],
            "Week": future_week,
            "Auction_Number": future_auction_number,
            "SARIMAX_Predicted": sarimax_predictions[i]
        })

    # Convert future data into a DataFrame
    future_data_df = pd.DataFrame(future_data)

    # Concatenate the future data with the original dataset
    extended_data = pd.concat([original_df, future_data_df], ignore_index=True)

    # Fill missing data for economic indicators using the last known data
    extended_data["USD_Buying"].fillna(last_known_data[0], inplace=True)
    extended_data["Crude_Oil_Price_LKR"].fillna(last_known_data[1], inplace=True)

    # Save the extended data to a CSV file
    save_dataframe_as_csv(extended_data, "Data/PreProcessedData/extended_data.csv")
    
    return extended_data

def _scale_and_create_sequences(
    extended_data: pd.DataFrame, 
    features: List[str], 
    X_scaler: StandardScaler, 
    seq_length: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Scales the data and creates sequences for BLSTM input.
    
    Args:
        extended_data: Extended dataset with historical and future data
        features: List of feature names to use
        X_scaler: Scaler for input features
        seq_length: Length of sequences for BLSTM
        
    Returns:
        Tuple containing:
            np.ndarray: Last 10 sequences of scaled data for BLSTM
            pd.DataFrame: Extended dataset
            
    Raises:
        ValueError: If there are NaN values in the data
    """
    # Ensure no NaN values are left before scaling
    if extended_data[features].isnull().any().any():
        logger.warning("There are still NaN values in the data before scaling.")
        # Consider raising an exception here if NaN values are critical

    # Scale the data using X_scaler
    scaled_data = X_scaler.transform(extended_data[features])

    # Save the scaled data to a CSV file
    save_dataframe_as_csv(
        pd.DataFrame(scaled_data, columns=features), 
        "Data/PreProcessedData/scaled_data.csv"
    )

    # Create sequences of data for BLSTM
    X = np.array([scaled_data[i:i + seq_length] for i in range(len(scaled_data) - seq_length)])

    # Return the last 10 sequences
    X_last_10 = X[-10:]

    return X_last_10, extended_data
