"""
Prediction Service Module
This module handles the tea price prediction logic using a hybrid approach
combining SARIMAX and BLSTM models. It processes input data, generates predictions,
and combines results from both models for final price forecasting.
"""

from Data.modelLoader import load_sarimax_model, load_blstm_model, load_y_scaler, load_all_models
from Data.dataPreProcessor import scale_input, inverse_scale_output, prepare_data_for_blstm
from Data.datasetLoader import load_and_filter_data
import numpy as np
import pandas as pd
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

def get_prediction():
    """
    Gets the final prediction by combining SARIMAX and BLSTM model outputs.
    
    Returns:
        list: List of dictionaries containing auction numbers and final predictions
              for the next 10 auctions.
              
    Raises:
        Exception: If there's an error in model loading, data processing, or prediction
    """
    try:
        # Load all required models and data scalers
        sarimax_model, blstm_model, X_scaler, y_scaler = load_all_models()
        logging.debug("Models and scalers loaded")

        # Load and filter data for specific tea grade and category
        dataset_df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", sarimax_model)
        logging.debug(f"Data loaded and filtered: {dataset_df.head()}")

        # Generate predictions using SARIMAX model
        dataset_df_sarimax = get_sarimax_prediction(dataset_df, 10, sarimax_model)
        sarimax_predictions = extract_sarimax_predictions(dataset_df_sarimax)

        # Prepare data for BLSTM model
        X_test, final_prediction_df = prepare_data_for_blstm(dataset_df, dataset_df_sarimax, 10, 10, X_scaler)
        logging.debug(f"Data prepared for BLSTM: {X_test.shape}")

        # Get residual predictions from BLSTM model
        predicted_residuals = get_blstm_residual_prediction(X_test, blstm_model, y_scaler)

        # Flatten predicted residuals for easier processing
        predicted_residuals = predicted_residuals.flatten()

        # Validate prediction lengths
        if len(predicted_residuals) != len(sarimax_predictions):
            logging.error("Length mismatch: Residuals and SARIMAX predictions.")
            return None

        # Combine predictions from both models
        final_prediction = sarimax_predictions + predicted_residuals

        # Round predictions to whole numbers
        final_prediction = np.round(final_prediction).astype(int)

        # Update DataFrame with predictions
        final_prediction_df["Predicted_Residuals"] = np.nan
        final_prediction_df["Final_Prediction"] = np.nan

        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Predicted_Residuals")] = predicted_residuals
        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Final_Prediction")] = final_prediction

        # Save predictions to CSV for persistence
        final_prediction_df.to_csv("Data/Final_Prediction.csv", index=False)

        # Log the final predictions
        logging.debug(f"Final prediction DataFrame after prediction:\n{final_prediction_df.tail()}")

        # Return formatted predictions
        return final_prediction_df[["Auction_Number", "Final_Prediction"]].tail(10).to_dict(orient='records')

    except Exception as e:
        logging.error(f"Error in getting prediction: {e}")
        raise

def extract_sarimax_predictions(dataset_df_sarimax):
    """
    Extracts SARIMAX predictions from the model output.
    
    Args:
        dataset_df_sarimax: Either a pandas Series or DataFrame containing SARIMAX predictions
        
    Returns:
        numpy.ndarray: Array of the last 10 SARIMAX predictions
    """
    if isinstance(dataset_df_sarimax, pd.Series):
        return dataset_df_sarimax.values  # Extract values if it's a Series
    else:
        return dataset_df_sarimax["SARIMAX_Predicted"].iloc[-10:].values  # Extract last 10 values if it's a DataFrame

def get_sarimax_prediction(data_df, forecast_auctions, sarimax_model):
    """
    Generates SARIMAX predictions using the provided model and exogenous data.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing required features:
            - USD_Buying
            - Crude_Oil_Price_LKR
            - Week
            - Auction_Number
        forecast_auctions (int): Number of auctions to forecast ahead
        sarimax_model: Pre-loaded SARIMAX model
        
    Returns:
        pd.Series or pd.DataFrame: Forecasted SARIMAX predicted values
    """
    logging.debug("Generating SARIMAX prediction")

    # Prepare exogenous features for SARIMAX
    exog_features = data_df[["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number"]]
    exog_forecast = exog_features.iloc[-forecast_auctions:]

    # Generate SARIMAX forecast
    sarimax_forecast = sarimax_model.forecast(steps=forecast_auctions, exog=exog_forecast)
    logging.debug(f"SARIMAX forecast: {sarimax_forecast}")

    return sarimax_forecast

def get_blstm_residual_prediction(X_test, blstm_model, y_scaler):
    """
    Predicts residuals using the BLSTM model.
    
    Args:
        X_test: Input data for BLSTM (sequences of past auctions and SARIMAX predictions)
        blstm_model: Pre-trained BLSTM model
        y_scaler: Scaler used for the residuals (target)
        
    Returns:
        numpy.ndarray: Predicted residuals after inverse scaling
    """
    logging.debug("Predicting BLSTM residuals")
    predicted_residuals_scaled = blstm_model.predict(X_test)
    predicted_residuals = y_scaler.inverse_transform(predicted_residuals_scaled)
    logging.debug(f"Predicted BLSTM residuals: {predicted_residuals}")
    return predicted_residuals
