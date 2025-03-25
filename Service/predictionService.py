from Data.modelLoader import load_sarimax_model, load_blstm_model, load_y_scaler, load_all_models
from Data.dataPreProcessor import scale_input, inverse_scale_output, prepare_data_for_blstm
from Data.datasetLoader import load_and_filter_data
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def get_prediction():
    # Load models and scalers once
    logging.debug("Loading models and scalers")
    sarimax_model, blstm_model, X_scaler, y_scaler = load_all_models()
    logging.debug("Models and scalers loaded")

    # Load and filter data
    logging.debug("Loading and filtering data")
    dataset_df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", sarimax_model)
    logging.debug(f"Data loaded and filtered: {dataset_df.head()}")

    # Get SARIMAX prediction
    logging.debug("Getting SARIMAX prediction")
    dataset_df_sarimax = get_sarimax_prediction(dataset_df, 10, sarimax_model)

    # Ensure that sarimax_predictions is a numpy array (not a pandas Series)
    if isinstance(dataset_df_sarimax, pd.Series):
        # If it's a Series, we need to get the values as a numpy array
        logging.debug("dataset_df_sarimax is a Series, converting to numpy array")
        sarimax_predictions = dataset_df_sarimax.values  # Extract SARIMAX values as a numpy array
    else:
        # If it's a DataFrame with SARIMAX_Predicted column, get the last 10 values
        logging.debug("dataset_df_sarimax is a DataFrame")
        sarimax_predictions = dataset_df_sarimax["SARIMAX_Predicted"].iloc[-10:].values  # Last 10 SARIMAX predictions

    logging.debug(f"SARIMAX predictions received: {sarimax_predictions}")

    # Prepare data for BLSTM
    logging.debug("Preparing data for BLSTM")
    X_test, final_prediction_df = prepare_data_for_blstm(dataset_df, dataset_df_sarimax, 10, 10, X_scaler)
    logging.debug(f"Data prepared for BLSTM: {X_test.shape}")

    # Get BLSTM residual prediction
    logging.debug("Getting BLSTM residual prediction")
    predicted_residuals = get_blstm_residual_prediction(X_test, blstm_model, y_scaler)

    # Flatten predicted_residuals if needed (in case it's a 2D array)
    predicted_residuals = predicted_residuals.flatten()

    # Log the types of both predicted_residuals and sarimax_predictions
    logging.debug(f"Type of predicted_residuals: {type(predicted_residuals)}")
    logging.debug(f"Type of sarimax_predictions: {type(sarimax_predictions)}")

    # Ensure both predicted_residuals and SARIMAX_Predicted are numpy arrays of the same length
    if len(predicted_residuals) != len(sarimax_predictions):
        logging.error(f"Length mismatch: predicted_residuals ({len(predicted_residuals)}) and sarimax_predictions ({len(sarimax_predictions)})")
        return None  # Return None if there's a length mismatch

    # Calculate the final prediction by adding SARIMAX predictions and residuals
    final_prediction = sarimax_predictions + predicted_residuals

    # Initialize columns with NaN values
    final_prediction_df["Predicted_Residuals"] = np.nan
    final_prediction_df["Final_Prediction"] = np.nan

    # Log the initialized DataFrame
    logging.debug(f"Final prediction DataFrame after initialization:\n{final_prediction_df.tail()}")

    # Add predicted_residuals and final_prediction to the last 10 rows
    final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Predicted_Residuals")] = predicted_residuals
    final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Final_Prediction")] = final_prediction

    # Save the final prediction DataFrame to a CSV file
    final_prediction_df.to_csv("Data/Final_Prediction.csv", index=False)

    # Return final prediction (if you only want the predictions themselves)
    return final_prediction



def get_sarimax_prediction(data_df, forecast_auctions, sarimax_model):
    """
    This function generates a prediction using the SARIMAX model with exogenous features.

    :param data_df: DataFrame containing the required features such as 'USD_Buying', 'Crude_Oil_Price_LKR', etc.
    :param forecast_auctions: The number of auctions to forecast ahead.
    :param sarimax_model: The pre-loaded SARIMAX model.
    :return: An array containing the forecasted SARIMAX predicted values.
    """
    logging.debug("Generating SARIMAX prediction")

    # Prepare the exogenous features (external factors) from the data
    exog_features = data_df[["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number"]]
    exog_forecast = exog_features.iloc[-forecast_auctions:]  # Use the last known external factors for forecasting

    # Generate the SARIMAX forecast for the next 'forecast_auctions' auctions
    sarimax_forecast = sarimax_model.forecast(steps=forecast_auctions, exog=exog_forecast)
    logging.debug(f"SARIMAX forecast: {sarimax_forecast}")

    # Return only the forecasted SARIMAX values
    return sarimax_forecast

def get_blstm_residual_prediction(X_test, blstm_model, y_scaler):
    """
    Use the BLSTM model to predict the residuals for the next auctions based on the prepared data.

    :param X_test: The input data for BLSTM (sequences of past auctions and SARIMAX predictions).
    :param blstm_model: The pre-trained BLSTM model.
    :param y_scaler: The scaler used for the residuals (target).
    :return: The predicted residuals after inverse scaling.
    """
    logging.debug("Predicting BLSTM residuals")
    predicted_residuals_scaled = blstm_model.predict(X_test)
    predicted_residuals = y_scaler.inverse_transform(predicted_residuals_scaled)
    logging.debug(f"Predicted BLSTM residuals: {predicted_residuals}")
    return predicted_residuals

