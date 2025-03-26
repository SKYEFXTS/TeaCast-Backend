from Data.modelLoader import load_sarimax_model, load_blstm_model, load_y_scaler, load_all_models
from Data.dataPreProcessor import scale_input, inverse_scale_output, prepare_data_for_blstm
from Data.datasetLoader import load_and_filter_data
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def get_prediction():
    """Gets the final prediction from SARIMAX and BLSTM models."""
    try:
        # Load models and scalers
        sarimax_model, blstm_model, X_scaler, y_scaler = load_all_models()
        logging.debug("Models and scalers loaded")

        # Load and filter data
        dataset_df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", sarimax_model)
        logging.debug(f"Data loaded and filtered: {dataset_df.head()}")

        # Get SARIMAX predictions
        dataset_df_sarimax = get_sarimax_prediction(dataset_df, 10, sarimax_model)
        sarimax_predictions = extract_sarimax_predictions(dataset_df_sarimax)

        # Prepare data for BLSTM
        X_test, final_prediction_df = prepare_data_for_blstm(dataset_df, dataset_df_sarimax, 10, 10, X_scaler)
        logging.debug(f"Data prepared for BLSTM: {X_test.shape}")

        # Get BLSTM residual prediction
        predicted_residuals = get_blstm_residual_prediction(X_test, blstm_model, y_scaler)

        # Flatten predicted residuals
        predicted_residuals = predicted_residuals.flatten()

        # Ensure no length mismatch between predictions
        if len(predicted_residuals) != len(sarimax_predictions):
            logging.error("Length mismatch: Residuals and SARIMAX predictions.")
            return None

        # Calculate the final prediction
        final_prediction = sarimax_predictions + predicted_residuals

        # Add predicted_residuals and final_prediction to the DataFrame
        final_prediction_df["Predicted_Residuals"] = np.nan
        final_prediction_df["Final_Prediction"] = np.nan

        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Predicted_Residuals")] = predicted_residuals
        final_prediction_df.iloc[-10:, final_prediction_df.columns.get_loc("Final_Prediction")] = final_prediction

        # Save the final prediction DataFrame to a CSV file
        final_prediction_df.to_csv("Data/Final_Prediction.csv", index=False)

        # Log the initialized DataFrame
        logging.debug(f"Final prediction DataFrame after prediction:\n{final_prediction_df.tail()}")

        # Return the final prediction
        return final_prediction

    except Exception as e:
        logging.error(f"Error in getting prediction: {e}")
        raise

def extract_sarimax_predictions(dataset_df_sarimax):
    """
    Extracts SARIMAX predictions either from a Series or DataFrame.
    Returns a numpy array of SARIMAX predictions.
    """
    if isinstance(dataset_df_sarimax, pd.Series):
        return dataset_df_sarimax.values  # Extract values if it's a Series
    else:
        return dataset_df_sarimax["SARIMAX_Predicted"].iloc[-10:].values  # Extract last 10 values if it's a DataFrame

def get_sarimax_prediction(data_df, forecast_auctions, sarimax_model):
    """
    Generates SARIMAX predictions using the provided model and exogenous data.

    :param data_df: DataFrame containing the required features such as 'USD_Buying', 'Crude_Oil_Price_LKR', etc.
    :param forecast_auctions: The number of auctions to forecast ahead.
    :param sarimax_model: The pre-loaded SARIMAX model.
    :return: A pandas Series or DataFrame containing the forecasted SARIMAX predicted values.
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
