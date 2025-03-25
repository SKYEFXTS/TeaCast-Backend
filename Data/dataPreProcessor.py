from Data.modelLoader import load_X_scaler, load_y_scaler
from Data.datasetSaver import save_dataframe_as_csv
import numpy as np
import pandas as pd
import logging

def scale_input(data_df):
    """
    Scales the input features using the pre-loaded X scaler.

    :param data: DataFrame containing features such as 'USD_Buying', 'Crude_Oil_Price_LKR', etc.
    :return: Scaled input data for model prediction
    """

    # Load the X scaler
    X_scaler = load_X_scaler()

    # Assuming the data contains ["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number", "SARIMAX_Predicted"]
    input_data = np.array([data_df['USD_Buying'], data_df['Crude_Oil_Price_LKR'], data_df['Week'], data_df['Auction_Number'], data_df['SARIMAX_Predicted']])

    # Reshape the data to match the scaler's expected input shape (1 row, multiple columns)
    input_data = X_scaler.transform(input_data.reshape(1, -1))  # Shape (1, number of features)

    return input_data

def inverse_scale_output(prediction):
    """
    Inverse scales the prediction output using the pre-loaded y scaler.

    :param prediction: The model's predicted output (scaled)
    :return: The actual prediction (inverse scaled)
    """

    # Load the y scaler
    y_scaler = load_y_scaler()

    # Since prediction could be 1D, reshape to (n_samples, 1) before applying inverse_transform
    prediction_reshaped = prediction.reshape(-1, 1)

    # Inverse scale the prediction
    prediction = y_scaler.inverse_transform(prediction_reshaped)

    # Return the result (can be reshaped or flattened to 1D if necessary)
    return prediction.flatten()  # Flatten back to 1D array if needed

def prepare_data_for_blstm(original_df, sarimax_predictions, forecast_auctions=10, seq_length=10, X_scaler=None):
    """
    Prepares the data for BLSTM by creating sequences of the past auctions' features
    along with future SARIMAX predictions.

    :param original_df: The original DataFrame containing the historical data.
    :param sarimax_predictions: The forecasted SARIMAX predictions for future auctions.
    :param forecast_auctions: Number of future auctions to forecast.
    :param seq_length: The length of the sequence (e.g., 10 past auctions).
    :param X_scaler: The scaler used for feature scaling.
    :return: The input data (X) for BLSTM (sequences of past data and SARIMAX predictions).
    """

    features = ["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number", "SARIMAX_Predicted"]

    # Drop the 'Price' column because it's not needed for the prediction
    original_df.drop("Price", axis=1, inplace=True)

    # Get the last known data (for filling future data)
    last_known_data = original_df.iloc[-1][["USD_Buying", "Crude_Oil_Price_LKR"]].values
    last_week = original_df["Week"].iloc[-1]
    last_auction_number = original_df["Auction_Number"].iloc[-1]

    logging.debug(f"Last known data: USD_Buying: {last_known_data[0]}, Crude_Oil_Price_LKR: {last_known_data[1]}, Week: {last_week}, Auction_Number: {last_auction_number}")

    # Ensure that sarimax_predictions is a list or numpy array (not a pandas Series)
    if isinstance(sarimax_predictions, pd.Series):
        sarimax_predictions = sarimax_predictions.values  # Convert to numpy array if it's a pandas Series

    logging.debug(f"SARIMAX predictions received: {sarimax_predictions[:forecast_auctions]}")

    # Prepare the data for the future auctions (forecasted values)
    future_data = []

    for i in range(forecast_auctions):
        try:
            # If the last auction number is >= 50, reset to 1 and increment from 1
            if last_auction_number >= 50:
                future_auction_number = i + 1  # Reset auction number to 1 and increment
                future_week = i + 1  # Reset week to 1 and increment
            else:
                # Increment the auction number and week normally
                future_auction_number = last_auction_number + i
                future_week = last_week + i

            # Create a dictionary for future data
            future_data.append({
                "USD_Buying": last_known_data[0],  # Use the last known value for `USD_Buying`
                "Crude_Oil_Price_LKR": last_known_data[1],  # Use the last known value for `Crude_Oil_Price_LKR`
                "Week": future_week,  # Increment Week for the forecast
                "Auction_Number": future_auction_number,  # Increment Auction Number
                "SARIMAX_Predicted": sarimax_predictions[i]  # Use the SARIMAX prediction for the future
            })
        except Exception as e:
            logging.error(f"Error in the loop while creating future data for auction {i+1}: {e}")
            return None  # Return None to stop execution in case of error

    # Convert future_data into a DataFrame
    future_data_df = pd.DataFrame(future_data)

    # Concatenate the future data with the original dataset
    try:
        extended_data = pd.concat([original_df, future_data_df], ignore_index=True)
    except Exception as e:
        logging.error(f"Error while concatenating data: {e}")
        return None

    # Fill `USD_Buying` and `Crude_Oil_Price_LKR` with the last known values for forecasted rows
    extended_data.loc[extended_data["USD_Buying"].isna(), "USD_Buying"] = last_known_data[0]
    extended_data.loc[extended_data["Crude_Oil_Price_LKR"].isna(), "Crude_Oil_Price_LKR"] = last_known_data[1]

    # Manually increment `Week` and `Auction_Number` for the forecasted rows
    extended_data.loc[extended_data["Week"].isna(), "Week"] = last_week + 1
    extended_data.loc[extended_data["Auction_Number"].isna(), "Auction_Number"] = last_auction_number + 1


    # Ensure no NaNs are left in the data before scaling
    if extended_data[features].isnull().any().any():
        logging.warning("There are still NaN values in the data before scaling.")

    print("Saving the extended data...")

    # Save the extended data to a CSV file
    save_dataframe_as_csv(extended_data, "Data/PreProcessedData/extended_data.csv")

    print("Scaling the data...")

    # Scale the data using X_scaler
    if X_scaler:
        try:
            scaled_data = X_scaler.transform(extended_data[features])
            print("Saving the scaled data...")
            save_dataframe_as_csv(pd.DataFrame(scaled_data, columns=features), "Data/PreProcessedData/scaled_data.csv")
        except Exception as e:
            logging.error(f"Error during scaling: {e}")
            return None

        # Create sequences of data for BLSTM
        X = np.array([scaled_data[i:i + seq_length] for i in range(len(scaled_data) - seq_length)])

        # Get the last 10 sequences
        X_last_10 = X[-10:]

        return X_last_10, extended_data
    else:
        logging.error("X_scaler is required for scaling data.")
        return None



#print(extended_data.tail(50))
#print("Scaling the data...")
#print(last_known_data, last_week, last_auction_number)
#print(future_data_df.head(50))