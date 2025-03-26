from Data.modelLoader import load_X_scaler, load_y_scaler
from Data.datasetSaver import save_dataframe_as_csv
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def scale_input(data_df):
    """Scales the input features using the pre-loaded X scaler."""
    try:
        # Load the X scaler
        X_scaler = load_X_scaler()

        # Prepare the input data
        input_data = np.array([data_df['USD_Buying'], data_df['Crude_Oil_Price_LKR'],
                               data_df['Week'], data_df['Auction_Number'], data_df['SARIMAX_Predicted']])

        # Reshape to match scaler input shape (1 row, multiple columns)
        input_data = X_scaler.transform(input_data.reshape(1, -1))

        return input_data
    except Exception as e:
        logging.error(f"Error in scaling input data: {e}")
        raise

def inverse_scale_output(prediction):
    """Inverse scales the prediction output using the pre-loaded y scaler."""
    try:
        # Load the y scaler
        y_scaler = load_y_scaler()

        # Reshape prediction for inverse transformation
        prediction_reshaped = prediction.reshape(-1, 1)

        # Inverse scale
        return y_scaler.inverse_transform(prediction_reshaped).flatten()
    except Exception as e:
        logging.error(f"Error in inverse scaling output: {e}")
        raise

def prepare_data_for_blstm(original_df, sarimax_predictions, forecast_auctions=10, seq_length=10, X_scaler=None):
    """
    Prepares the data for BLSTM by creating sequences of the past auctions' features along with future SARIMAX predictions.
    """
    features = ["USD_Buying", "Crude_Oil_Price_LKR", "Week", "Auction_Number", "SARIMAX_Predicted"]

    try:
        # Drop the 'Price' column because it's not needed for the prediction
        original_df.drop("Price", axis=1, inplace=True)

        # Get the last known data (for filling future data)
        last_known_data = original_df.iloc[-1][["USD_Buying", "Crude_Oil_Price_LKR"]].values
        last_week = original_df["Week"].iloc[-1]
        last_auction_number = original_df["Auction_Number"].iloc[-1]

        logging.debug(f"Last known data: USD_Buying: {last_known_data[0]}, Crude_Oil_Price_LKR: {last_known_data[1]}, Week: {last_week}, Auction_Number: {last_auction_number}")

        # Ensure that sarimax_predictions is a numpy array
        if isinstance(sarimax_predictions, pd.Series):
            sarimax_predictions = sarimax_predictions.values  # Convert to numpy array if it's a pandas Series

        logging.debug(f"SARIMAX predictions received: {sarimax_predictions[:forecast_auctions]}")

        # Prepare the data for the future auctions (forecasted values)
        future_data = []

        for i in range(forecast_auctions):
            try:
                # Check auction number and reset it if >= 50
                future_auction_number = 1 if last_auction_number >= 50 else last_auction_number + i
                future_week = 1 if last_auction_number >= 50 else last_week + i

                # Create a dictionary for future data
                future_data.append({
                    "USD_Buying": last_known_data[0],
                    "Crude_Oil_Price_LKR": last_known_data[1],
                    "Week": future_week,
                    "Auction_Number": future_auction_number,
                    "SARIMAX_Predicted": sarimax_predictions[i]
                })
            except Exception as e:
                logging.error(f"Error in creating future data for auction {i+1}: {e}")
                return None

        # Convert future data into a DataFrame
        future_data_df = pd.DataFrame(future_data)

        # Concatenate the future data with the original dataset
        extended_data = pd.concat([original_df, future_data_df], ignore_index=True)

        # Fill missing data for 'USD_Buying' and 'Crude_Oil_Price_LKR' using the last known data
        extended_data["USD_Buying"].fillna(last_known_data[0], inplace=True)
        extended_data["Crude_Oil_Price_LKR"].fillna(last_known_data[1], inplace=True)

        # Manually increment 'Week' and 'Auction_Number' for the forecasted rows
        extended_data["Week"].fillna(last_week + 1, inplace=True)
        extended_data["Auction_Number"].fillna(last_auction_number + 1, inplace=True)

        # Ensure no NaN values are left before scaling
        if extended_data[features].isnull().any().any():
            logging.warning("There are still NaN values in the data before scaling.")

        # Save the extended data to a CSV file
        save_dataframe_as_csv(extended_data, "Data/PreProcessedData/extended_data.csv")

        # Scale the data using X_scaler
        if X_scaler:
            scaled_data = X_scaler.transform(extended_data[features])

            # Save the scaled data to a CSV file
            save_dataframe_as_csv(pd.DataFrame(scaled_data, columns=features), "Data/PreProcessedData/scaled_data.csv")

            # Create sequences of data for BLSTM
            X = np.array([scaled_data[i:i + seq_length] for i in range(len(scaled_data) - seq_length)])

            # Return the last 10 sequences
            X_last_10 = X[-10:]

            return X_last_10, extended_data
        else:
            logging.error("X_scaler is required for scaling data.")
            return None
    except Exception as e:
        logging.error(f"Error in preparing data for BLSTM: {e}")
        raise
