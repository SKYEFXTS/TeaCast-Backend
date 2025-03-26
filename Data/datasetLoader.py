import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_and_filter_data(category, grade, sarimax_model):
    """Loads and filters the dataset by category and grade."""
    try:
        logging.debug("Loading and filtering dataset by category and grade")

        # Load the dataset
        df = pd.read_csv("Data/TeaCast Dataset.csv")

        # Convert 'Date' to datetime and sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # Filter the dataset by Category and Grade
        df_filtered = df[(df["Category"] == category) & (df["Grade"] == grade)]

        # Aggregate data by Date, keeping the first occurrence for non-aggregated columns
        df_filtered = df_filtered.groupby("Date").agg({
            "Price": "mean",  # Calculate mean for Price
            "USD_Buying": "first",  # Keep the first occurrence for other columns
            "Crude_Oil_Price_LKR": "first",
            "Week": "first",
            "Auction_Number": "first"
        })

        # Add the SARIMAX predictions as a new column
        df_filtered["SARIMAX_Predicted"] = sarimax_model.fittedvalues

        logging.debug(f"Filtered data: {df_filtered.head()}")
        return df_filtered

    except Exception as e:
        logging.error(f"Error in loading and filtering data: {e}")
        raise
