import logging

import pandas as pd

def load_and_filter_data(category, grade, sarimax_model):
    """Loads and filters dataset by category and grade."""
    try:
        # Load dataset
        df = pd.read_csv("Data/TeaCast Dataset.csv")

        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # Filter by Category and Grade
        df_filtered = df[(df["Category"] == category) & (df["Grade"] == grade)]

        # Aggregate by Date
        df_filtered = df_filtered.groupby("Date").agg({
            "Price": "mean",
            "USD_Buying": "first",
            "Crude_Oil_Price_LKR": "first",
            "Week": "first",
            "Auction_Number": "first"
        })

        df_filtered["SARIMAX_Predicted"] = sarimax_model.fittedvalues
        return df_filtered

    except Exception as e:
        logging.error(f"Error in loading and filtering data: {e}")
        raise
