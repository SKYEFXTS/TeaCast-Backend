import pandas as pd

def load_and_filter_data(category, grade, sarimax_model):
    """
    Loads the dataset, filters by category and grade, and aggregates the data by date.

    :param file_path: Path to the CSV file.
    :param category: The category to filter the dataset.
    :param grade: The grade to filter the dataset.
    :return: A DataFrame containing the filtered and aggregated data.
    """
    # Load dataset
    df = pd.read_csv("Data/TeaCast Dataset.csv")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Filter the dataset by Category and Grade
    df_filtered = df[(df["Category"] == category) & (df["Grade"] == grade)]

    # Aggregate the data by Date
    df_filtered = df_filtered.groupby("Date").agg({
        "Price": "mean",  # Compute mean for Price
        "USD_Buying": "first",  # Take the first occurrence (same value for the day)
        "Crude_Oil_Price_LKR": "first",  # Take the first occurrence (same value for the day)
        "Week": "first",  # Take the first occurrence (same value for the day)
        "Auction_Number": "first"  # Take the first occurrence (same value for the day)
    })

    df_filtered["SARIMAX_Predicted"] = sarimax_model.fittedvalues

    return df_filtered