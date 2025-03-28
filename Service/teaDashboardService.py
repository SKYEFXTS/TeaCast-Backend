import pandas as pd
import logging
from Data.datasetLoader import load_and_filter_data, load_dataset
from flask import jsonify

def get_tea_price_over_time():
    """
    Fetches historical data for Tea Price, USD Rate, and Crude Oil Price over time.
    :return: A dictionary containing lists of dictionaries for tea prices, USD rates, and crude oil prices
    """
    try:
        # Load and filter data for the specified tea categories and grades
        tea_df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", None)

        # Get the last 50 rows of the specified columns
        last_50_rows = tea_df[['Price', 'USD_Buying', 'Crude_Oil_Price_LKR']].tail(50).reset_index()

        # Prepare the data for the response
        tea_prices = [{"date": row['Date'].isoformat(), "price": round(row['Price'])} for _, row in last_50_rows.iterrows()]
        usd_rates = [{"date": row['Date'].isoformat(), "rate": round(row['USD_Buying'])} for _, row in last_50_rows.iterrows()]
        crude_oil_prices = [{"date": row['Date'].isoformat(), "price": round(row['Crude_Oil_Price_LKR'])} for _, row in last_50_rows.iterrows()]

        # Return the data as a dictionary
        return {
            "tea_prices": tea_prices,
            "usd_rates": usd_rates,
            "crude_oil_prices": crude_oil_prices
        }

    except Exception as e:
        logging.error(f"Error fetching tea price over time: {e}")
        raise

def get_average_price_for_category(category_name, file_path):
    """
    Given a category name, this function returns the average price for that category from the latest auction,
    along with the date of the auction and auction number.

    :param category_name: The name of the tea category (e.g., 'WESTERN MEDIUM')
    :param file_path: The path to the CSV dataset file
    :return: A dictionary containing the average price, date, and auction number
    """
    try:
        # Load the dataset using the load_dataset function
        df = load_dataset(file_path)

        # Filter the dataset based on the category name
        category_df = df[df['Category'] == category_name]

        # Check if there is any data for the specified category
        if category_df.empty:
            return {'error': f"No data found for category: {category_name}"}

        # Group by 'Date', assuming the most recent auction is the latest date
        latest_auction_date = category_df['Date'].max()

        # Get all records from the latest auction date
        latest_auction_df = category_df[category_df['Date'] == latest_auction_date]

        # Calculate the average price for the latest auction and round it up to an integer
        average_price = round(float(latest_auction_df['Price'].mean()))  # Convert to float and round

        # Extract the auction number from the first record
        auction_number = int(latest_auction_df['Auction_Number'].iloc[0])  # Convert to int

        # Format the auction date
        auction_date = latest_auction_date.strftime('%b %d, %Y')  # Formatting date as 'Month Day, Year'

        # Return the result as a dictionary
        return {
            'category': category_name,
            'average_price': average_price,
            'date': auction_date,
            'auction_number': auction_number
        }

    except Exception as e:
        logging.error(f"Error fetching average price for category {category_name}: {e}")
        return {'error': f"An error occurred: {str(e)}"}

def get_all_average_prices(file_path):
    """
    Fetches the average prices for all categories from the dataset.

    :param file_path: The path to the CSV dataset file
    :return: A dictionary containing the average prices for all categories
    """
    try:
        # Load the dataset using the load_dataset function
        df = load_dataset(file_path)

        # Get unique categories in the dataset
        unique_categories = df['Category'].unique()

        # Initialize a dictionary to hold average price data for each category
        all_average_price_data = {}

        # Loop through each category and get the average price data
        for category in unique_categories:
            # Get the average price for each category using the existing function
            category_average_data = get_average_price_for_category(category, file_path)

            # Store the average price data in the dictionary
            if 'error' not in category_average_data:
                all_average_price_data[category] = category_average_data
            else:
                logging.warning(f"Error fetching data for category: {category}")

        return all_average_price_data

    except Exception as e:
        logging.error(f"Error fetching average prices for all categories: {e}")
        raise
