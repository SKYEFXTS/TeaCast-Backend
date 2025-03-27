import pandas as pd
import logging
from Data.datasetLoader import load_and_filter_data

def get_last_auctions_average_prices():
    """
    Calculate the average prices for the last auctions for 3 main tea categories.
    """
    try:
        # Load and filter data for the specified tea categories and grades
        WBOPF_df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", None)
        WBOP_df = load_and_filter_data("WESTERN HIGH", "BOP", None)
        LFBOPF_df = load_and_filter_data("LOW GROWNS", "FBOPF1", None)

        # Get the last row information for each category
        WBOPF_last_row = get_last_row_info(WBOPF_df)
        WBOP_last_row = get_last_row_info(WBOP_df)  # Overwrite is fine since it's used separately
        LFBOPF_last_row = get_last_row_info(LFBOPF_df)

        # Handle case if any of the last row info is None (empty DataFrame)
        if None in [WBOPF_last_row, WBOP_last_row, LFBOPF_last_row]:
            logging.warning("One or more DataFrames are empty, cannot fetch average prices.")
            return {'error': 'Data not available for one or more categories'}

        # Format the data as required by the frontend
        formatted_data = [
            create_formatted_entry('WESTERN HIGH - BOPF/BOPFSp', WBOPF_last_row),
            create_formatted_entry('WESTERN HIGH - BOP', WBOP_last_row),
            create_formatted_entry('LOW GROWNS - FBOPF1', LFBOPF_last_row)
        ]

        logging.debug(f'Formatted data: {formatted_data}')
        return formatted_data

    except Exception as e:
        logging.error(f'Error calculating average prices: {e}')
        raise

def create_formatted_entry(name, row_data):
    """Helper function to format the row data for the frontend response."""
    return {
        'name': name,
        'date': row_data['date'].strftime('%b %d, %Y'),
        'price': f"LKR{row_data['price']}/kg"
    }

def get_last_row_info(df):
    """
    Get the last row's date (from the index), price, and concatenated category and grade as name.

    :param df: DataFrame containing the data
    :return: A dictionary with the last row's date and price
    """
    if df.empty:
        return None

    # Get the last row
    last_row = df.iloc[-1]

    # Extract the required information
    return {
        'date': last_row.name,  # Date is the index
        'price': last_row['Price']
    }
