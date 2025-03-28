"""
Tea Auction Price Service Module
This module handles the retrieval and processing of tea auction price data.
It provides functionality to calculate and format average prices for different tea categories
and grades from the most recent auctions.
"""

import pandas as pd
import logging
from Data.datasetLoader import load_and_filter_data

def get_last_auctions_average_prices():
    """
    Calculate the average prices for the last auctions for 3 main tea categories.
    
    Returns:
        list: List of dictionaries containing formatted price data for each category:
            {
                'name': 'Category - Grade',
                'date': 'Formatted Date',
                'price': 'Formatted Price'
            }
            
    Raises:
        Exception: If there's an error in data loading or processing
    """
    try:
        # Load and filter data for the specified tea categories and grades
        # These are the main categories used for price tracking
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
    """
    Helper function to format the row data for the frontend response.
    
    Args:
        name (str): Category and grade name
        row_data (dict): Dictionary containing date and price information
        
    Returns:
        dict: Formatted entry with name, date, and price
    """
    return {
        'name': name,
        'date': row_data['date'].strftime('%b %d, %Y'),
        'price': f"LKR{row_data['price']}/kg"
    }

def get_last_row_info(df):
    """
    Get the last row's date (from the index), price, and concatenated category and grade as name.
    
    Args:
        df (pd.DataFrame): DataFrame containing the auction data
        
    Returns:
        dict: Dictionary containing the last row's date and price information
        None: If the DataFrame is empty
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
