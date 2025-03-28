"""
Dataset Saver Module
This module handles the saving of pandas DataFrames to CSV files.
It provides functionality for persisting processed data and model outputs.
"""

import pandas as pd
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

def save_dataframe_as_csv(dataframe, file_path):
    """
    Saves the given DataFrame as a CSV file.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save
        file_path (str): The path where the CSV file will be saved
        
    Raises:
        Exception: If there's an error saving the DataFrame to CSV
    """
    try:
        # Save the DataFrame as a CSV file without including the row index
        dataframe.to_csv(file_path, index=False)
        logging.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        # Log any errors that occur while saving the DataFrame
        logging.error(f"Error while saving DataFrame to CSV: {e}")
