"""
Dataset Saver Module
This module handles the saving of pandas DataFrames to CSV files.
It provides functionality for persisting processed data and model outputs.
"""

import pandas as pd
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)

def save_dataframe_as_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save
        file_path (str): The path where to save the CSV file
        index (bool, optional): Whether to save the index. Defaults to False.

    Raises:
        Exception: If there's an error saving the DataFrame
    """
    try:
        df.to_csv(file_path, index=index)
        logging.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        # Log any errors that occur while saving the DataFrame
        logging.error(f"Error while saving DataFrame to CSV: {e}")
        raise Exception(f"Error saving DataFrame to {file_path}: {str(e)}")
