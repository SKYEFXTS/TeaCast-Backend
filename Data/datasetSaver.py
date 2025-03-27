import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def save_dataframe_as_csv(dataframe, file_path):
    """
    Saves the given DataFrame as a CSV file.

    :param dataframe: The DataFrame to save.
    :param file_path: The path where the CSV file will be saved.
    """
    try:
        # Save the DataFrame as a CSV file without including the row index
        dataframe.to_csv(file_path, index=False)
        logging.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        # Log any errors that occur while saving the DataFrame
        logging.error(f"Error while saving DataFrame to CSV: {e}")
