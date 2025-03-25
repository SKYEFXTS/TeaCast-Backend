import pandas as pd

def save_dataframe_as_csv(dataframe, file_path):
    """
    Saves the given DataFrame as a CSV file.

    :param dataframe: The DataFrame to save.
    :param file_path: The path where the CSV file will be saved.
    """
    try:
        dataframe.to_csv(file_path, index=False)  # index=False prevents writing row numbers
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error while saving DataFrame to CSV: {e}")
