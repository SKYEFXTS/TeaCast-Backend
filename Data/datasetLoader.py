import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_and_filter_data(category, grade, sarimax_model):
    """Loads and filters the dataset by category and grade."""
    try:
        logging.debug("Loading and filtering dataset by category and grade")

        # Load the dataset
        df = load_dataset("Data/TeaCast Dataset.csv")

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

        # Add the SARIMAX predictions as a new column if sarimax_model is provided
        if sarimax_model is not None:
            df_filtered["SARIMAX_Predicted"] = sarimax_model.fittedvalues

        logging.debug(f"Filtered data: {df_filtered.head()}")
        return df_filtered

    except Exception as e:
        logging.error(f"Error in loading and filtering data: {e}")
        raise

def load_dataset(path):
    """Loads the dataset from the CSV file."""
    try:
        logging.debug("Loading dataset from path: %s", path)

        # Load the dataset
        df = pd.read_csv(path)

        # Check if the expected columns exist in the dataset
        expected_columns = ['Date', 'Category', 'Grade', 'Price', 'USD_Buying', 'Crude_Oil_Price_LKR']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            logging.error(f"Missing columns in dataset: {missing_columns}")
            raise ValueError(f"Dataset is missing expected columns: {', '.join(missing_columns)}")

        # Convert 'Date' to datetime and sort by date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle invalid dates gracefully
        df = df.sort_values(by='Date')

        # Optionally, check if there are any rows with invalid or missing dates
        invalid_dates = df[df['Date'].isnull()]
        if not invalid_dates.empty:
            logging.warning(f"Found {len(invalid_dates)} rows with invalid or missing dates. They will be dropped.")
            df = df.dropna(subset=['Date'])

        logging.debug(f"Dataset loaded successfully with {len(df)} rows.")
        return df

    except FileNotFoundError:
        logging.error(f"File not found at path: {path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The file at {path} is empty.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file at {path}. It may not be in a valid CSV format.")
        raise
    except Exception as e:
        logging.error(f"Error in loading dataset: {e}")
        raise

