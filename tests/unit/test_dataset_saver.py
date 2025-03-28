"""
Unit tests for the dataset saver module.
"""
import os
import pytest
import pandas as pd
from Data.datasetSaver import save_dataframe_as_csv

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Date': dates,
        'Price': [100, 110, 105, 115, 120],
        'Category': ['WESTERN HIGH'] * 5
    }
    return pd.DataFrame(data)

def test_save_dataframe_as_csv(sample_dataframe, tmp_path):
    """Test saving non-empty DataFrame to CSV."""
    file_path = str(tmp_path / "test_output.csv")
    
    save_dataframe_as_csv(sample_dataframe, file_path)
    
    assert os.path.exists(file_path)
    saved_df = pd.read_csv(file_path)
    # Convert Date column back to datetime for comparison
    saved_df['Date'] = pd.to_datetime(saved_df['Date'])
    pd.testing.assert_frame_equal(
        saved_df,
        sample_dataframe.reset_index(drop=True),
        check_dtype=False
    )

def test_save_dataframe_as_csv_empty(tmp_path):
    """Test saving empty DataFrame."""
    empty_df = pd.DataFrame(columns=['Date', 'Price', 'Category'])
    file_path = str(tmp_path / "empty_output.csv")
    
    save_dataframe_as_csv(empty_df, file_path)
    
    assert os.path.exists(file_path)
    saved_df = pd.read_csv(file_path)
    assert saved_df.empty
    assert list(saved_df.columns) == ['Date', 'Price', 'Category']

def test_save_dataframe_as_csv_invalid_path():
    """Test saving DataFrame with invalid path."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    invalid_path = "/invalid/path/test.csv"

    with pytest.raises(Exception) as exc_info:
        save_dataframe_as_csv(df, invalid_path)
    assert "Error saving DataFrame" in str(exc_info.value)

def test_save_dataframe_as_csv_with_index(sample_dataframe, tmp_path):
    """Test saving DataFrame with index."""
    file_path = str(tmp_path / "indexed_output.csv")

    # Set an index
    indexed_df = sample_dataframe.set_index('Date')
    save_dataframe_as_csv(indexed_df, file_path, index=True)

    # Read back and verify
    saved_df = pd.read_csv(file_path)
    saved_df = saved_df.set_index(saved_df.columns[0])
    saved_df.index = pd.to_datetime(saved_df.index)

    # Compare with the indexed DataFrame
    pd.testing.assert_frame_equal(
        saved_df,
        indexed_df,
        check_dtype=False
    )

def test_save_dataframe_as_csv_with_special_chars(tmp_path):
    """Test saving DataFrame with special characters."""
    special_df = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Name': ['Test, with, commas'],
        'Description': ['Test "quotes" and spaces']
    })
    file_path = str(tmp_path / "special_output.csv")
    
    save_dataframe_as_csv(special_df, file_path)
    
    saved_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(
        saved_df,
        special_df.reset_index(drop=True),
        check_dtype=False
    ) 