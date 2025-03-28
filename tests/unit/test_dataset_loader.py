"""
Unit tests for the dataset loader module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from Data.datasetLoader import load_dataset, load_and_filter_data

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Date': dates,
        'Category': ['WESTERN HIGH'] * 5,
        'Grade': ['BOPF/BOPFSp'] * 5,
        'Price': [100, 110, 105, 115, 120],
        'USD_Buying': [300, 305, 302, 308, 310],
        'Crude_Oil_Price_LKR': [8000, 8100, 8050, 8150, 8200],
        'Week': [1, 2, 3, 4, 5],
        'Auction_Number': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_csv_file(tmp_path, sample_dataset):
    """Create a temporary CSV file with sample data."""
    file_path = tmp_path / "test_dataset.csv"
    sample_dataset.to_csv(file_path, index=False)
    return str(file_path)

def test_load_dataset(mock_csv_file):
    """Test loading dataset from CSV file."""
    df = load_dataset(mock_csv_file)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert all(col in df.columns for col in ['Date', 'Category', 'Grade', 'Price'])
    assert isinstance(df['Date'].iloc[0], datetime)

def test_load_dataset_missing_columns(tmp_path):
    """Test loading dataset with missing required columns."""
    # Create a CSV with missing columns
    data = {'Date': ['2023-01-01'], 'Category': ['WESTERN HIGH']}
    df = pd.DataFrame(data)
    file_path = tmp_path / "invalid_dataset.csv"
    df.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Dataset is missing expected columns"):
        load_dataset(str(file_path))

def test_load_and_filter_data(sample_dataset, mocker):
    """Test filtering data by category and grade."""
    # Mock the load_dataset function to return our sample dataset
    mocker.patch('Data.datasetLoader.load_dataset', return_value=sample_dataset)
    
    df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", None)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Check if the filtered data contains only the specified category and grade
    if 'Category' in df.columns and 'Grade' in df.columns:
        assert all(df['Category'] == 'WESTERN HIGH')
        assert all(df['Grade'] == 'BOPF/BOPFSp')

def test_load_and_filter_data_empty_result(sample_dataset, mocker):
    """Test filtering data with no matching results."""
    mocker.patch('Data.datasetLoader.load_dataset', return_value=sample_dataset)
    
    df = load_and_filter_data("NONEXISTENT", "NONEXISTENT", None)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert all(col in df.columns for col in ['Price', 'USD_Buying', 'Crude_Oil_Price_LKR'])

def test_load_and_filter_data_with_sarimax(sample_dataset, mocker):
    """Test filtering data with SARIMAX model predictions."""
    mocker.patch('Data.datasetLoader.load_dataset', return_value=sample_dataset)
    
    # Mock SARIMAX model
    mock_sarimax = mocker.MagicMock()
    mock_sarimax.fittedvalues = pd.Series([100, 110, 105, 115, 120])
    
    df = load_and_filter_data("WESTERN HIGH", "BOPF/BOPFSp", mock_sarimax)
    
    assert 'SARIMAX_Predicted' in df.columns
    assert len(df['SARIMAX_Predicted']) == 5 