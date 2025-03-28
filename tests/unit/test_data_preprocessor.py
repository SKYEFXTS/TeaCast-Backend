"""
Unit tests for the data preprocessor module.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Data.dataPreProcessor import (
    scale_input,
    inverse_scale_output,
    prepare_data_for_blstm
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Date': dates,
        'USD_Buying': [300, 305, 308, 302, 310],
        'Crude_Oil_Price_LKR': [8000, 8100, 8150, 8050, 8200],
        'Week': [1, 2, 3, 4, 5],
        'Auction_Number': [1, 2, 3, 4, 5],
        'Price': [100, 110, 105, 115, 120],
        'SARIMAX_Predicted': [102, 108, 106, 113, 118]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_scalers(mocker):
    """Create mock scalers for testing."""
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Sample data to fit scalers
    X_data = np.array([[300, 8000, 1, 1, 100],
                       [305, 8100, 2, 2, 110],
                       [308, 8150, 3, 3, 105]])
    y_data = np.array([100, 110, 105])
    
    X_scaler.fit(X_data)
    y_scaler.fit(y_data.reshape(-1, 1))
    
    # Mock the loader functions
    mocker.patch('Data.dataPreProcessor.load_X_scaler', return_value=X_scaler)
    mocker.patch('Data.dataPreProcessor.load_y_scaler', return_value=y_scaler)
    
    return X_scaler, y_scaler

def test_scale_input(sample_data, mock_scalers):
    """Test scaling input data."""
    input_data = sample_data.iloc[0]
    result = scale_input(input_data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 5)  # One row, five features
    assert not np.any(np.isnan(result))

def test_inverse_scale_output(mock_scalers):
    """Test inverse scaling of output data."""
    test_data = np.array([[-1.0], [0.0], [1.0]])
    result = inverse_scale_output(test_data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)  # Flattened array
    assert not np.any(np.isnan(result))

def test_prepare_data_for_blstm(sample_data, mock_scalers, mocker):
    """Test preparing data for BLSTM model."""
    # Mock the save function to avoid file operations
    mocker.patch('Data.dataPreProcessor.save_dataframe_as_csv')
    
    # Create SARIMAX predictions
    sarimax_predictions = pd.Series([102, 108, 106, 113, 118])
    
    X_sequences, extended_data = prepare_data_for_blstm(
        sample_data.copy(),  # Use copy to avoid modifying original
        sarimax_predictions,
        forecast_auctions=2,
        seq_length=3,
        X_scaler=mock_scalers[0]
    )
    
    assert isinstance(X_sequences, np.ndarray)
    assert isinstance(extended_data, pd.DataFrame)
    assert X_sequences.shape[0] > 0
    assert X_sequences.shape[2] == 5  # Five features
    assert len(extended_data) == len(sample_data) + 2  # Original + forecast_auctions

def test_prepare_data_for_blstm_no_scaler(sample_data):
    """Test preparing data for BLSTM without scaler."""
    sarimax_predictions = pd.Series([102, 108, 106, 113, 118])
    
    result = prepare_data_for_blstm(
        sample_data.copy(),  # Use copy to avoid modifying original
        sarimax_predictions,
        forecast_auctions=2,
        seq_length=3,
        X_scaler=None
    )
    
    assert result is None

def test_prepare_data_for_blstm_future_data(sample_data, mock_scalers, mocker):
    """Test future data generation in BLSTM preparation."""
    # Mock the save function to avoid file operations
    mocker.patch('Data.dataPreProcessor.save_dataframe_as_csv')
    
    sarimax_predictions = pd.Series([102, 108, 106, 113, 118])
    
    _, extended_data = prepare_data_for_blstm(
        sample_data.copy(),  # Use copy to avoid modifying original
        sarimax_predictions,
        forecast_auctions=2,
        seq_length=3,
        X_scaler=mock_scalers[0]
    )
    
    assert len(extended_data) == len(sample_data) + 2
    assert all(col in extended_data.columns for col in ['USD_Buying', 'Crude_Oil_Price_LKR', 'Week', 'Auction_Number', 'SARIMAX_Predicted'])
    # Check only non-Date columns for null values
    non_date_columns = [col for col in extended_data.columns if col != 'Date']
    assert not extended_data[non_date_columns].isnull().any().any() 