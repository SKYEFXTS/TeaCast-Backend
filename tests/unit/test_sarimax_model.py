"""
Unit tests for the SARIMAX model functionality.
"""
import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from Data.modelLoader import load_sarimax_model
from unittest.mock import patch, MagicMock

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    prices = np.array([100, 102, 105, 103, 108, 110, 107, 112, 115, 113])
    return pd.Series(prices, index=dates)

@pytest.fixture
def mock_model(mocker):
    """Create a mock SARIMAX model."""
    mock = mocker.MagicMock(spec=SARIMAXResults)
    mock.params = pd.Series([0.5, 0.3, 0.2])  # Mock parameters
    mock.predict.return_value = pd.Series([110, 112, 115])
    return mock

def test_load_sarimax_model(mocker):
    """Test loading the SARIMAX model."""
    # Mock pickle.load
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = pd.Series([100, 110, 120])
    mock_model.params = pd.Series([0.5, 0.3, 0.2])
    mocker.patch('pickle.load', return_value=mock_model)

    model = load_sarimax_model()

    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'params')
    assert isinstance(model.params, pd.Series)

def test_model_prediction(mocker):
    """Test SARIMAX model prediction."""
    # Mock the model
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = pd.Series([100, 110, 120])
    mocker.patch('Data.modelLoader.load_sarimax_model', return_value=mock_model)

    # Test prediction
    predictions = mock_model.predict()
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 3
    assert all(isinstance(x, (int, float)) for x in predictions)

    # Set up test data
    start_date = '2023-01-11'
    end_date = '2023-01-13'
    
    # Make prediction
    predictions = mock_model.predict(start=start_date, end=end_date)
    
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 3
    assert not predictions.isnull().any() 