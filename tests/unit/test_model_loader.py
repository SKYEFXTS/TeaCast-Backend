"""
Unit tests for the model loader module.
"""
import pytest
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from Data.modelLoader import (
    load_model,
    load_sarimax_model,
    load_blstm_model,
    load_X_scaler,
    load_y_scaler,
    load_all_models
)
import pickle
import os
import pandas as pd
from unittest.mock import MagicMock

class MockSARIMAX:
    """A picklable mock SARIMAX model."""
    def __init__(self):
        self.predict_values = np.array([100, 110, 120])
        self.params = pd.Series([0.5, 0.3, 0.2])

    def predict(self, *args, **kwargs):
        return self.predict_values

@pytest.fixture
def mock_sarimax_model():
    """Create a mock SARIMAX model."""
    # Create a picklable mock
    model = MockSARIMAX()
    
    # Create a temporary file with the mock
    temp_path = "temp_sarimax.joblib"
    with open(temp_path, 'wb') as f:
        pickle.dump(model, f)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def mock_blstm_model():
    """Create a mock BLSTM model."""
    model = Sequential()
    return model

@pytest.fixture
def mock_X_scaler():
    """Mock X scaler fixture."""
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5]])
    return mock_scaler

@pytest.fixture
def mock_y_scaler():
    """Mock y scaler fixture."""
    mock_scaler = MagicMock()
    mock_scaler.inverse_transform.return_value = np.array([[100], [110], [105], [115], [120]])
    return mock_scaler

def test_load_model_sarimax(mock_sarimax_model):
    """Test loading a SARIMAX model."""
    model = load_model("SARIMAX", mock_sarimax_model)
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'params')
    predictions = model.predict()
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3

def test_load_model_blstm(mock_blstm_model, mocker):
    """Test loading BLSTM model."""
    mocker.patch('tensorflow.keras.models.load_model', return_value=mock_blstm_model)
    result = load_model('BLSTM', 'dummy_path.keras')
    assert result is not None
    assert isinstance(result, Sequential)

def test_load_model_scaler(mock_X_scaler, mock_y_scaler, mocker):
    """Test loading scaler."""
    mocker.patch('Data.modelLoader.joblib.load', side_effect=[mock_X_scaler, mock_y_scaler])
    
    X_scaler = load_X_scaler("temp_X_scaler.joblib")
    assert X_scaler is not None
    assert hasattr(X_scaler, 'transform')

    y_scaler = load_y_scaler("temp_y_scaler.joblib")
    assert y_scaler is not None
    assert hasattr(y_scaler, 'inverse_transform')

def test_load_model_invalid_type():
    """Test loading model with invalid type."""
    with pytest.raises(ValueError):
        load_model('INVALID_TYPE', 'dummy_path')

def test_load_sarimax_model(mock_sarimax_model):
    """Test loading the SARIMAX model."""
    model = load_sarimax_model(mock_sarimax_model)
    assert model is not None
    assert hasattr(model, 'predict')
    predictions = model.predict()
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3

def test_load_blstm_model(mock_blstm_model, mocker):
    """Test loading BLSTM model specifically."""
    mocker.patch('tensorflow.keras.models.load_model', return_value=mock_blstm_model)
    model = load_blstm_model('dummy_path.keras')
    assert model is not None
    assert isinstance(model, Sequential)

def test_load_X_scaler(mock_X_scaler, mocker):
    """Test loading X scaler."""
    mocker.patch('Data.modelLoader.joblib.load', return_value=mock_X_scaler)
    scaler = load_X_scaler("temp_X_scaler.joblib")
    assert scaler is not None
    assert hasattr(scaler, 'transform')

def test_load_y_scaler(mock_y_scaler, mocker):
    """Test loading y scaler."""
    mocker.patch('Data.modelLoader.joblib.load', return_value=mock_y_scaler)
    scaler = load_y_scaler("temp_y_scaler.joblib")
    assert scaler is not None
    assert hasattr(scaler, 'inverse_transform')

def test_load_all_models(mock_sarimax_model, mock_blstm_model, mock_X_scaler, mock_y_scaler, mocker):
    """Test loading all models."""
    mock_sarimax = MagicMock()
    mock_sarimax.predict.return_value = pd.Series([100, 110, 105, 115, 120])
    mock_sarimax.params = pd.Series({'ar1': 0.5, 'ma1': 0.3})

    mocker.patch('Data.modelLoader.load_sarimax_model', return_value=mock_sarimax)
    mocker.patch('Data.modelLoader.load_blstm_model', return_value=mock_blstm_model)
    mocker.patch('Data.modelLoader.load_X_scaler', return_value=mock_X_scaler)
    mocker.patch('Data.modelLoader.load_y_scaler', return_value=mock_y_scaler)

    sarimax_model, blstm_model, X_scaler, y_scaler = load_all_models()
    assert sarimax_model is not None
    assert blstm_model is not None
    assert X_scaler is not None
    assert y_scaler is not None
    assert hasattr(sarimax_model, 'predict')
    assert hasattr(blstm_model, 'predict')
    assert hasattr(X_scaler, 'transform')
    assert hasattr(y_scaler, 'inverse_transform')

def test_load_all_models_error(mocker):
    """Test error handling in loading all models."""
    mocker.patch('Data.modelLoader.load_sarimax_model', side_effect=Exception("Test error"))
    
    with pytest.raises(Exception):
        load_all_models() 