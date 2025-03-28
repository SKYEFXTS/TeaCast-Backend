"""
Unit tests for the BLSTM model functionality.
"""
import pytest
import numpy as np
import tensorflow as tf
from Data.modelLoader import load_blstm_model

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample sequences
    X = np.random.rand(10, 3, 5)  # 10 samples, 3 timesteps, 5 features
    y = np.random.rand(10, 1)  # 10 samples, 1 target value
    return X, y

@pytest.fixture
def mock_model(mocker):
    """Create a mock BLSTM model."""
    mock = mocker.MagicMock(spec=tf.keras.Model)
    mock.input_shape = (None, 3, 5)
    mock.output_shape = (None, 1)
    return mock

def test_load_blstm_model(mocker):
    """Test loading the BLSTM model."""
    # Mock tensorflow's load_model function
    mock_model = mocker.MagicMock(spec=tf.keras.Model)
    mock_model.input_shape = (None, 3, 5)
    mock_model.output_shape = (None, 1)
    mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)
    
    model = load_blstm_model()
    
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 3, 5)
    assert model.output_shape == (None, 1)

def test_model_prediction(mock_model):
    """Test model prediction functionality."""
    # Create sample input
    X = np.random.rand(1, 3, 5)
    
    # Set up mock prediction
    mock_model.predict.return_value = np.array([[100.0]])
    
    # Make prediction
    prediction = mock_model.predict(X)
    
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 1)
    assert not np.any(np.isnan(prediction)) 