"""
Unit tests for the prediction service module.
"""
import pytest
import pandas as pd
import numpy as np
from Service.predictionService import (
    get_prediction,
    extract_sarimax_predictions,
    get_sarimax_prediction,
    get_blstm_residual_prediction
)

@pytest.fixture
def mock_models():
    """Create mock models and scalers."""
    class MockSARIMAX:
        def forecast(self, steps, exog):
            # Return predictions based on the requested number of steps
            predictions = [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]
            return pd.Series(predictions[:steps])
    
    class MockBLSTM:
        def predict(self, X):
            # Return predictions based on input shape
            batch_size = X.shape[0]
            predictions = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])
            return predictions[:batch_size]
    
    class MockScaler:
        def transform(self, X):
            return X
        def inverse_transform(self, X):
            return X
    
    return MockSARIMAX(), MockBLSTM(), MockScaler(), MockScaler()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'USD_Buying': [300, 305, 302, 308, 310],
        'Crude_Oil_Price_LKR': [8000, 8100, 8050, 8150, 8200],
        'Week': [1, 2, 3, 4, 5],
        'Auction_Number': [1, 2, 3, 4, 5],
        'SARIMAX_Predicted': [100, 110, 105, 115, 120]
    })

def test_extract_sarimax_predictions_series():
    """Test extracting predictions from Series."""
    predictions = pd.Series([100, 110, 105, 115, 120])
    result = extract_sarimax_predictions(predictions)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 5
    assert all(result == predictions.values)

def test_extract_sarimax_predictions_dataframe():
    """Test extracting predictions from DataFrame."""
    predictions = pd.DataFrame({
        'SARIMAX_Predicted': [100, 110, 105, 115, 120]
    })
    result = extract_sarimax_predictions(predictions)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 5
    assert all(result == predictions['SARIMAX_Predicted'].values)

def test_get_sarimax_prediction(sample_data, mock_models):
    """Test SARIMAX prediction generation."""
    sarimax_model, _, _, _ = mock_models
    
    result = get_sarimax_prediction(sample_data, 5, sarimax_model)
    
    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert all(isinstance(x, (int, float)) for x in result)

def test_get_blstm_residual_prediction(mock_models):
    """Test BLSTM residual prediction."""
    _, blstm_model, _, y_scaler = mock_models
    
    X_test = np.random.rand(5, 10)  # Random test data
    result = get_blstm_residual_prediction(X_test, blstm_model, y_scaler)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 1)

def test_get_prediction(sample_data, mock_models, mocker):
    """Test complete prediction process."""
    sarimax_model, blstm_model, X_scaler, y_scaler = mock_models

    # Extend sample data to 10 rows
    extended_data = {
        'USD_Buying': [300, 305, 302, 308, 310, 315, 320, 325, 330, 335],
        'Crude_Oil_Price_LKR': [8000, 8100, 8050, 8150, 8200, 8250, 8300, 8350, 8400, 8450],
        'Week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Auction_Number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'SARIMAX_Predicted': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]
    }
    sample_data_with_price = pd.DataFrame(extended_data)
    sample_data_with_price['Price'] = [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]

    # Mock the data loading and model loading functions
    mocker.patch('Service.predictionService.load_and_filter_data', return_value=sample_data_with_price)
    mocker.patch('Service.predictionService.load_all_models', return_value=(sarimax_model, blstm_model, X_scaler, y_scaler))

    # Create a DataFrame for prepare_data_for_blstm
    prepared_df = pd.DataFrame({
        'Price': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145],
        'SARIMAX_Predicted': [101, 111, 106, 116, 121, 126, 131, 136, 141, 146],
        'Predicted_Residuals': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'Auction_Number': list(range(1, 11))
    }, index=pd.date_range(start='2023-01-01', periods=10, freq='D'))

    # Mock prepare_data_for_blstm to return correct length data
    def mock_prepare_data(*args, **kwargs):
        return np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), prepared_df

    mocker.patch('Service.predictionService.prepare_data_for_blstm', side_effect=mock_prepare_data)

    # Mock the BLSTM prediction to return correct length
    mocker.patch.object(blstm_model, 'predict', return_value=np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]))

    # Create a final prediction DataFrame
    final_df = pd.DataFrame({
        'Price': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145],
        'SARIMAX_Predicted': [101, 111, 106, 116, 121, 126, 131, 136, 141, 146],
        'Predicted_Residuals': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'Final_Prediction': [101.1, 111.2, 106.3, 116.4, 121.5, 126.6, 131.7, 136.8, 141.9, 147.0],
        'Auction_Number': list(range(1, 11))
    }, index=pd.date_range(start='2023-01-01', periods=10, freq='D'))

    # Mock DataFrame creation
    original_df = pd.DataFrame
    def mock_df(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict):
            return original_df(args[0])
        if 'index' in kwargs:
            return final_df
        return final_df

    mocker.patch('Service.predictionService.pd.DataFrame', side_effect=mock_df)

    result = get_prediction()
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 10
    assert all('Auction_Number' in r and 'Final_Prediction' in r for r in result)

def test_get_prediction_error_handling(sample_data, mock_models, mocker):
    """Test error handling in prediction process."""
    sarimax_model, blstm_model, X_scaler, y_scaler = mock_models
    
    # Mock the data loading to raise an exception
    mocker.patch('Service.predictionService.load_and_filter_data', side_effect=Exception("Test error"))
    
    with pytest.raises(Exception, match="Test error"):
        get_prediction()

def test_get_prediction_length_mismatch(sample_data, mock_models, mocker):
    """Test handling of length mismatch between predictions."""
    sarimax_model, blstm_model, X_scaler, y_scaler = mock_models

    # Add Price column to sample data
    sample_data_with_price = sample_data.copy()
    sample_data_with_price['Price'] = [100, 110, 105, 115, 120]

    # Mock the data loading and model loading functions
    mocker.patch('Service.predictionService.load_and_filter_data', return_value=sample_data_with_price)
    mocker.patch('Service.predictionService.load_all_models', return_value=(sarimax_model, blstm_model, X_scaler, y_scaler))

    # Mock prepare_data_for_blstm to return mismatched length data
    def mock_prepare_data(*args, **kwargs):
        return np.array([[1, 2]]), pd.DataFrame({
            'Price': [100, 110],
            'SARIMAX_Predicted': [101, 111],
            'Predicted_Residuals': [0.1, 0.2]
        }, index=pd.date_range(start='2023-01-01', periods=2, freq='D'))
    mocker.patch('Service.predictionService.prepare_data_for_blstm', side_effect=mock_prepare_data)

    # Mock the BLSTM prediction to return different length
    mocker.patch.object(blstm_model, 'predict', return_value=np.array([[0.1], [0.2], [0.3]]))

    # The function should raise a ValueError when there's a length mismatch
    with pytest.raises(ValueError, match="Length mismatch: Residuals .* and SARIMAX predictions"):
        get_prediction()