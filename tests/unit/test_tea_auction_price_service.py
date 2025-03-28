"""
Unit tests for the tea auction price service module.
"""
import pytest
import pandas as pd
from datetime import datetime
from Service.teaAuctionPriceService import get_last_auctions_average_prices

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Price': [100, 110, 90, 105, 95]
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_get_last_auctions_average_prices(sample_data, mocker):
    """Test getting average prices for last auctions."""
    # Mock the data loading function to return filtered data
    def mock_load_data(category, grade, _):
        if category == "WESTERN HIGH" and grade == "BOPF/BOPFSp":
            return sample_data.iloc[[0, 3]]
        elif category == "WESTERN HIGH" and grade == "BOP":
            return sample_data.iloc[[1]]
        elif category == "LOW GROWNS" and grade == "FBOPF1":
            return sample_data.iloc[[2, 4]]
        return None
    
    mocker.patch('Service.teaAuctionPriceService.load_and_filter_data', side_effect=mock_load_data)
    
    result = get_last_auctions_average_prices()
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(item, dict) for item in result)
    assert all(key in item for item in result for key in ['name', 'date', 'price'])

def test_get_last_auctions_average_prices_empty_data(mocker):
    """Test handling of empty data."""
    # Mock to return empty DataFrame
    mocker.patch('Service.teaAuctionPriceService.load_and_filter_data', return_value=pd.DataFrame())
    
    result = get_last_auctions_average_prices()
    
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error'] == 'Data not available for one or more categories'

def test_get_last_auctions_average_prices_missing_data(mocker):
    """Test handling of missing data for some categories."""
    # Mock to return None for some categories
    def mock_load_data(category, grade, _):
        if category == 'WESTERN HIGH' and grade == 'BOPF/BOPFSp':
            return None
        dates = pd.date_range(start='2023-01-01', periods=1)
        return pd.DataFrame({'Price': [100]}, index=dates)
    
    mocker.patch('Service.teaAuctionPriceService.load_and_filter_data', side_effect=mock_load_data)
    
    result = get_last_auctions_average_prices()
    
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error'] == 'Data not available for one or more categories'

def test_get_last_auctions_average_prices_data_format(sample_data, mocker):
    """Test the format of returned data."""
    def mock_load_data(category, grade, _):
        if category == "WESTERN HIGH" and grade == "BOPF/BOPFSp":
            return sample_data.iloc[[0, 3]]
        elif category == "WESTERN HIGH" and grade == "BOP":
            return sample_data.iloc[[1]]
        elif category == "LOW GROWNS" and grade == "FBOPF1":
            return sample_data.iloc[[2, 4]]
        return None
    
    mocker.patch('Service.teaAuctionPriceService.load_and_filter_data', side_effect=mock_load_data)
    
    result = get_last_auctions_average_prices()
    
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        assert 'name' in item
        assert 'date' in item
        assert 'price' in item
        assert isinstance(item['date'], str)
        assert isinstance(item['price'], str)
        assert item['price'].startswith('LKR')
        assert item['price'].endswith('/kg')

def test_get_last_auctions_average_prices_error_handling(mocker):
    """Test error handling in the service."""
    # Mock to raise an exception
    mocker.patch(
        'Service.teaAuctionPriceService.load_and_filter_data',
        side_effect=Exception("Test error")
    )
    
    with pytest.raises(Exception):
        get_last_auctions_average_prices() 