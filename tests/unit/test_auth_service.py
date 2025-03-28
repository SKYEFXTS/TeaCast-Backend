"""
Unit tests for the authentication service module.
"""
import pytest
from Service.authService import authenticate_user

def test_authenticate_user_admin():
    """Test authentication for admin user."""
    result = authenticate_user('admin', 'admin')
    
    assert result is not None
    assert result['username'] == 'admin'
    assert result['role'] == 'admin'

def test_authenticate_user_regular():
    """Test authentication for regular user."""
    result = authenticate_user('user', 'user')
    
    assert result is not None
    assert result['username'] == 'user'
    assert result['role'] == 'user'

def test_authenticate_user_invalid_username():
    """Test authentication with invalid username."""
    result = authenticate_user('invalid', 'admin')
    
    assert result is None

def test_authenticate_user_invalid_password():
    """Test authentication with invalid password."""
    result = authenticate_user('admin', 'wrong_password')
    
    assert result is None

def test_authenticate_user_empty_credentials():
    """Test authentication with empty credentials."""
    result = authenticate_user('', '')
    
    assert result is None

def test_authenticate_user_none_credentials():
    """Test authentication with None credentials."""
    result = authenticate_user(None, None)
    
    assert result is None 