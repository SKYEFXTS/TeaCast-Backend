import pytest
from app import create_app

"""
To run tests with coverage from the terminal:

1. Install required packages (if not already installed):
   pip install pytest pytest-cov

2. Run tests with coverage:
   pytest --cov=. tests/
   
   For a more detailed report:
   pytest --cov=. --cov-report=term-missing tests/
   
   To generate an HTML report:
   pytest --cov=. --cov-report=html tests/
   
   To run a specific test file:
   pytest --cov=. tests/test_app.py
"""

@pytest.fixture
def client():
    """Create a test client for the app."""
    app = create_app(install_deps=False)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_app_creation():
    """Test that app is created successfully."""
    app = create_app(install_deps=False)
    assert app is not None

# Add more tests for your endpoints
