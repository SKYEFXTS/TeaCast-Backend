# TeaCast API

TeaCast is a sophisticated API service designed for tea price prediction and analysis. It leverages machine learning models to forecast tea auction prices and provides comprehensive dashboard analytics for the tea industry.

## Features

- Tea price prediction using hybrid ML models (BLSTM, SARIMAX)
- Authentication system for secure API access
- Tea auction price analysis and historical data
- Interactive dashboard with real-time analytics
- CORS-enabled API endpoints for frontend integration

## Project Structure

```
TeaCast API/
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
├── Controller/           # API route controllers
├── Model/               # ML models and scalers
├── Service/             # Business logic layer
├── Data/                # Data processing and storage
├── Utilities/           # Helper functions
├── templates/           # HTML templates
└── static/             # Static assets
```

## Prerequisites

- Python 3.x
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TeaCast-API
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Flask - Web framework
- joblib - Model persistence
- numpy - Numerical computing
- scikit-learn - Machine learning utilities
- tensorflow - Deep learning framework

## API Endpoints

### Authentication
- `/auth/*` - Authentication-related endpoints

### Data and Predictions
- `/data/*` - Data analysis and prediction endpoints
  - Tea price predictions
  - Auction price analysis
  - Dashboard analytics

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:3000` (Frontend development server)

## Models

The project includes several machine learning models:
- BLSTM Model (v3.0)
- SARIMAX Model

## Data Processing

The system includes comprehensive data preprocessing pipelines:
- Data loading and validation
- Feature engineering
- Model scaling and normalization

## Development

The project follows a modular architecture:
- Controllers handle HTTP requests
- Services contain business logic
- Models manage ML predictions
- Utilities provide helper functions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TeaCast Development Team
- Sri Lankan Tea Industry Stakeholders
- Open Source Community

## Contact

For any queries or support, please contact:
- Email: oshannr@gmail.com
- Project Link: [https://github.com/SKYEFXTS/TeaCast-Backend](https://github.com/SKYEFXTS/TeaCast-Backend)