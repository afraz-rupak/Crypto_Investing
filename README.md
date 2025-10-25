# Bitcoin Price Prediction API 🚀

Machine Learning API for predicting Bitcoin HIGH price for the next day using LightGBM model.

## 📋 Project Overview

This FastAPI application provides real-time Bitcoin price predictions using advanced machine learning techniques. The model predicts the **HIGH price** for the next trading day based on historical price data and technical indicators.

## 🎯 Objectives

- Provide real-time Bitcoin price predictions via REST API
- Predict the HIGH price for the next trading day
- Use 60+ engineered features including technical indicators and price patterns
- Deploy as a containerized Docker application

## 🏗️ Project Structure

```
Crypto_Investing/
├── app/
│   └── main.py              # FastAPI application
├── models/                   # Trained ML models
│   ├── bitcoin_lightgbm_model.pkl
│   ├── bitcoin_scaler_lightgbm.pkl
│   ├── feature_columns_lightgbm.pkl
│   └── model_metadata_lightgbm.pkl
├── data/                     # Data storage
├── notebooks/                # Jupyter notebooks for training
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── Dockerfile               # Docker configuration
└── github.txt               # GitHub repository link
```

## 📡 API Endpoints

### `GET /`
Returns project information, API documentation, and endpoint descriptions.

**Response:**
```json
{
  "project": "Bitcoin Price Prediction API",
  "description": "...",
  "endpoints": {...},
  "model_info": {...}
}
```

### `GET /health/`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Welcome to Bitcoin Price Prediction API! 🚀",
  "model_status": "loaded"
}
```

### `GET /predict/{token}`
Get prediction for token's HIGH price tomorrow.

**Parameters:**
- `token`: Token symbol (e.g., 'bitcoin', 'btc')

**Example:**
```bash
curl http://localhost:8000/predict/bitcoin
```

**Response:**
```json
{
  "token": "Bitcoin (BTC-USD)",
  "prediction": {
    "predicted_high_price": 67500.25,
    "prediction_date": "2025-10-26",
    "predicted_change_from_current": "+2.5%"
  },
  "current_data": {
    "current_close_price": 66000.00,
    "current_high_price": 66500.00,
    "current_date": "2025-10-25"
  },
  "model_info": {
    "model_type": "LightGBM",
    "features_used": 65,
    "data_source": "Yahoo Finance"
  }
}
```

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the API:**
```bash
uvicorn app.main:app --reload
```

3. **Access the API:**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t bitcoin-prediction-api .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 bitcoin-prediction-api
```

3. **Test the API:**
```bash
curl http://localhost:8000/health/
curl http://localhost:8000/predict/bitcoin
```

## 📊 Model Information

- **Algorithm:** LightGBM (Gradient Boosting)
- **Target:** Next day's HIGH price
- **Features:** 60+ engineered features including:
  - Price-based features (returns, volatility)
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Volume features
  - Lagged features
- **Data Source:** Yahoo Finance (BTC-USD)

## 🔧 Development

### Training New Models

Check the `notebooks/` directory for model training notebooks:
- `bitcoin_data_collection.ipynb` - Data fetching
- `bitcoin_feature_engineering.ipynb` - Feature creation
- `bitcoin_model_lightgbm_FIXED.ipynb` - Model training

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health/

# Test prediction endpoint
curl http://localhost:8000/predict/bitcoin

# View API documentation
open http://localhost:8000/docs
```

## 📦 Dependencies

Key packages:
- FastAPI - Web framework
- Uvicorn - ASGI server
- LightGBM - ML model
- yfinance - Data fetching
- pandas, numpy - Data processing
- scikit-learn - Preprocessing

## 📝 License

See LICENSE file for details.

## 👥 Authors

Crypto Investing Team

## 🔗 Links

- GitHub: [See github.txt](github.txt)
- API Documentation: http://localhost:8000/docs (when running)

## 📞 Support

For issues or questions, please open an issue on GitHub.

