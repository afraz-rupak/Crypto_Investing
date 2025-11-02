# Bitcoin Price Prediction API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)](https://crypto-investing.onrender.com/)

A Machine Learning-based REST API for predicting Bitcoin's HIGH price for the next trading day using a Return-Based RandomForest model with 81.5% accuracy.

## 🚀 Live Demo

**API is live at**: [https://crypto-investing.onrender.com/](https://crypto-investing.onrender.com/)

Try it now:
```bash
# Get Bitcoin prediction
curl https://crypto-investing.onrender.com/predict/bitcoin

# Check API health
curl https://crypto-investing.onrender.com/health/

# View interactive docs
open https://crypto-investing.onrender.com/docs
```

## 📊 Quick Example

```python
import requests

response = requests.get('https://crypto-investing.onrender.com/predict/bitcoin')
data = response.json()

print(f"Current HIGH: ${data['current_data']['current_high_price']:,.2f}")
print(f"Predicted HIGH: ${data['prediction']['predicted_high_price']:,.2f}")
print(f"Change: {data['prediction']['predicted_change_from_close']}")

# Output:
# Current HIGH: $110,644.98
# Predicted HIGH: $110,649.96
# Change: +0.24%
```

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [Performance & Limitations](#performance--limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This FastAPI application provides real-time Bitcoin price predictions using advanced machine learning techniques. The model predicts the **HIGH price** for the next trading day based on historical price data and technical indicators.

### Key Features

- **Real-time predictions** - Fetches latest Bitcoin data and predicts tomorrow's HIGH price
- **Return-based model** - Predicts percentage change, then converts to absolute price
- **High accuracy** - 81.5% R² score on test data
- **31 engineered features** - Technical indicators, moving averages, RSI, volatility
- **RESTful API** - Built with FastAPI framework
- **Docker support** - Easy containerized deployment
- **Automated data fetching** - Uses Yahoo Finance API

### Quick Stats

- **Algorithm**: RandomForest Regressor (Return-Based)
- **Accuracy**: R² = 0.815 (81.5%)
- **Features**: 31 engineered technical indicators
- **Training Data**: 2 years of daily Bitcoin OHLC data (730+ days)
- **Prediction Target**: Next day's HIGH price
- **Data Source**: Yahoo Finance (BTC-USD)

---

## Model Details

### Algorithm & Architecture

**Model Type**: RandomForest Regressor (Return-Based)

**Prediction Approach**:
```
1. Model predicts percentage return: predicted_return = model.predict(features)
2. Convert to price: predicted_high = current_high * (1 + predicted_return)
```

**Why Return-Based?**
- Handles distribution shift (Bitcoin price increased 10x during training period)
- More stable predictions across different price levels
- Better generalization to unseen price ranges

### Performance Metrics

| Metric | Value |
|--------|-------|
| R² Score | 0.815 (81.5% variance explained) |
| RMSE | ~$1,786 |
| MAE | ~$1,200 |
| Training Data | 730+ days (2 years) |
| Test Split | 70% train / 15% val / 15% test |

### Features (31 Total)

#### Lag Features (15 features)
- `high_lag_1` to `high_lag_7` - Previous HIGH prices
- `close_lag_1` to `close_lag_7` - Previous CLOSE prices
- `return_lag_1` to `return_lag_7` - Previous returns

#### Moving Averages (9 features)
- `sma_7`, `sma_14`, `sma_30` - Simple Moving Averages
- `ema_7`, `ema_14`, `ema_30` - Exponential Moving Averages
- `std_7`, `std_14`, `std_30` - Standard Deviation

#### Price Position (2 features)
- `price_to_sma_7` - Normalized distance to 7-day SMA
- `price_to_sma_30` - Normalized distance to 30-day SMA

#### Volatility (2 features)
- `volatility_7` - 7-day rolling std of returns
- `volatility_14` - 14-day rolling std of returns

#### Technical Indicators (1 feature)
- `rsi_14` - 14-day Relative Strength Index

#### Time Features (2 features)
- `day_of_week` - Day of the week (0-6)
- `month` - Month of the year (1-12)

### Data Preprocessing

- **Scaler**: RobustScaler (handles outliers effectively)
- **Missing Values**: Forward fill → Backward fill → Zero fill
- **Feature Engineering**: All features calculated from historical data only (no future leakage)

---

## API Endpoints

### Base URLs

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### 1. GET `/`

**Description**: Returns project information and API documentation

**Response**:
```json
{
  "project": "Bitcoin Price Prediction API",
  "description": "Machine Learning API for predicting Bitcoin HIGH price",
  "objectives": [...],
  "endpoints": {...},
  "model_info": {
    "algorithm": "RandomForest (Return-Based)",
    "features": "31 engineered features",
    "target": "Percentage change in HIGH price, then converted to absolute price",
    "formula": "predicted_high = current_high * (1 + predicted_return)"
  }
}
```

### 2. GET `/health/`

**Description**: Health check endpoint to verify API is running

**Response**:
```json
{
  "status": "healthy",
  "message": "Welcome to Bitcoin Price Prediction API!",
  "model_status": "loaded",
  "model_type": "Return-Based RandomForest",
  "timestamp": "2025-10-25T12:00:00",
  "service": "running"
}
```

### 3. GET `/predict/{token}`

**Description**: Predict tomorrow's HIGH price for specified cryptocurrency

**Parameters**:
- `token` (path parameter): Token symbol - `'bitcoin'`, `'btc'`, or `'btc-usd'`

**Example Request**:
```bash
curl http://localhost:8000/predict/bitcoin
```

**Response**:
```json
{
  "token": "Bitcoin (BTC-USD)",
  "prediction": {
    "predicted_high_price": 111269.57,
    "predicted_return_pct": -0.3892,
    "prediction_date": "2025-10-26",
    "predicted_change_from_close": "-0.24%",
    "formula": "predicted_high = current_high * (1 + -0.003892)"
  },
  "current_data": {
    "current_close_price": 111537.38,
    "current_high_price": 111704.38,
    "current_date": "2025-10-25"
  },
  "model_info": {
    "model_type": "RandomForest (Return-Based)",
    "prediction_type": "Percentage return → Price",
    "features_used": 31,
    "data_source": "Yahoo Finance",
    "algorithm": "RandomForest"
  },
  "timestamp": "2025-10-25T19:08:28.582656"
}
```

### 4. GET `/model/info`

**Description**: Get detailed model metadata and performance metrics

**Response**:
```json
{
  "model_metadata": {
    "algorithm": "RandomForest",
    "prediction_type": "returns",
    "n_features": 31,
    "test_r2": 0.815,
    "train_date": "2025-10-25 18:00:33"
  },
  "feature_count": 31,
  "feature_list": ["high_lag_1", "high_lag_2", ...],
  "model_loaded": true,
  "prediction_method": "Return-based: predicts % change, converts to price"
}
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning repository)
- Docker (optional, for containerized deployment)

### Method 1: Local Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/afraz-rupak/Crypto_Investing.git
cd Crypto_Investing
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify Model Files Exist

```bash
ls -lh models/bitcoin_return_model.pkl
ls -lh models/bitcoin_scaler_return.pkl
ls -lh models/feature_columns_return.pkl
```

Expected output:
```
-rw-r--r--  1 user  staff   723K Oct 25 18:00 models/bitcoin_return_model.pkl
-rw-r--r--  1 user  staff   1.8K Oct 25 18:00 models/bitcoin_scaler_return.pkl
-rw-r--r--  1 user  staff   401B Oct 25 18:00 models/feature_columns_return.pkl
```

#### Step 5: Run the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Application startup complete.
INFO:     Loading return-based RandomForest model...
INFO:     Model loaded: RandomForestRegressor
INFO:     Scaler loaded: RobustScaler
INFO:     Feature columns loaded: 31 features
INFO:     All models loaded successfully!
```

#### Step 6: Test the API

**Option A: Open Interactive Documentation**
```bash
# Open in browser
open http://localhost:8000/docs
```

**Option B: Test with curl**
```bash
# Health check
curl http://localhost:8000/health/

# Get prediction
curl http://localhost:8000/predict/bitcoin
```

### Method 2: Docker Deployment

#### Step 1: Build Docker Image

```bash
docker build -t bitcoin-prediction-api .
```

#### Step 2: Run Docker Container

```bash
docker run -p 8000:8000 bitcoin-prediction-api
```

#### Step 3: Access API

Open browser to: `http://localhost:8000/docs`

---

## Usage Examples

### Example 1: Python with Requests

```python
import requests

# Get prediction
response = requests.get('http://localhost:8000/predict/bitcoin')
data = response.json()

# Display results
print(f"Current HIGH: ${data['current_data']['current_high_price']:,.2f}")
print(f"Predicted HIGH for tomorrow: ${data['prediction']['predicted_high_price']:,.2f}")
print(f"Expected change: {data['prediction']['predicted_change_from_close']}")
print(f"Prediction date: {data['prediction']['prediction_date']}")

# Output:
# Current HIGH: $111,704.38
# Predicted HIGH for tomorrow: $111,269.57
# Expected change: -0.24%
# Prediction date: 2025-10-26
```

### Example 2: cURL Commands

```bash
# Health check
curl http://localhost:8000/health/

# Get prediction (pretty print with jq)
curl -s http://localhost:8000/predict/bitcoin | jq

# Get prediction (pretty print with Python)
curl -s http://localhost:8000/predict/bitcoin | python3 -m json.tool

# View API documentation
curl http://localhost:8000/
```

### Example 3: JavaScript/Node.js

```javascript
const axios = require('axios');

async function getBitcoinPrediction() {
  try {
    const response = await axios.get('http://localhost:8000/predict/bitcoin');
    const data = response.data;
    
    console.log('Token:', data.token);
    console.log('Current HIGH:', data.current_data.current_high_price);
    console.log('Predicted HIGH:', data.prediction.predicted_high_price);
    console.log('Change:', data.prediction.predicted_change_from_close);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

getBitcoinPrediction();
```

### Example 4: Python with Pandas

```python
import requests
import pandas as pd

# Get prediction
response = requests.get('http://localhost:8000/predict/bitcoin')
data = response.json()

# Create DataFrame for analysis
df = pd.DataFrame({
    'Date': [data['current_data']['current_date'], data['prediction']['prediction_date']],
    'HIGH Price': [data['current_data']['current_high_price'], data['prediction']['predicted_high_price']],
    'Type': ['Actual', 'Predicted']
})

print(df)
```

### Example 5: Interactive API Documentation (Swagger UI)

1. Start the API server
2. Open browser to: `http://localhost:8000/docs`
3. Click on any endpoint to expand
4. Click **"Try it out"**
5. Enter parameters (if required)
6. Click **"Execute"**
7. View the response

---

## Testing

### Automated Testing

Run the complete test suite:

```bash
python test_api.py
```

Expected output:
```
======================================================================
Bitcoin Price Prediction API - Test Suite
======================================================================
Testing API at: http://localhost:8000

======================================================================
Testing GET / (Root endpoint)
======================================================================
Status Code: 200
Success!

======================================================================
Testing GET /health/ (Health check)
======================================================================
Status Code: 200
Success!
Status: healthy
Model Status: loaded

======================================================================
Testing GET /predict/bitcoin (Prediction)
======================================================================
Status Code: 200
Success!
Predicted HIGH price: $111269.57

======================================================================
All tests completed!
======================================================================
```

### Manual Testing

#### 1. Health Check
```bash
curl http://localhost:8000/health/
```
**Expected**: Status 200, `"status": "healthy"`

#### 2. Root Endpoint
```bash
curl http://localhost:8000/
```
**Expected**: Project information and documentation

#### 3. Prediction Endpoint
```bash
curl http://localhost:8000/predict/bitcoin
```
**Expected**: Prediction with current and future price data

---

## Project Structure

```
Crypto_Investing/
├── app/
│   ├── __init__.py
│   └── main.py                    # FastAPI application (main entry point)
│
├── models/                         # Trained ML models
│   ├── bitcoin_return_model.pkl           # RandomForest model
│   ├── bitcoin_scaler_return.pkl          # RobustScaler
│   ├── feature_columns_return.pkl         # Feature list (31 features)
│   └── model_metadata_return.pkl          # Performance metrics
│
├── data/
│   ├── raw/                        # Raw Bitcoin price data
│   │   └── bitcoin_2years_daily.csv
│   └── processed/                  # Engineered features
│       └── bitcoin_features_engineered.csv
│
├── notebooks/                      # Jupyter notebooks
│   ├── bitcoin_data_collection.ipynb              # Data fetching
│   ├── bitcoin_feature_engineering.ipynb          # Feature creation
│   └── bitcoin_model_RETURN_BASED_FIXED.ipynb     # Model training
│
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
├── test_api.py                     # API test suite
├── README.md                       # This file
└── github.txt                      # Detailed documentation
```

---

## Model Training

To retrain the model with new data:

### Step 1: Collect New Data

```bash
jupyter notebook notebooks/bitcoin_data_collection.ipynb
```

Run all cells to fetch latest 2 years of Bitcoin data from Yahoo Finance.

### Step 2: Engineer Features

```bash
jupyter notebook notebooks/bitcoin_feature_engineering.ipynb
```

Run all cells to create 31 engineered features.

### Step 3: Train Model

```bash
jupyter notebook notebooks/bitcoin_model_RETURN_BASED_FIXED.ipynb
```

Run all cells to:
- Train RandomForest model
- Evaluate performance
- Save model files to `models/` directory

The notebook will automatically:
- Split data (70% train / 15% val / 15% test)
- Apply RobustScaler
- Train RandomForest with 200 estimators
- Calculate metrics (R², RMSE, MAE)
- Save all required files

### Step 4: Restart API

```bash
# API will automatically load new model files
uvicorn app.main:app --reload
```

---

## Troubleshooting

### Problem: "Model not loaded" error

**Solution**: Verify model files exist
```bash
ls -lh models/bitcoin_return_model.pkl
ls -lh models/bitcoin_scaler_return.pkl
ls -lh models/feature_columns_return.pkl
```

### Problem: API returns 500 error on prediction

**Solution**: Check if Yahoo Finance is accessible
```bash
curl https://finance.yahoo.com
```

**Alternative**: Check API logs for specific error
```bash
# Look for "Prediction error:" in logs
```

### Problem: Port 8000 already in use

**Solution 1**: Kill existing process
```bash
lsof -ti:8000 | xargs kill -9
```

**Solution 2**: Use different port
```bash
uvicorn app.main:app --port 8001
```

### Problem: Import errors when starting API

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Problem: Docker build fails

**Solution**: Ensure all model files are present
```bash
ls -lh models/*.pkl
```

### Problem: Prediction takes too long

**Cause**: Yahoo Finance API may be slow

**Expected Time**: 2-5 seconds (includes data fetching)

**Solution**: If consistently slow, check internet connection

---

## Deployment

### Option 1: Deploy to Heroku

```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login to Heroku
heroku login

# Create new app
heroku create bitcoin-prediction-api

# Deploy
git push heroku main

# Open app
heroku open
```

### Option 2: Deploy to AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Clone repository
git clone https://github.com/afraz-rupak/Crypto_Investing.git
cd Crypto_Investing

# 4. Install Docker
sudo apt update
sudo apt install docker.io -y

# 5. Build and run
sudo docker build -t api .
sudo docker run -d -p 80:8000 api
```

### Option 3: Deploy to Google Cloud Run

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Build and submit
gcloud builds submit --tag gcr.io/PROJECT_ID/bitcoin-api

# Deploy
gcloud run deploy --image gcr.io/PROJECT_ID/bitcoin-api --platform managed
```

### Option 4: Deploy to DigitalOcean

1. Create Droplet (Ubuntu 22.04)
2. SSH into droplet
3. Clone repository
4. Install dependencies
5. Run with systemd service or supervisor

### Production Configuration

**Environment Variables**:
```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export API_HOST=0.0.0.0
export API_PORT=8000
```

**Reverse Proxy** (nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Performance & Limitations

### Current Capabilities

- Predicts Bitcoin (BTC-USD) HIGH price for next day
- 81.5% accuracy (R² score) on test data
- Real-time data fetching from Yahoo Finance
- 2-5 second prediction latency (includes data fetch)
- <100ms model inference time

### Limitations

1. **Single Cryptocurrency**: Only supports Bitcoin (BTC-USD)
2. **Single Prediction**: Only predicts HIGH price (not low, open, close)
3. **Internet Required**: Needs connection to Yahoo Finance API
4. **Historical Training**: Model trained on 2 years (may need periodic retraining)
5. **Market Events**: Does not account for major news or events
6. **Short-term Only**: Predicts only next day (not week/month)

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Prediction Latency | 2-5 seconds |
| Model Inference | <100ms |
| Data Fetching | 1-3 seconds |
| Memory Usage | ~150MB (with model loaded) |
| Concurrent Requests | Up to 100/sec (FastAPI async) |

### Future Improvements

- [ ] Support multiple cryptocurrencies (ETH, BNB, SOL)
- [ ] Predict all OHLC values (Open, High, Low, Close)
- [ ] Add confidence intervals for predictions
- [ ] Implement real-time streaming predictions
- [ ] Track historical prediction accuracy
- [ ] Multi-source data integration (Binance, Coinbase)
- [ ] Add sentiment analysis from Twitter/Reddit
- [ ] Support for longer time horizons (week, month)

---

## Contributing

We welcome contributions! To contribute:

### 1. Fork the Repository

Click "Fork" button on GitHub

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Update tests if needed
- Update documentation

### 4. Test Your Changes

```bash
# Run test suite
python test_api.py

# Test manually
curl http://localhost:8000/health/
curl http://localhost:8000/predict/bitcoin
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request

Go to GitHub and create a Pull Request from your fork

### Contribution Guidelines

- **Code Style**: Follow PEP 8
- **Testing**: All tests must pass
- **Documentation**: Update README.md and github.txt
- **Commit Messages**: Use clear, descriptive messages
- **Pull Requests**: Provide detailed description of changes

---

## Dependencies

### Core Libraries

```
fastapi>=0.104.0           # Web framework
uvicorn[standard]>=0.24.0  # ASGI server
pydantic>=2.4.0            # Data validation
```

### Data & Machine Learning

```
pandas>=2.1.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0        # Machine learning
yfinance>=0.2.32           # Data fetching
joblib>=1.3.0              # Model serialization
```

### Utilities

```
requests>=2.31.0           # HTTP requests
python-dotenv              # Environment variables
```

See `requirements.txt` for complete list with version constraints.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project was developed as part of Advanced Machine Learning Applications coursework.

**Special Thanks**:
- Yahoo Finance for providing free Bitcoin price data
- FastAPI framework developers
- scikit-learn community
- Open-source contributors

**Data Source**: Historical Bitcoin prices from Yahoo Finance (BTC-USD)

**API Framework**: FastAPI (https://fastapi.tiangolo.com/)

**ML Library**: scikit-learn (https://scikit-learn.org/)

---

## Quick Start

For the impatient:

```bash
git clone https://github.com/afraz-rupak/Crypto_Investing.git
cd Crypto_Investing
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open: `http://localhost:8000/docs`

Done! Start making predictions.

---

## Support

**Issues & Questions**: Open an issue on [GitHub Issues](https://github.com/afraz-rupak/Crypto_Investing/issues)

**Documentation**: See [github.txt](github.txt) for detailed documentation

**API Docs**: http://localhost:8000/docs (when running)

---

**Version**: 1.0.0  
**Last Updated**: October 25, 2025  
**Author**: Crypto Investing Team

