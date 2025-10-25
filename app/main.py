"""
FastAPI application for Bitcoin price prediction.

This application provides endpoints for:
- Project information and API documentation
- Health check
- Bitcoin HIGH price prediction for the next day
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os
import logging
from pathlib import Path
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bitcoin Price Prediction API",
    description="Machine Learning API for predicting Bitcoin HIGH price for the next day",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None
feature_columns = None
model_metadata = None

# Model paths - USING RETURN-BASED MODEL (FIXED)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "bitcoin_return_model.pkl"
SCALER_PATH = MODEL_DIR / "bitcoin_scaler_return.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns_return.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata_return.pkl"


def load_models():
    """Load the trained return-based model, scaler, and feature columns."""
    global model, scaler, feature_columns, model_metadata
    
    try:
        logger.info("Loading return-based RandomForest model...")
        
        # Load model (should be RandomForestRegressor)
        model = joblib.load(BEST_MODEL_PATH)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # Verify it's the correct type
        if not hasattr(model, 'predict'):
            logger.error(f"Model file contains {type(model)}, not a trained model!")
            return False
        
        # Load scaler (should be RobustScaler)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded: {type(scaler).__name__}")
        
        # Load feature columns
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info(f"Feature columns loaded: {len(feature_columns)} features")
        
        # Load metadata
        try:
            model_metadata = joblib.load(METADATA_PATH)
            logger.info(f"Model metadata loaded")
            logger.info(f"   Prediction type: {model_metadata.get('prediction_type', 'unknown')}")
            logger.info(f"   Test R2: {model_metadata.get('test_r2', 'N/A')}")
        except:
            model_metadata = {'algorithm': 'RandomForest', 'prediction_type': 'returns'}
            logger.warning("Metadata not found, using defaults")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def fetch_latest_bitcoin_data(days=60):
    """
    Fetch latest Bitcoin data from Yahoo Finance.
    
    Args:
        days: Number of days to fetch (default 60 for feature engineering)
    
    Returns:
        DataFrame with Bitcoin OHLCV data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching Bitcoin data from {start_date.date()} to {end_date.date()}")
        
        # Download data
        btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d', progress=False)
        
        if btc.empty:
            raise Exception("No data received from Yahoo Finance")
        
        # Flatten MultiIndex columns if present
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        
        # Process data
        df = btc.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Fetched {len(df)} days of data")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Bitcoin data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Bitcoin data: {str(e)}")


def engineer_features(df):
    """
    Engineer features from raw Bitcoin data - MATCHING RETURN-BASED MODEL.
    
    This creates the same features used during return-based model training:
    - Lag features (high, close, returns)
    - Moving averages (SMA, EMA, STD)
    - Price position features
    - Volatility
    - RSI
    - Time features
    """
    df = df.copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['high'].pct_change(periods=lag)
    
    # Moving averages
    for window in [7, 14, 30]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        df[f'std_{window}'] = df['close'].rolling(window=window).std()
    
    # Price position (normalized) - FIX: Ensure proper division
    df['price_to_sma_7'] = (df['close'] / df['sma_7']) - 1
    df['price_to_sma_30'] = (df['close'] / df['sma_30']) - 1
    
    # Volatility
    df['volatility_7'] = df['close'].pct_change().rolling(7).std()
    df['volatility_14'] = df['close'].pct_change().rolling(14).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Time features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    return df


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting up FastAPI application...")
    success = load_models()
    if not success:
        logger.error("Failed to load models on startup!")
    else:
        logger.info("All models loaded successfully!")


@app.get("/")
async def root():
    """
    Root endpoint - Project information and API documentation.
    
    Returns:
        JSON with project description, endpoints, and usage information
    """
    return {
        "project": "Bitcoin Price Prediction API",
        "description": "Machine Learning API for predicting Bitcoin HIGH price for the next day using Return-Based RandomForest model",
        "objectives": [
            "Provide real-time Bitcoin price predictions",
            "Predict the HIGH price for the next trading day",
            "Use return-based prediction (percentage change) for better accuracy",
            "Convert predicted returns to actual price predictions"
        ],
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Project information and API documentation"
            },
            "/health/": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/predict/{token}": {
                "method": "GET",
                "description": "Get prediction for token's HIGH price tomorrow",
                "parameters": {
                    "token": "Token symbol (e.g., 'bitcoin', 'btc')"
                },
                "example": "/predict/bitcoin"
            }
        },
        "model_info": {
            "algorithm": "RandomForest (Return-Based)",
            "prediction_type": "Percentage return (converted to price)",
            "features": f"{len(feature_columns) if feature_columns else 'N/A'} engineered features",
            "target": "Percentage change in HIGH price, then converted to absolute price",
            "formula": "predicted_high = current_high * (1 + predicted_return)",
            "metrics": model_metadata if model_metadata else "Not available"
        },
        "input_format": {
            "description": "No input required - API automatically fetches latest Bitcoin data",
            "data_source": "Yahoo Finance (BTC-USD)"
        },
        "output_format": {
            "predicted_high_price": "float - Predicted HIGH price in USD for tomorrow",
            "predicted_return_pct": "float - Predicted percentage change",
            "current_price": "float - Current close price in USD",
            "current_high": "float - Today's high price in USD",
            "prediction_date": "string - Date for which prediction is made",
            "features_used": "int - Number of features used in prediction"
        },
        "github": "https://github.com/YOUR_USERNAME/Crypto_Investing",
        "version": "1.0.0",
        "last_updated": "2025-10-25"
    }


@app.get("/health/")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status 200 with welcome message and model status
    """
    model_status = "loaded" if model is not None else "not loaded"
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Welcome to Bitcoin Price Prediction API!",
            "model_status": model_status,
            "model_type": "Return-Based RandomForest",
            "timestamp": datetime.now().isoformat(),
            "service": "running"
        }
    )


@app.get("/predict/{token}")
async def predict(token: str):
    """
    Predict HIGH price for the next day using return-based model.
    
    The model predicts the PERCENTAGE CHANGE (return) and converts it to price:
    predicted_high = current_high * (1 + predicted_return)
    
    Args:
        token: Token symbol (e.g., 'bitcoin', 'btc')
    
    Returns:
        JSON with prediction and metadata
    """
    # Validate token
    token_lower = token.lower()
    if token_lower not in ['bitcoin', 'btc', 'btc-usd']:
        raise HTTPException(
            status_code=400,
            detail=f"Token '{token}' not supported. Currently only 'bitcoin' or 'btc' is supported."
        )
    
    # Check if model is loaded
    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Fetch latest data
        logger.info(f"Fetching latest Bitcoin data for prediction...")
        df = fetch_latest_bitcoin_data(days=60)
        
        # Engineer features
        logger.info("Engineering features...")
        df_features = engineer_features(df)
        
        # Get the latest row (most recent data)
        latest_data = df_features.iloc[-1:].copy()
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(df_features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features as NaN
            for feat in missing_features:
                latest_data[feat] = np.nan
        
        # Select only the features used during training
        X = latest_data[feature_columns].copy()
        
        # Handle any remaining NaN values
        X = X.ffill().bfill().fillna(0)
        
        # Scale features
        logger.info("Scaling features...")
        X_scaled = scaler.transform(X)
        
        # Make prediction - THIS PREDICTS RETURN (percentage change)
        logger.info("Making prediction (return-based)...")
        predicted_return = model.predict(X_scaled)[0]
        
        # Get current data
        current_close = float(df.iloc[-1]['close'])
        current_high = float(df.iloc[-1]['high'])
        current_date = df.iloc[-1]['timestamp']
        prediction_date = (pd.to_datetime(current_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Convert return to PRICE
        # Formula: predicted_high = current_high * (1 + predicted_return)
        predicted_high_price = current_high * (1 + predicted_return)
        
        # Calculate percentage change for display
        predicted_return_pct = predicted_return * 100
        predicted_change_from_close_pct = ((predicted_high_price - current_close) / current_close) * 100
        
        response = {
            "token": "Bitcoin (BTC-USD)",
            "prediction": {
                "predicted_high_price": round(float(predicted_high_price), 2),
                "predicted_return_pct": round(float(predicted_return_pct), 4),
                "prediction_date": prediction_date,
                "predicted_change_from_close": f"{predicted_change_from_close_pct:+.2f}%",
                "formula": f"predicted_high = current_high * (1 + {predicted_return:.6f})"
            },
            "current_data": {
                "current_close_price": round(current_close, 2),
                "current_high_price": round(current_high, 2),
                "current_date": current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
            },
            "model_info": {
                "model_type": "RandomForest (Return-Based)",
                "prediction_type": "Percentage return → Price",
                "features_used": len(feature_columns),
                "data_source": "Yahoo Finance",
                "algorithm": model_metadata.get('algorithm', 'RandomForest') if model_metadata else 'RandomForest'
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: Return={predicted_return:.4%}, Price=${predicted_high_price:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get detailed model information.
    
    Returns:
        JSON with model metadata and performance metrics
    """
    if model_metadata is None:
        model_metadata_display = {
            "algorithm": "RandomForest",
            "prediction_type": "returns",
            "note": "Metadata file not found"
        }
    else:
        model_metadata_display = model_metadata
    
    return {
        "model_metadata": model_metadata_display,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "feature_list": feature_columns if feature_columns else [],
        "model_loaded": model is not None,
        "prediction_method": "Return-based: predicts % change, converts to price"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
