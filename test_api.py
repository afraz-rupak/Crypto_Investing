"""
Test script for Bitcoin Price Prediction API
Run this after starting the API to verify all endpoints work correctly.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    print("\n" + "="*70)
    print("Testing GET / (Root endpoint)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"\nProject: {data.get('project')}")
        print(f"Description: {data.get('description')}")
        print(f"\nEndpoints available:")
        for endpoint, info in data.get('endpoints', {}).items():
            print(f"  - {endpoint}: {info.get('description')}")
    else:
        print(f"❌ Failed: {response.text}")

def test_health():
    """Test the health check endpoint"""
    print("\n" + "="*70)
    print("Testing GET /health/ (Health check)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health/")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"\nStatus: {data.get('status')}")
        print(f"Message: {data.get('message')}")
        print(f"Model Status: {data.get('model_status')}")
    else:
        print(f"❌ Failed: {response.text}")

def test_predict():
    """Test the prediction endpoint"""
    print("\n" + "="*70)
    print("Testing GET /predict/bitcoin (Prediction)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/predict/bitcoin")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"\nToken: {data.get('token')}")
        
        prediction = data.get('prediction', {})
        print(f"\nPrediction:")
        print(f"  - Predicted HIGH price: ${prediction.get('predicted_high_price')}")
        print(f"  - Prediction date: {prediction.get('prediction_date')}")
        print(f"  - Change from current: {prediction.get('predicted_change_from_current')}")
        
        current = data.get('current_data', {})
        print(f"\nCurrent Data:")
        print(f"  - Close price: ${current.get('current_close_price')}")
        print(f"  - High price: ${current.get('current_high_price')}")
        print(f"  - Date: {current.get('current_date')}")
        
        model_info = data.get('model_info', {})
        print(f"\nModel Info:")
        print(f"  - Type: {model_info.get('model_type')}")
        print(f"  - Features used: {model_info.get('features_used')}")
    else:
        print(f"❌ Failed: {response.text}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Bitcoin Price Prediction API - Test Suite")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    
    try:
        test_root()
        test_health()
        test_predict()
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
