"""
Bitcoin Data Fetching Script
Fetches 5 years of daily OHLC data from Kraken API
Author: AT3 Assignment
Date: 2025-10-25
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os

def fetch_kraken_ohlc(pair='XBTUSD', interval=1440, since=None):
    """
    Fetch OHLC data from Kraken API
    
    Parameters:
    - pair: Trading pair (XBTUSD for Bitcoin/USD)
    - interval: Time frame interval in minutes (1440 = 1 day)
    - since: Unix timestamp to fetch data from
    
    Returns:
    - DataFrame with OHLC data
    """
    url = 'https://api.kraken.com/0/public/OHLC'
    
    params = {
        'pair': pair,
        'interval': interval
    }
    
    if since:
        params['since'] = since
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['error']:
            print(f"API Error: {data['error']}")
            return None, None
            
        # Extract the OHLC data
        pair_data = data['result'][list(data['result'].keys())[0]]
        last_timestamp = data['result']['last']
        
        return pair_data, last_timestamp
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None

def fetch_historical_bitcoin_data(years=5):
    """
    Fetch historical Bitcoin data for specified number of years
    
    Parameters:
    - years: Number of years of historical data to fetch
    
    Returns:
    - DataFrame with complete historical data
    """
    print(f"Fetching {years} years of Bitcoin OHLC data from Kraken...")
    
    # Calculate the starting timestamp (5 years ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    since = int(start_date.timestamp())
    
    all_data = []
    iteration = 0
    
    while True:
        iteration += 1
        print(f"Fetching batch {iteration}... (since: {datetime.fromtimestamp(since)})")
        
        ohlc_data, last_timestamp = fetch_kraken_ohlc(since=since)
        
        if ohlc_data is None:
            print("Failed to fetch data")
            break
            
        all_data.extend(ohlc_data)
        
        # Check if we've reached the end
        if last_timestamp == since or len(ohlc_data) < 720:  # 720 is typical batch size
            print("Reached end of available data")
            break
            
        since = last_timestamp
        
        # Be respectful to the API - add a small delay
        time.sleep(1)
        
        # Safety check - don't fetch more than we need
        if iteration > 100:  # Should be way more than enough for 5 years
            print("Reached iteration limit")
            break
    
    print(f"Total data points fetched: {len(all_data)}")
    
    return all_data

def process_bitcoin_data(raw_data):
    """
    Process raw OHLC data into a clean DataFrame
    
    Parameters:
    - raw_data: Raw OHLC data from Kraken API
    
    Returns:
    - Cleaned pandas DataFrame
    """
    # Column names based on Kraken API documentation
    columns = [
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'vwap',
        'volume',
        'count'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Convert price columns to float
    price_columns = ['open', 'high', 'low', 'close', 'vwap', 'volume']
    for col in price_columns:
        df[col] = df[col].astype(float)
    
    df['count'] = df['count'].astype(int)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add additional columns for the assignment format
    df['name'] = 'Bitcoin'
    df['timeOpen'] = df['timestamp']
    df['timeClose'] = df['timestamp']
    df['timeHigh'] = df['timestamp']
    df['timeLow'] = df['timestamp']
    df['marketCap'] = None  # Not provided by this endpoint
    
    # Reorder columns to match assignment data dictionary
    final_columns = [
        'timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name',
        'open', 'high', 'low', 'close', 'volume', 'marketCap', 'timestamp'
    ]
    
    df = df[final_columns]
    
    return df

def main():
    """
    Main function to fetch and save Bitcoin data
    """
    print("="*60)
    print("Bitcoin Data Fetching Script")
    print("="*60)
    
    # Fetch 5 years of data
    raw_data = fetch_historical_bitcoin_data(years=5)
    
    if not raw_data:
        print("No data fetched. Exiting.")
        return
    
    # Process the data
    print("\nProcessing data...")
    df = process_bitcoin_data(raw_data)
    
    # Display basic information
    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Save to CSV
    output_file = 'bitcoin_5years_daily.csv'
    df.to_csv(output_file, index=False)
    file_size = os.path.getsize(output_file) / 1024
    print(f"\nData saved to: {output_file}")
    print(f"  File size: {file_size:.2f} KB")
    print(f"  Records: {len(df)}")
    
    # Also save as parquet for faster loading
    output_parquet = output_file.replace('.csv', '.parquet')
    df.to_parquet(output_parquet, index=False)
    file_size_parquet = os.path.getsize(output_parquet) / 1024
    print(f"Data also saved to: {output_parquet}")
    
    print("\n" + "="*60)
    print("Data fetching complete!")
    print("="*60)

if __name__ == "__main__":
    main()
