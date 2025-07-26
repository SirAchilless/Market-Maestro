"""
Simple Price Model Training Script
Train the price prediction model with a focused approach
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def get_technical_indicators(data):
    """Calculate technical indicators for price prediction"""
    df = data.copy()
    
    # Simple moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5'] = df['Close'].pct_change(periods=5)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # High-Low spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # RSI approximation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def create_features_and_targets(symbol, periods='1y'):
    """Create features and targets for training"""
    try:
        # Download stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period=periods)
        
        if len(data) < 50:
            return None, None
        
        # Calculate technical indicators
        data = get_technical_indicators(data)
        
        # Create features
        feature_columns = ['SMA_5', 'SMA_10', 'SMA_20', 'Price_Change', 'Price_Change_5', 
                          'Volume_Ratio', 'Volatility', 'HL_Spread', 'RSI']
        
        # Target: next day's close price
        data['Target'] = data['Close'].shift(-1)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        if len(data) < 30:
            return None, None
        
        X = data[feature_columns].values
        y = data['Target'].values
        
        return X, y
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None, None

def train_price_model():
    """Train the price prediction model"""
    print("Training Price Prediction Model...")
    
    # Training stocks (major Indian stocks)
    training_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS',
        'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'HDFCBANK.NS', 'LT.NS',
        'WIPRO.NS', 'MARUTI.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS'
    ]
    
    all_X = []
    all_y = []
    
    print(f"Processing {len(training_stocks)} stocks...")
    
    for i, symbol in enumerate(training_stocks):
        print(f"Processing {symbol} ({i+1}/{len(training_stocks)})")
        
        X, y = create_features_and_targets(symbol)
        
        if X is not None and y is not None:
            all_X.extend(X)
            all_y.extend(y)
            print(f"  Added {len(X)} samples")
        else:
            print(f"  Skipped {symbol} - insufficient data")
    
    if len(all_X) == 0:
        print("No training data available!")
        return False
    
    # Convert to arrays
    X = np.array(all_X)
    y = np.array(all_y)
    
    print(f"Total training samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Ensemble prediction
    ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Save models
    model_data = {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'feature_columns': ['SMA_5', 'SMA_10', 'SMA_20', 'Price_Change', 'Price_Change_5', 
                           'Volume_Ratio', 'Volatility', 'HL_Spread', 'RSI'],
        'performance_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        },
        'is_trained': True
    }
    
    # Save to models directory
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/price_predictor.joblib')
    
    print("✅ Price prediction model saved successfully!")
    return True

if __name__ == "__main__":
    success = train_price_model()
    if success:
        print("Price model training completed successfully!")
    else:
        print("Price model training failed!")