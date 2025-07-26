#!/usr/bin/env python3
"""Simple ML model trainer for quick setup"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import yfinance as yf

def create_simple_features(data):
    """Create basic features for ML model"""
    features = {}
    
    # Simple technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # Use last available values
    features['price'] = data['Close'].iloc[-1] if not data.empty else 0
    features['sma_20'] = data['SMA_20'].iloc[-1] if not data['SMA_20'].isna().all() else features['price']
    features['sma_50'] = data['SMA_50'].iloc[-1] if not data['SMA_50'].isna().all() else features['price']
    features['rsi'] = data['RSI'].iloc[-1] if not data['RSI'].isna().all() else 50
    features['volatility'] = data['Volatility'].iloc[-1] if not data['Volatility'].isna().all() else 0.02
    features['volume'] = data['Volume'].iloc[-1] if not data.empty else 0
    
    # Price ratios (safe division)
    features['price_sma20_ratio'] = features['price'] / features['sma_20'] if features['sma_20'] > 0 else 1
    features['price_sma50_ratio'] = features['price'] / features['sma_50'] if features['sma_50'] > 0 else 1
    features['sma_ratio'] = features['sma_20'] / features['sma_50'] if features['sma_50'] > 0 else 1
    
    return features

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_simple_model():
    """Train a basic ML model"""
    print("Starting simple ML model training...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Comprehensive stock list for training - using NIFTY 50 for robust training
    from utils.stock_lists import NIFTY_50
    stocks = [f"{stock}.NS" for stock in NIFTY_50[:30]]  # Top 30 stocks for faster training
    
    all_features = []
    all_labels = []
    
    for stock in stocks:
        try:
            print(f"Processing {stock}...")
            
            # Get stock data
            ticker = yf.Ticker(stock)
            data = ticker.history(period='1y')
            
            if len(data) < 60:  # Need minimum data
                continue
                
            # Create features for multiple time points
            for i in range(60, len(data) - 5):  # Leave 5 days for future return
                subset = data.iloc[:i+1].copy()
                features = create_simple_features(subset)
                
                # Calculate future return (5-day)
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5]
                future_return = (future_price - current_price) / current_price
                
                # Create label
                if future_return > 0.03:  # 3% gain
                    label = 1  # Buy
                elif future_return < -0.03:  # 3% loss
                    label = -1  # Sell
                else:
                    label = 0  # Hold
                
                all_features.append(list(features.values()))
                all_labels.append(label)
                
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue
    
    if not all_features:
        print("No training data available")
        return False
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Training with {len(X)} samples")
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_scaled, y)
    
    # Save model and scaler
    joblib.dump(model, 'models/trading_signal_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    feature_names = ['price', 'sma_20', 'sma_50', 'rsi', 'volatility', 'volume', 
                    'price_sma20_ratio', 'price_sma50_ratio', 'sma_ratio']
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("Model training completed successfully!")
    return True

if __name__ == "__main__":
    success = train_simple_model()
    print(f"Training result: {success}")