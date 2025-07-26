import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st
from datetime import datetime, timedelta
import os

from utils.technical_analysis import calculate_all_indicators
from utils.fundamental_analysis import get_fundamental_data
from utils.news_sentiment import get_news_sentiment_score

class MLSignalPredictor:
    """Machine Learning based trading signal predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_path = "models/trading_signal_model.joblib"
        self.scaler_path = "models/scaler.joblib"
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def create_features(self, stock_data, symbol):
        """
        Create feature set for ML model
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            symbol (str): Stock symbol
        
        Returns:
            dict: Feature dictionary
        """
        try:
            features = {}
            
            # Technical indicators
            indicators = calculate_all_indicators(stock_data)
            
            # Core technical features
            features['rsi'] = indicators.get('RSI', 50)
            features['macd'] = indicators.get('MACD', 0)
            features['macd_signal'] = indicators.get('MACD_Signal', 0)
            features['macd_histogram'] = indicators.get('MACD_Histogram', 0)
            features['bb_position'] = self._calculate_bb_position(indicators)
            features['volume_ratio'] = indicators.get('Volume_Ratio', 1)
            
            # Moving average features
            current_price = indicators.get('Current_Price', 0)
            ma20 = indicators.get('MA20', current_price)
            ma50 = indicators.get('MA50', current_price)  
            ma200 = indicators.get('MA200', current_price)
            
            features['price_vs_ma20'] = (current_price / ma20) - 1 if ma20 > 0 else 0
            features['price_vs_ma50'] = (current_price / ma50) - 1 if ma50 > 0 else 0
            features['price_vs_ma200'] = (current_price / ma200) - 1 if ma200 > 0 else 0
            
            # Price momentum features
            features['price_change_pct'] = indicators.get('Price_Change_Pct', 0)
            features['volatility'] = self._calculate_volatility(stock_data)
            
            # Support/Resistance features
            support = indicators.get('Support', 0)
            resistance = indicators.get('Resistance', 0)
            if support > 0 and current_price > 0:
                features['support_distance'] = (current_price - support) / support
            else:
                features['support_distance'] = 0
                
            if resistance > 0 and current_price > 0:
                features['resistance_distance'] = (resistance - current_price) / current_price
            else:
                features['resistance_distance'] = 0
            
            # Fundamental features
            fundamentals = get_fundamental_data(symbol)
            features['pe_ratio'] = fundamentals.get('PE Ratio', 20)
            features['roe'] = fundamentals.get('ROE', 0.15) * 100
            features['debt_equity'] = fundamentals.get('Debt to Equity', 0.5)
            features['profit_margin'] = fundamentals.get('Profit Margin', 0.1) * 100
            
            # News sentiment features
            sentiment_data = get_news_sentiment_score(symbol)
            features['news_sentiment'] = sentiment_data.get('sentiment_score', 50)
            features['news_confidence'] = sentiment_data.get('confidence', 0)
            features['news_article_count'] = min(sentiment_data.get('article_count', 0), 20)  # Cap at 20
            
            # Market timing features
            now = datetime.now()
            features['month'] = now.month
            features['day_of_week'] = now.weekday()
            features['quarter'] = (now.month - 1) // 3 + 1
            
            # Breakout features
            features['bullish_breakout'] = int(indicators.get('Bullish_Breakout', False))
            features['bearish_breakout'] = int(indicators.get('Bearish_Breakout', False))
            
            return features
            
        except Exception as e:
            st.error(f"Error creating features: {str(e)}")
            return {}
    
    def _calculate_bb_position(self, indicators):
        """Calculate position relative to Bollinger Bands"""
        try:
            current_price = indicators.get('Current_Price', 0)
            bb_upper = indicators.get('BB_Upper', 0)
            bb_lower = indicators.get('BB_Lower', 0)
            
            if bb_upper > bb_lower > 0:
                return (current_price - bb_lower) / (bb_upper - bb_lower)
            return 0.5
        except:
            return 0.5
    
    def _calculate_volatility(self, stock_data, window=20):
        """Calculate price volatility"""
        try:
            if len(stock_data) < window:
                return 0
            returns = stock_data['Close'].pct_change().tail(window)
            return returns.std() * 100
        except:
            return 0
    
    def prepare_training_data(self, stock_symbols, data_period='1y'):
        """
        Prepare training data from historical stock data
        
        Args:
            stock_symbols (list): List of stock symbols
            data_period (str): Data period for training
        
        Returns:
            tuple: (X, y) training data
        """
        try:
            from utils.data_fetcher import get_multiple_stocks_data
            
            all_features = []
            all_labels = []
            
            st.info(f"Preparing training data for {len(stock_symbols)} stocks...")
            
            # Get historical data for all stocks
            stocks_data = get_multiple_stocks_data(stock_symbols, period=data_period)
            
            for symbol, stock_data in stocks_data.items():
                if stock_data.empty or len(stock_data) < 100:
                    continue
                
                try:
                    # Create sliding window for training
                    for i in range(50, len(stock_data) - 5):  # Leave 5 days for future return
                        window_data = stock_data.iloc[:i+1].copy()
                        
                        # Create features for this window
                        features = self.create_features(window_data, symbol)
                        
                        if not features:
                            continue
                        
                        # Calculate future return (5-day forward return)
                        current_price = stock_data.iloc[i]['Close']
                        future_price = stock_data.iloc[i+5]['Close']
                        future_return = (future_price - current_price) / current_price
                        
                        # Create label based on future return
                        if future_return > 0.05:  # 5% gain
                            label = 2  # Strong Buy
                        elif future_return > 0.02:  # 2% gain
                            label = 1  # Buy
                        elif future_return < -0.05:  # 5% loss
                            label = -1  # Sell
                        else:
                            label = 0  # Hold
                        
                        all_features.append(features)
                        all_labels.append(label)
                
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {str(e)}")
                    continue
            
            if not all_features:
                st.error("No training data could be prepared")
                return None, None
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features)
            labels_array = np.array(all_labels)
            
            # Store feature columns
            self.feature_columns = features_df.columns.tolist()
            
            # Handle missing values
            features_df = features_df.fillna(features_df.median())
            
            st.success(f"Prepared {len(features_df)} training samples with {len(self.feature_columns)} features")
            
            return features_df.values, labels_array
            
        except Exception as e:
            st.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def train_model(self, stock_symbols):
        """
        Train the ML model
        
        Args:
            stock_symbols (list): List of stock symbols for training
        """
        try:
            st.info("Starting model training...")
            
            # Prepare training data
            X, y = self.prepare_training_data(stock_symbols)
            
            if X is None or len(X) == 0:
                st.error("No training data available")
                return False
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Train both models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)
            
            rf_accuracy = accuracy_score(y_test, rf_pred)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            
            # Choose the better model
            if rf_accuracy >= gb_accuracy:
                self.model = rf_model
                best_accuracy = rf_accuracy
                model_type = "Random Forest"
            else:
                self.model = gb_model
                best_accuracy = gb_accuracy
                model_type = "Gradient Boosting"
            
            # Save the model
            self.save_model()
            self.is_trained = True
            
            st.success(f"Model trained successfully!")
            st.info(f"Best model: {model_type} with accuracy: {best_accuracy:.3f}")
            
            # Show feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                st.dataframe(importance_df.head(10))
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def predict_signal(self, stock_data, symbol):
        """
        Predict trading signal for a stock
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            symbol (str): Stock symbol
        
        Returns:
            dict: Prediction results
        """
        try:
            if not self.is_trained or self.model is None:
                return {
                    'signal': 'Hold',
                    'confidence': 0,
                    'ml_score': 50,
                    'recommendation': 'Model not trained'
                }
            
            # Create features
            features = self.create_features(stock_data, symbol)
            
            if not features:
                return {
                    'signal': 'Hold',
                    'confidence': 0,
                    'ml_score': 50,
                    'recommendation': 'Insufficient data'
                }
            
            # Convert to DataFrame and ensure all features are present
            feature_df = pd.DataFrame([features])
            
            # Add missing features with default values
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Reorder columns to match training data
            feature_df = feature_df[self.feature_columns]
            
            # Handle missing values
            feature_df = feature_df.fillna(0)
            
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(feature_df.values)
            else:
                features_scaled = feature_df.values
            
            # Make prediction
            if self.model is None:
                raise ValueError("Model is not trained")
                
            predictions = self.model.predict(features_scaled)
            if predictions is None or len(predictions) == 0:
                raise ValueError("Model prediction failed")
            prediction = predictions[0]
            
            probabilities_array = self.model.predict_proba(features_scaled)
            if probabilities_array is None or len(probabilities_array) == 0:
                raise ValueError("Model probability prediction failed")
            probabilities = probabilities_array[0]
            
            # Convert prediction to signal
            signal_map = {-1: 'Sell', 0: 'Hold', 1: 'Buy', 2: 'Strong Buy'}
            signal = signal_map.get(prediction, 'Hold')
            
            # Calculate confidence
            if isinstance(probabilities, np.ndarray):
                confidence = np.max(probabilities)
            else:
                confidence = max(probabilities) if hasattr(probabilities, '__iter__') else 0.5
            
            # Calculate ML score (0-100)
            ml_score = (prediction + 1) * 25  # Convert -1,2 to 0,75 then scale
            ml_score = max(0, min(100, ml_score + (confidence - 0.25) * 50))
            
            # Generate recommendation
            if signal == 'Strong Buy' and confidence > 0.7:
                recommendation = f"Strong buy signal with {confidence:.1%} confidence"
            elif signal == 'Buy' and confidence > 0.6:
                recommendation = f"Buy signal with {confidence:.1%} confidence"
            elif signal == 'Sell' and confidence > 0.6:
                recommendation = f"Sell signal with {confidence:.1%} confidence"
            else:
                recommendation = f"Hold recommendation with {confidence:.1%} confidence"
            
            return {
                'signal': signal,
                'confidence': round(confidence, 3),
                'ml_score': round(ml_score, 1),
                'recommendation': recommendation,
                'raw_prediction': prediction,
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            st.error(f"Error predicting signal: {str(e)}")
            return {
                'signal': 'Hold',
                'confidence': 0,
                'ml_score': 50,
                'recommendation': f'Prediction error: {str(e)}'
            }
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            if self.model is not None:
                model_data = {
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'trained_date': datetime.now()
                }
                joblib.dump(model_data, self.model_path)
                
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
                
        except Exception as e:
            st.warning(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                self.is_trained = True
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                
        except Exception as e:
            st.warning(f"Error loading model: {str(e)}")
            self.is_trained = False

# Global ML predictor instance
ml_predictor = MLSignalPredictor()

def get_simple_ml_signal(stock_data, symbol):
    """Get ML signal using the simple trained model"""
    try:
        import joblib
        
        # Load simple model
        model = joblib.load("models/trading_signal_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        
        # Create simple features (matching train_ml_simple.py)
        data = stock_data.copy()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Extract features
        features = {}
        features['price'] = data['Close'].iloc[-1] if not data.empty else 0
        features['sma_20'] = data['SMA_20'].iloc[-1] if not data['SMA_20'].isna().all() else features['price']
        features['sma_50'] = data['SMA_50'].iloc[-1] if not data['SMA_50'].isna().all() else features['price']
        features['rsi'] = data['RSI'].iloc[-1] if not data['RSI'].isna().all() else 50
        features['volatility'] = data['Volatility'].iloc[-1] if not data['Volatility'].isna().all() else 0.02
        features['volume'] = data['Volume'].iloc[-1] if not data.empty else 0
        
        # Safe ratios
        features['price_sma20_ratio'] = features['price'] / features['sma_20'] if features['sma_20'] > 0 else 1
        features['price_sma50_ratio'] = features['price'] / features['sma_50'] if features['sma_50'] > 0 else 1
        features['sma_ratio'] = features['sma_20'] / features['sma_50'] if features['sma_50'] > 0 else 1
        
        # Convert to array and predict
        feature_array = np.array([[
            features['price'], features['sma_20'], features['sma_50'], 
            features['rsi'], features['volatility'], features['volume'],
            features['price_sma20_ratio'], features['price_sma50_ratio'], features['sma_ratio']
        ]])
        
        # Scale and predict
        feature_scaled = scaler.transform(feature_array)
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
        # Convert to signals
        signal_map = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
        signal = signal_map.get(prediction, 'Hold')
        confidence = max(probabilities)
        
        # Calculate ML score
        ml_score = 50 + (prediction * 25) + ((confidence - 0.33) * 50)
        ml_score = max(0, min(100, ml_score))
        
        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'ml_score': round(ml_score, 1),
            'recommendation': f'{signal} with {confidence:.1%} confidence'
        }
        
    except Exception as e:
        return {
            'signal': 'Hold',
            'confidence': 0,
            'ml_score': 50,
            'recommendation': f'Simple ML error: {str(e)}'
        }

def get_ml_signal(stock_data, symbol):
    """
    Get ML-based trading signal for a stock
    
    Args:
        stock_data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
    
    Returns:
        dict: ML prediction results
    """
    try:
        # Try simple model first
        if (os.path.exists("models/trading_signal_model.joblib") and 
            os.path.exists("models/scaler.joblib")):
            return get_simple_ml_signal(stock_data, symbol)
        else:
            # Fallback to complex model
            return ml_predictor.predict_signal(stock_data, symbol)
    except Exception as e:
        return {
            'signal': 'Hold',
            'confidence': 0,
            'ml_score': 50,
            'recommendation': f'ML prediction error: {str(e)}'
        }

def train_ml_model(stock_symbols):
    """
    Train the ML model with given stock symbols
    
    Args:
        stock_symbols (list): List of stock symbols for training
    
    Returns:
        bool: Success status
    """
    return ml_predictor.train_model(stock_symbols)

def is_model_trained():
    """Check if ML model is trained"""
    # Check for simple model files (from train_ml_simple.py)
    simple_model_exists = (
        os.path.exists("models/trading_signal_model.joblib") and 
        os.path.exists("models/scaler.joblib") and 
        os.path.exists("models/feature_names.txt")
    )
    
    # Also check complex model
    complex_model_exists = ml_predictor.is_trained
    
    return simple_model_exists or complex_model_exists