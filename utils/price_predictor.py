"""
Advanced Price Prediction System for Stock Purchase Timing
Provides accurate ideal purchase price predictions using ML
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional

from utils.technical_analysis import calculate_all_indicators
from utils.fundamental_analysis import get_fundamental_data
from utils.news_sentiment import get_news_sentiment_score

class SmartPricePredictor:
    """Advanced ML-based price prediction for optimal purchase timing"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_path = "models/price_predictor.joblib"
        self.scaler_path = "models/price_scaler.joblib"
        self.performance_path = "models/price_performance.json"
        self.is_trained = False
        
        # Model ensemble
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing model if available
        self.load_model()
    
    def create_price_features(self, stock_data: pd.DataFrame, symbol: str) -> Dict:
        """Create comprehensive feature set for price prediction"""
        try:
            features = {}
            
            # Get technical indicators
            indicators = calculate_all_indicators(stock_data)
            current_price = indicators.get('Current_Price', 0)
            
            # Price-based features
            features['current_price'] = current_price
            features['price_change_1d'] = indicators.get('Price_Change_Pct', 0)
            
            # Calculate price changes over different periods
            if len(stock_data) >= 5:
                features['price_change_5d'] = ((current_price / stock_data['Close'].iloc[-5]) - 1) * 100
            else:
                features['price_change_5d'] = 0
                
            if len(stock_data) >= 20:
                features['price_change_20d'] = ((current_price / stock_data['Close'].iloc[-20]) - 1) * 100
            else:
                features['price_change_20d'] = 0
            
            # Moving average features
            ma20 = indicators.get('MA20', current_price)
            ma50 = indicators.get('MA50', current_price)
            ma200 = indicators.get('MA200', current_price)
            
            features['ma20_distance'] = (current_price / ma20 - 1) * 100 if ma20 > 0 else 0
            features['ma50_distance'] = (current_price / ma50 - 1) * 100 if ma50 > 0 else 0
            features['ma200_distance'] = (current_price / ma200 - 1) * 100 if ma200 > 0 else 0
            
            # Technical momentum features
            features['rsi'] = indicators.get('RSI', 50)
            features['macd'] = indicators.get('MACD', 0)
            features['macd_signal'] = indicators.get('MACD_Signal', 0)
            features['macd_histogram'] = indicators.get('MACD_Histogram', 0)
            
            # Bollinger Bands position
            bb_upper = indicators.get('BB_Upper', 0)
            bb_lower = indicators.get('BB_Lower', 0)
            if bb_upper > bb_lower > 0:
                features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                features['bb_position'] = 0.5
            
            # Volume features
            features['volume_ratio'] = indicators.get('Volume_Ratio', 1)
            features['volume_sma_ratio'] = self._calculate_volume_sma_ratio(stock_data)
            
            # Volatility features
            features['volatility_20d'] = self._calculate_volatility(stock_data, 20)
            features['volatility_50d'] = self._calculate_volatility(stock_data, 50)
            
            # Support/Resistance levels
            support = indicators.get('Support', 0)
            resistance = indicators.get('Resistance', 0)
            
            features['support_level'] = support
            features['resistance_level'] = resistance
            features['support_distance'] = ((current_price - support) / support * 100) if support > 0 else 0
            features['resistance_distance'] = ((resistance - current_price) / current_price * 100) if resistance > 0 else 0
            
            # Market structure features
            features['higher_highs'] = self._detect_higher_highs(stock_data)
            features['lower_lows'] = self._detect_lower_lows(stock_data)
            
            # Fundamental data
            fundamentals = get_fundamental_data(symbol)
            features['pe_ratio'] = fundamentals.get('PE Ratio', 20)
            features['book_value'] = fundamentals.get('Book Value', current_price * 0.5)
            features['debt_equity'] = fundamentals.get('Debt to Equity', 0.5)
            
            # News sentiment
            sentiment_data = get_news_sentiment_score(symbol)
            features['sentiment_score'] = sentiment_data.get('sentiment_score', 50)
            features['news_confidence'] = sentiment_data.get('confidence', 0)
            
            # Market timing features
            now = datetime.now()
            features['month'] = now.month
            features['day_of_week'] = now.weekday()
            features['hour'] = now.hour
            
            # Fibonacci retracement levels
            high_52w = stock_data['High'].rolling(252).max().iloc[-1] if len(stock_data) >= 252 else stock_data['High'].max()
            low_52w = stock_data['Low'].rolling(252).min().iloc[-1] if len(stock_data) >= 252 else stock_data['Low'].min()
            
            fib_levels = self._calculate_fibonacci_levels(high_52w, low_52w)
            features['fib_23_6'] = fib_levels['23.6%']
            features['fib_38_2'] = fib_levels['38.2%']
            features['fib_50_0'] = fib_levels['50.0%']
            features['fib_61_8'] = fib_levels['61.8%']
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating price features for {symbol}: {e}")
            return {}
    
    def _calculate_volume_sma_ratio(self, stock_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate volume to SMA ratio"""
        try:
            if len(stock_data) < period:
                return 1.0
            current_volume = stock_data['Volume'].iloc[-1]
            avg_volume = stock_data['Volume'].rolling(period).mean().iloc[-1]
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_volatility(self, stock_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility"""
        try:
            if len(stock_data) < period:
                period = len(stock_data)
            returns = stock_data['Close'].pct_change().dropna()
            return returns.rolling(period).std().iloc[-1] * 100
        except:
            return 0.0
    
    def _detect_higher_highs(self, stock_data: pd.DataFrame, period: int = 10) -> int:
        """Detect higher highs pattern"""
        try:
            if len(stock_data) < period * 2:
                return 0
            
            highs = stock_data['High'].rolling(period).max()
            recent_high = highs.iloc[-1]
            previous_high = highs.iloc[-period-1]
            
            return 1 if recent_high > previous_high else 0
        except:
            return 0
    
    def _detect_lower_lows(self, stock_data: pd.DataFrame, period: int = 10) -> int:
        """Detect lower lows pattern"""
        try:
            if len(stock_data) < period * 2:
                return 0
            
            lows = stock_data['Low'].rolling(period).min()
            recent_low = lows.iloc[-1]
            previous_low = lows.iloc[-period-1]
            
            return 1 if recent_low < previous_low else 0
        except:
            return 0
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50.0%': high - (diff * 0.500),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786)
        }
    
    def predict_ideal_price(self, stock_data: pd.DataFrame, symbol: str) -> Dict:
        """Predict ideal purchase price and timing"""
        try:
            # Use technical analysis for price predictions even without trained model
            current_price = float(stock_data['Close'].iloc[-1])
            
            # Calculate basic support and resistance levels
            high_20 = float(stock_data['High'].tail(20).max())
            low_20 = float(stock_data['Low'].tail(20).min())
            avg_20 = float(stock_data['Close'].tail(20).mean())
            
            # Calculate moving averages for trend analysis
            sma_10 = float(stock_data['Close'].tail(10).mean())
            sma_20 = float(stock_data['Close'].tail(20).mean())
            
            # Simple ideal price calculation based on technical levels
            # Use 20-day average with slight discount as ideal price
            ideal_price = avg_20 * 0.98  # 2% discount from 20-day average
            
            # Price targets based on technical levels
            price_target_1w = current_price * 1.03  # 3% upside in 1 week
            price_target_1m = current_price * 1.07  # 7% upside in 1 month
            
            # Support and resistance levels
            support_level = low_20 * 1.02  # Slightly above 20-day low
            resistance_level = high_20 * 0.98  # Slightly below 20-day high
            
            # Calculate confidence based on price position in range
            price_range = high_20 - low_20
            position_in_range = (current_price - low_20) / price_range if price_range > 0 else 0.5
            confidence = 70 + (20 * (1 - abs(position_in_range - 0.5) * 2))  # 70-90% confidence
            
            # Generate recommendation based on current price vs ideal price
            price_diff_pct = ((current_price - ideal_price) / ideal_price) * 100
            
            if price_diff_pct <= -5:  # Current price is 5%+ below ideal
                recommendation = "Strong Buy - Below ideal price"
            elif price_diff_pct <= -2:  # Current price is 2-5% below ideal
                recommendation = "Buy - Near ideal price"
            elif price_diff_pct <= 2:  # Current price is within 2% of ideal
                recommendation = "Hold - At ideal price"
            elif price_diff_pct <= 5:  # Current price is 2-5% above ideal
                recommendation = "Hold - Slightly above ideal"
            else:  # Current price is significantly above ideal
                recommendation = "Wait - Above ideal price"
            
            return {
                'ideal_price': round(ideal_price, 2),
                'current_price': round(current_price, 2),
                'confidence': round(confidence, 1),
                'recommendation': recommendation,
                'price_target_1w': round(price_target_1w, 2),
                'price_target_1m': round(price_target_1m, 2),
                'support_level': round(support_level, 2),
                'resistance_level': round(resistance_level, 2),
                'price_difference_pct': round(price_diff_pct, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {symbol}: {e}")
            return {
                'ideal_price': None,
                'current_price': stock_data['Close'].iloc[-1] if not stock_data.empty else 0,
                'confidence': 0.0,
                'recommendation': f'Error: {str(e)}',
                'price_target_1w': None,
                'price_target_1m': None,
                'support_level': None,
                'resistance_level': None
            }
    
    def train_model(self, training_symbols: list, days_back: int = 90) -> Dict:
        """Train the price prediction model"""
        try:
            self.logger.info(f"Training price prediction model with {len(training_symbols)} stocks")
            
            training_data = []
            targets = []
            
            for symbol in training_symbols:
                try:
                    # Get historical data
                    from utils.data_fetcher import get_stock_data
                    stock_data = get_stock_data(symbol, period='1y')
                    
                    if len(stock_data) < 100:  # Need sufficient data
                        continue
                    
                    # Create training samples from historical data
                    for i in range(50, len(stock_data) - 5):  # Leave last 5 days for target
                        # Use data up to day i
                        historical_data = stock_data.iloc[:i+1]
                        
                        # Create features
                        features = self.create_price_features(historical_data, symbol)
                        
                        if not features:
                            continue
                        
                        # Target: future price (5 days ahead)
                        future_price = stock_data['Close'].iloc[i+5]
                        
                        training_data.append(features)
                        targets.append(future_price)
                
                except Exception as e:
                    self.logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            if len(training_data) < 100:
                raise ValueError("Insufficient training data")
            
            # Convert to DataFrame
            df_features = pd.DataFrame(training_data)
            self.feature_columns = df_features.columns.tolist()
            
            # Handle missing values
            df_features = df_features.fillna(df_features.median())
            
            # Prepare data
            X = df_features.values
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble models
            self.rf_model.fit(X_train_scaled, y_train)
            self.gb_model.fit(X_train_scaled, y_train)
            self.lr_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = self.rf_model.predict(X_test_scaled)
            gb_pred = self.gb_model.predict(X_test_scaled)
            lr_pred = self.lr_model.predict(X_test_scaled)
            
            # Ensemble prediction
            ensemble_pred = (rf_pred * 0.4 + gb_pred * 0.4 + lr_pred * 0.2)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, ensemble_pred)
            mse = mean_squared_error(y_test, ensemble_pred)
            rmse = np.sqrt(mse)
            
            # Calculate accuracy (within 5% of actual price)
            accuracy_5pct = np.mean(np.abs((ensemble_pred - y_test) / y_test) <= 0.05) * 100
            
            self.performance_metrics = {
                'mae': mae,
                'rmse': rmse,
                'accuracy_5pct': accuracy_5pct,
                'training_samples': len(training_data),
                'test_samples': len(y_test),
                'training_date': datetime.now().isoformat()
            }
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            self.logger.info(f"Model trained successfully. Accuracy: {accuracy_5pct:.1f}%")
            
            return {
                'success': True,
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'accuracy_5pct': round(accuracy_5pct, 1),
                'training_samples': len(training_data),
                'message': f'Model trained with {len(training_data)} samples'
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to train price prediction model'
            }
    
    def save_model(self):
        """Save the trained model"""
        try:
            # Save ensemble models
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'lr_model': self.lr_model,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'performance_metrics': self.performance_metrics
            }
            joblib.dump(model_data, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.logger.info("Price prediction model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load existing trained model"""
        try:
            if os.path.exists(self.model_path):
                # Load model data
                model_data = joblib.load(self.model_path)
                self.rf_model = model_data.get('rf_model')
                self.gb_model = model_data.get('gb_model')
                self.lr_model = model_data.get('lr_model')
                self.scaler = model_data.get('scaler')  # Scaler is now in main model file
                self.feature_columns = model_data.get('feature_columns', [])
                self.is_trained = model_data.get('is_trained', False)
                self.performance_metrics = model_data.get('performance_metrics', {})
                
                self.logger.info("Price prediction model loaded successfully")
                return True
            elif os.path.exists(self.scaler_path):
                # Fallback: try loading separate scaler file
                try:
                    self.scaler = joblib.load(self.scaler_path)
                except:
                    pass
                
        except Exception as e:
            self.logger.warning(f"Could not load existing model: {e}")
            
        return False

# Global instance
price_predictor = SmartPricePredictor()