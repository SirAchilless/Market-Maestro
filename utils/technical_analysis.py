import pandas as pd
import numpy as np
import streamlit as st

def calculate_rsi(data, period=14):
    """Calculate RSI (Relative Strength Index)"""
    try:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        return np.array([])

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        middle = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.values, middle.values, lower.values
    except Exception as e:
        st.error(f"Error calculating Bollinger Bands: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def calculate_moving_averages(data, periods=[20, 44, 50, 200]):
    """Calculate multiple moving averages"""
    try:
        mas = {}
        for period in periods:
            mas[f'MA{period}'] = data['Close'].rolling(window=period).mean().values
        return mas
    except Exception as e:
        st.error(f"Error calculating moving averages: {str(e)}")
        return {}

def calculate_volume_sma(data, period=20):
    """Calculate volume moving average"""
    try:
        volume_sma = data['Volume'].rolling(window=period).mean()
        return volume_sma.values
    except Exception as e:
        st.error(f"Error calculating volume SMA: {str(e)}")
        return np.array([])

def detect_breakout(data, lookback_period=20):
    """Detect if stock has broken out of recent highs/lows"""
    try:
        if len(data) < lookback_period + 1:
            return None, None
        
        # Get recent high and low (excluding current day)
        recent_high = data['High'].iloc[-(lookback_period+1):-1].max()
        recent_low = data['Low'].iloc[-(lookback_period+1):-1].min()
        
        current_price = data['Close'].iloc[-1]
        
        bullish_breakout = current_price > recent_high
        bearish_breakout = current_price < recent_low
        
        return bullish_breakout, bearish_breakout
    except Exception as e:
        return None, None

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels"""
    try:
        if len(data) < window:
            return None, None
        
        # Simple support/resistance calculation
        recent_data = data.tail(window)
        resistance = recent_data['High'].max()
        support = recent_data['Low'].min()
        
        return support, resistance
    except Exception as e:
        return None, None

def get_simplified_signal(data, symbol=None):
    """Generate simplified purchase signal for stocks with limited data"""
    try:
        if len(data) < 10:
            return "Insufficient Data", "Need at least 10 days of data"
        
        current_price = data['Close'].iloc[-1]
        signals = []
        signal_strength = 0
        
        # Simple price momentum analysis
        if len(data) >= 5:
            recent_avg = data['Close'].tail(5).mean()
            older_avg = data['Close'].head(5).mean() if len(data) >= 10 else recent_avg
            
            if current_price > recent_avg * 1.02:
                signals.append("Above Recent Average")
                signal_strength += 1
            elif current_price < recent_avg * 0.98:
                signals.append("Below Recent Average")
                signal_strength -= 1
                
            if recent_avg > older_avg * 1.01:
                signals.append("Upward Trend")
                signal_strength += 1
            elif recent_avg < older_avg * 0.99:
                signals.append("Downward Trend")
                signal_strength -= 1
        
        # Volume analysis if available
        if 'Volume' in data.columns:
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].mean()
            
            if current_volume > avg_volume * 1.5:
                signals.append("High Volume")
                signal_strength += 1
        
        # Generate recommendation
        if signal_strength >= 2:
            recommendation = "Buy"
        elif signal_strength == 1:
            recommendation = "Watch"
        elif signal_strength <= -2:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        
        return recommendation, " | ".join(signals) if signals else "Limited analysis - need more data"
    
    except Exception as e:
        return "Error", f"Simplified analysis error: {str(e)}"

def get_purchase_signal(data, symbol=None):
    """Generate enhanced purchase signal using technical indicators, ML, and news sentiment"""
    try:
        if data.empty or len(data) < 20:
            return "Insufficient Data", "Need at least 20 days of historical data"
        
        # For data with 20-49 points, use simplified analysis
        if len(data) < 50:
            return get_simplified_signal(data, symbol)
        
        # Calculate technical indicators
        rsi = calculate_rsi(data)
        macd, macd_signal, macd_hist = calculate_macd(data)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data)
        
        if len(rsi) == 0 or len(macd) == 0:
            return "Error", "Failed to calculate indicators"
        
        # Get latest values
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
        current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
        current_price = data['Close'].iloc[-1]
        lower_band = lower_bb[-1] if not np.isnan(lower_bb[-1]) else current_price
        
        # Traditional technical signals
        signals = []
        signal_strength = 0
        
        # RSI analysis
        if current_rsi <= 30:
            signals.append("RSI Oversold")
            signal_strength += 2
        elif current_rsi <= 40:
            signals.append("RSI Near Oversold")
            signal_strength += 1
        elif current_rsi >= 70:
            signals.append("RSI Overbought")
            signal_strength -= 1
        
        # MACD analysis
        if current_macd > current_macd_signal:
            signals.append("MACD Bullish")
            signal_strength += 1
        else:
            signals.append("MACD Bearish")
            signal_strength -= 1
        
        # Bollinger Bands analysis
        if current_price <= lower_band * 1.02:
            signals.append("Near Lower BB")
            signal_strength += 1
        elif current_price >= upper_bb[-1] * 0.98:
            signals.append("Near Upper BB")
            signal_strength -= 1
        
        # Breakout detection
        bullish_breakout, bearish_breakout = detect_breakout(data)
        if bullish_breakout:
            signals.append("Bullish Breakout")
            signal_strength += 2
        elif bearish_breakout:
            signals.append("Bearish Breakout")
            signal_strength -= 2
        
        # ML-based signal (if available and symbol provided)
        ml_signal_strength = 0
        if symbol:
            try:
                from utils.ml_signals import get_ml_signal, is_model_trained
                
                if is_model_trained():
                    ml_result = get_ml_signal(data, symbol)
                    ml_signal = ml_result.get('signal', 'Hold')
                    ml_confidence = ml_result.get('confidence', 0)
                    
                    if ml_signal == 'Strong Buy' and ml_confidence > 0.7:
                        signals.append("ML Strong Buy")
                        ml_signal_strength += 3
                    elif ml_signal == 'Buy' and ml_confidence > 0.6:
                        signals.append("ML Buy")
                        ml_signal_strength += 2
                    elif ml_signal == 'Sell' and ml_confidence > 0.6:
                        signals.append("ML Sell")
                        ml_signal_strength -= 2
                    else:
                        signals.append("ML Neutral")
                        ml_signal_strength += 0
            except Exception as e:
                # ML analysis failed, continue with technical analysis only
                pass
        
        # News sentiment analysis (if symbol provided)
        sentiment_strength = 0
        if symbol:
            try:
                from utils.news_sentiment import get_news_sentiment_score
                
                sentiment_data = get_news_sentiment_score(symbol)
                sentiment_score = sentiment_data.get('sentiment_score', 50)
                sentiment_confidence = sentiment_data.get('confidence', 0)
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                
                if sentiment_score > 70 and sentiment_confidence > 0.5:
                    signals.append("Bullish News")
                    sentiment_strength += 2
                elif sentiment_score > 60 and sentiment_confidence > 0.3:
                    signals.append("Positive News")
                    sentiment_strength += 1
                elif sentiment_score < 30 and sentiment_confidence > 0.5:
                    signals.append("Bearish News")
                    sentiment_strength -= 2
                elif sentiment_score < 40 and sentiment_confidence > 0.3:
                    signals.append("Negative News")
                    sentiment_strength -= 1
                else:
                    signals.append("Neutral News")
                    sentiment_strength += 0
            except Exception as e:
                # News analysis failed, continue without sentiment
                pass
        
        # Combine all signal strengths
        total_signal_strength = signal_strength + ml_signal_strength + sentiment_strength
        
        # Generate final recommendation
        if total_signal_strength >= 5:
            recommendation = "Strong Buy"
        elif total_signal_strength >= 3:
            recommendation = "Buy"
        elif total_signal_strength >= 1:
            recommendation = "Watch"
        elif total_signal_strength <= -3:
            recommendation = "Sell"
        elif total_signal_strength <= -1:
            recommendation = "Weak Sell"
        else:
            recommendation = "Hold"
        
        return recommendation, " | ".join(signals) if signals else "No clear signals"
    
    except Exception as e:
        return "Error", f"Calculation error: {str(e)}"

def calculate_all_indicators(data):
    """Calculate all technical indicators for a stock"""
    try:
        if data.empty or len(data) < 5:
            return {}
        
        indicators = {}
        
        # Basic price data
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) >= 2 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
        
        indicators['Current_Price'] = current_price
        indicators['Price_Change'] = price_change
        indicators['Price_Change_Pct'] = price_change_pct
        
        # RSI (needs at least 15 periods)
        if len(data) >= 15:
            rsi = calculate_rsi(data)
            indicators['RSI'] = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
        else:
            indicators['RSI'] = 50  # Neutral RSI when insufficient data
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(data)
        indicators['MACD'] = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
        indicators['MACD_Signal'] = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
        indicators['MACD_Histogram'] = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data)
        indicators['BB_Upper'] = upper_bb[-1] if len(upper_bb) > 0 and not np.isnan(upper_bb[-1]) else 0
        indicators['BB_Middle'] = middle_bb[-1] if len(middle_bb) > 0 and not np.isnan(middle_bb[-1]) else 0
        indicators['BB_Lower'] = lower_bb[-1] if len(lower_bb) > 0 and not np.isnan(lower_bb[-1]) else 0
        
        # Moving Averages
        mas = calculate_moving_averages(data, [20, 44, 50, 200])
        for ma_name, ma_values in mas.items():
            indicators[ma_name] = ma_values[-1] if len(ma_values) > 0 and not np.isnan(ma_values[-1]) else 0
        
        # Volume analysis
        volume_sma = calculate_volume_sma(data)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = volume_sma[-1] if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) else current_volume
        indicators['Volume_Ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support and Resistance
        support, resistance = calculate_support_resistance(data)
        indicators['Support'] = support if support else 0
        indicators['Resistance'] = resistance if resistance else 0
        
        # Breakout detection
        bullish_breakout, bearish_breakout = detect_breakout(data)
        indicators['Bullish_Breakout'] = bullish_breakout
        indicators['Bearish_Breakout'] = bearish_breakout
        
        return indicators
    
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return {}

def get_trend_status(indicators):
    """Determine overall trend based on indicators"""
    try:
        if not indicators:
            return "Unknown"
        
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI analysis
        rsi = indicators.get('RSI', 50)
        if rsi > 70:
            bearish_signals += 1
        elif rsi < 30:
            bullish_signals += 1
        elif rsi > 50:
            bullish_signals += 0.5
        else:
            bearish_signals += 0.5
        
        # MACD analysis
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Moving Average analysis
        current_price = indicators.get('Current_Price', 0)
        ma20 = indicators.get('MA20', 0)
        ma50 = indicators.get('MA50', 0)
        
        if current_price > ma20 > ma50:
            bullish_signals += 1
        elif current_price < ma20 < ma50:
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals > bearish_signals + 0.5:
            return "Bullish"
        elif bearish_signals > bullish_signals + 0.5:
            return "Bearish"
        else:
            return "Neutral"
    
    except Exception as e:
        return "Unknown"

def get_technical_score(data):
    """Calculate technical analysis score (0-100)"""
    try:
        if data.empty or len(data) < 10:
            return 0
        
        score = 0
        max_score = 0
        
        # Calculate indicators
        indicators = calculate_all_indicators(data)
        if not indicators:
            return 0
        
        # RSI scoring (30-70 range is good, oversold/overbought gets penalty)
        rsi = indicators.get('RSI', 50)
        if 40 <= rsi <= 60:
            score += 20  # Neutral zone
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score += 15  # Slight oversold/overbought
        elif rsi < 30:
            score += 25  # Oversold is good for buying
        elif rsi > 70:
            score += 5   # Overbought is risky
        max_score += 25
        
        # MACD scoring
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        macd_hist = indicators.get('MACD_Histogram', 0)
        
        if macd > macd_signal and macd_hist > 0:
            score += 20  # Bullish MACD
        elif macd > macd_signal:
            score += 15  # MACD above signal
        elif macd < macd_signal and macd_hist < 0:
            score += 5   # Bearish MACD
        else:
            score += 10  # Neutral
        max_score += 20
        
        # Moving average scoring
        current_price = indicators.get('Current_Price', 0)
        ma_20 = indicators.get('MA_20', 0)
        ma_50 = indicators.get('MA_50', 0)
        
        if current_price > ma_20 > ma_50:
            score += 15  # Strong uptrend
        elif current_price > ma_20:
            score += 10  # Above short-term MA
        elif current_price > ma_50:
            score += 5   # Above medium-term MA
        max_score += 15
        
        # Bollinger Bands scoring
        bb_upper = indicators.get('BB_Upper', 0)
        bb_lower = indicators.get('BB_Lower', 0)
        
        if bb_lower > 0 and current_price <= bb_lower * 1.02:
            score += 15  # Near lower band (good for buying)
        elif bb_upper > 0 and current_price >= bb_upper * 0.98:
            score += 5   # Near upper band (risky)
        else:
            score += 10  # Middle range
        max_score += 15
        
        # Volume scoring
        volume_ratio = indicators.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            score += 10  # High volume
        elif volume_ratio > 1.0:
            score += 7   # Above average volume
        else:
            score += 3   # Low volume
        max_score += 10
        
        # Price momentum scoring
        price_change_pct = indicators.get('Price_Change_Pct', 0)
        if 0 < price_change_pct <= 3:
            score += 15  # Positive momentum
        elif price_change_pct > 3:
            score += 10  # Strong momentum (but risky)
        elif -2 <= price_change_pct < 0:
            score += 5   # Slight decline
        else:
            score += 0   # Strong decline
        max_score += 15
        
        # Calculate final score
        final_score = (score / max_score) * 100 if max_score > 0 else 0
        return round(final_score, 1)
    
    except Exception as e:
        return 0

def get_technical_signal(score):
    """Get technical signal based on score"""
    if score >= 75:
        return "Strong Buy"
    elif score >= 60:
        return "Buy"
    elif score >= 45:
        return "Hold"
    elif score >= 30:
        return "Weak Sell"
    else:
        return "Sell"
