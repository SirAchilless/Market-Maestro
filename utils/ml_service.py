"""
ML Service Integration for Streamlit App
Provides seamless integration of background ML processing with the frontend
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.background_ml_processor import ml_processor
import threading
import time

class MLService:
    """Service class to integrate background ML processing with Streamlit"""
    
    def __init__(self):
        self.processor = ml_processor
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the ML service"""
        try:
            # Load existing cache data first
            self.processor.ml_cache = self.processor.load_cache()
            self.processor.sentiment_cache = self.processor.load_sentiment_cache()
            
            # Start background processing automatically for continuous training
            if 'ml_service_started' not in st.session_state:
                st.session_state.ml_service_started = True
                self.start_background_processing()
        except Exception as e:
            # If initialization fails, create empty caches
            self.processor.ml_cache = {}
            self.processor.sentiment_cache = {}
    
    def start_background_processing(self):
        """Start background ML processing"""
        try:
            self.processor.start_background_processing()
        except Exception as e:
            st.error(f"Error starting ML service: {e}")
    
    def stop_background_processing(self):
        """Stop background ML processing"""
        try:
            self.processor.stop_background_processing()
            st.success("Background ML processing stopped")
        except Exception as e:
            st.error(f"Error stopping ML service: {e}")
    
    def get_market_overview(self) -> dict:
        """Get market overview from ML processor"""
        return self.processor.get_market_overview()
    
    def get_stock_ml_signal(self, symbol: str) -> dict:
        """Get ML signal for a specific stock"""
        return self.processor.get_ml_signal_for_stock(symbol)
    
    def get_stock_sentiment(self, symbol: str) -> dict:
        """Get sentiment for a specific stock"""
        return self.processor.get_sentiment_for_stock(symbol)
    
    def get_top_buy_signals(self, limit: int = 10) -> list:
        """Get top buy signals"""
        return self.processor.get_top_stocks_by_signal('Buy', limit)
    
    def get_top_sell_signals(self, limit: int = 10) -> list:
        """Get top sell signals"""
        return self.processor.get_top_stocks_by_signal('Sell', limit)
    
    def get_strong_buy_signals(self, limit: int = 10) -> list:
        """Get strong buy signals"""
        return self.processor.get_top_stocks_by_signal('Strong Buy', limit)
    
    def is_processing_active(self) -> bool:
        """Check if background processing is active"""
        return self.processor.is_running
    
    def get_processing_stats(self) -> dict:
        """Get processing statistics"""
        overview = self.get_market_overview()
        return {
            'total_stocks_processed': overview['total_stocks'],
            'coverage_percent': overview.get('coverage_percent', 0),
            'last_update': overview['last_update'],
            'buy_signals': overview['buy_signals'],
            'sell_signals': overview['sell_signals'],
            'hold_signals': overview['hold_signals']
        }
    
    def get_all_ml_data(self) -> pd.DataFrame:
        """Get all ML data for all processed stocks"""
        return self.search_stocks_by_signal()  # No filters = all data
    
    def search_stocks_by_signal(self, signal_type: str = None, min_confidence: float = 0.0) -> pd.DataFrame:
        """Search stocks by ML signal criteria"""
        all_data = []
        
        for symbol, data in self.processor.ml_cache.items():
            if signal_type and data.get('ml_signal') != signal_type:
                continue
            if data.get('ml_confidence', 0) < min_confidence:
                continue
            
            # Get sentiment data
            sentiment_data = self.get_stock_sentiment(symbol)
            
            row = {
                'Symbol': symbol,
                'Current Price': data.get('current_price', 0),
                'Ideal Price': data.get('ideal_price', 0),
                'Price Target 1W': data.get('price_target_1w', 0),
                'Price Target 1M': data.get('price_target_1m', 0),
                'Price Confidence': data.get('price_confidence', 0),
                'Price Recommendation': data.get('price_recommendation', ''),
                'ML Signal': data.get('ml_signal', 'Hold'),
                'ML Confidence': data.get('ml_confidence', 0),
                'ML Score': data.get('ml_score', 50),
                'Technical Signal': data.get('technical_signal', 'Hold'),
                'Change %': data.get('change_percent', 0),
                'Volume': data.get('volume', 0),
                'Sentiment Score': sentiment_data.get('sentiment_score', 50),
                'Sentiment Label': sentiment_data.get('sentiment_label', 'neutral'),
                'News Articles': sentiment_data.get('article_count', 0),
                'Last Updated': data.get('timestamp', ''),
                'ML Recommendation': data.get('ml_recommendation', ''),
                'Support Level': data.get('support_level', 0),
                'Resistance Level': data.get('resistance_level', 0)
            }
            all_data.append(row)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Sort by ML confidence
        df = df.sort_values('ML Confidence', ascending=False)
        
        return df
    
    def get_sector_analysis(self) -> dict:
        """Get sector-wise ML analysis"""
        from utils.stock_lists import SECTOR_STOCKS
        
        sector_analysis = {}
        
        for sector, stocks in SECTOR_STOCKS.items():
            signals = []
            confidences = []
            
            for stock in stocks:
                data = self.get_stock_ml_signal(stock)
                if data.get('ml_signal') != 'Hold':
                    signals.append(data.get('ml_signal'))
                    confidences.append(data.get('ml_confidence', 0))
            
            if signals:
                buy_signals = sum(1 for s in signals if 'Buy' in s)
                sell_signals = sum(1 for s in signals if 'Sell' in s)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                sector_analysis[sector] = {
                    'total_signals': len(signals),
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'avg_confidence': avg_confidence,
                    'sector_sentiment': 'Bullish' if buy_signals > sell_signals else 'Bearish' if sell_signals > buy_signals else 'Neutral'
                }
        
        return sector_analysis
    
    def get_market_sentiment_overview(self) -> dict:
        """Get overall market sentiment from processed data"""
        if not self.processor.sentiment_cache:
            return {
                'market_mood': 'neutral',
                'market_sentiment_score': 50,
                'stocks_analyzed': 0,
                'bullish_stocks': 0,
                'bearish_stocks': 0
            }
        
        sentiment_scores = []
        bullish_count = 0
        bearish_count = 0
        
        for data in self.processor.sentiment_cache.values():
            score = data.get('sentiment_score', 50)
            sentiment_scores.append(score)
            
            if score > 60:
                bullish_count += 1
            elif score < 40:
                bearish_count += 1
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 50
        
        # Determine market mood
        if avg_sentiment > 60:
            mood = 'bullish'
        elif avg_sentiment < 40:
            mood = 'bearish'
        else:
            mood = 'neutral'
        
        return {
            'market_mood': mood,
            'market_sentiment_score': round(avg_sentiment, 2),
            'stocks_analyzed': len(sentiment_scores),
            'bullish_stocks': bullish_count,
            'bearish_stocks': bearish_count
        }

# Global service instance
ml_service = MLService()