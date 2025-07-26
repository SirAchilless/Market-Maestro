"""
Background ML Processing System for All NSE Stocks
Continuously processes ML signals and sentiment for the entire stock universe
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import joblib

from utils.stock_lists import NIFTY_50, NIFTY_100, NIFTY_200, SECTOR_STOCKS
from utils.data_fetcher import get_stock_data
from utils.ml_signals import get_ml_signal, get_simple_ml_signal
from utils.news_sentiment import get_news_sentiment_score
from utils.technical_analysis import get_purchase_signal

class BackgroundMLProcessor:
    """Background processor for ML signals across all NSE stocks"""
    
    def __init__(self):
        self.data_dir = "ml_data"
        self.cache_file = os.path.join(self.data_dir, "ml_signals_cache.json")
        self.sentiment_cache_file = os.path.join(self.data_dir, "sentiment_cache.json")
        self.last_update_file = os.path.join(self.data_dir, "last_update.json")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize caches
        self.ml_cache = self.load_cache()
        self.sentiment_cache = self.load_sentiment_cache()
        self.is_running = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get comprehensive stock list
        self.stock_universe = self.get_comprehensive_stock_list()
        
    def get_comprehensive_stock_list(self) -> List[str]:
        """Get comprehensive list of NSE stocks for processing"""
        stocks = set()
        
        # Add major indices
        stocks.update(NIFTY_50)
        stocks.update(NIFTY_100)
        stocks.update(NIFTY_200)
        
        # Add all sector stocks
        for sector_stocks in SECTOR_STOCKS.values():
            stocks.update(sector_stocks)
        
        # Convert to list and sort
        return sorted(list(stocks))
    
    def load_cache(self) -> Dict:
        """Load ML signals cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading ML cache: {e}")
        return {}
    
    def save_cache(self):
        """Save ML signals cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.ml_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving ML cache: {e}")
    
    def load_sentiment_cache(self) -> Dict:
        """Load sentiment cache"""
        try:
            if os.path.exists(self.sentiment_cache_file):
                with open(self.sentiment_cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading sentiment cache: {e}")
        return {}
    
    def save_sentiment_cache(self):
        """Save sentiment cache"""
        try:
            with open(self.sentiment_cache_file, 'w') as f:
                json.dump(self.sentiment_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving sentiment cache: {e}")
    
    def update_last_update_time(self):
        """Update last update timestamp"""
        try:
            update_info = {
                'last_update': datetime.now().isoformat(),
                'stocks_processed': len(self.ml_cache),
                'sentiment_stocks': len(self.sentiment_cache)
            }
            with open(self.last_update_file, 'w') as f:
                json.dump(update_info, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error updating timestamp: {e}")
    
    def process_single_stock_ml(self, symbol: str) -> Optional[Dict]:
        """Process ML signals and price predictions for a single stock"""
        try:
            # Get stock data
            stock_data = get_stock_data(symbol, period='6mo')
            
            if stock_data.empty:
                return None
            
            # Get ML signal
            ml_result = get_ml_signal(stock_data, symbol)
            
            # Get price prediction
            from utils.price_predictor import price_predictor
            price_prediction = price_predictor.predict_ideal_price(stock_data, symbol)
            
            # Get technical signal
            tech_signal, tech_details = get_purchase_signal(stock_data, symbol)
            
            # Combine results with price predictions
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(stock_data['Close'].iloc[-1]),
                'ml_signal': ml_result.get('signal', 'Hold'),
                'ml_confidence': float(ml_result.get('confidence', 0)),
                'ml_score': float(ml_result.get('ml_score', 50)),
                'ml_recommendation': ml_result.get('recommendation', ''),
                'technical_signal': tech_signal,
                'technical_details': tech_details,
                'volume': float(stock_data['Volume'].iloc[-1]),
                'change_percent': float(((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100),
                # Price prediction data
                'ideal_price': price_prediction.get('ideal_price'),
                'price_confidence': price_prediction.get('confidence', 0),
                'price_recommendation': price_prediction.get('recommendation', ''),
                'price_target_1w': price_prediction.get('price_target_1w'),
                'price_target_1m': price_prediction.get('price_target_1m'),
                'support_level': price_prediction.get('support_level'),
                'resistance_level': price_prediction.get('resistance_level'),
                'price_difference_pct': price_prediction.get('price_difference_pct', 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing ML for {symbol}: {e}")
            return None
    
    def process_single_stock_sentiment(self, symbol: str) -> Optional[Dict]:
        """Process sentiment for a single stock"""
        try:
            sentiment_data = get_news_sentiment_score(symbol)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sentiment_score': sentiment_data.get('sentiment_score', 50),
                'sentiment_label': sentiment_data.get('sentiment_label', 'neutral'),
                'sentiment_confidence': sentiment_data.get('confidence', 0),
                'article_count': sentiment_data.get('article_count', 0),
                'recent_articles': sentiment_data.get('recent_articles', [])[:3]  # Keep only top 3
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment for {symbol}: {e}")
            return None
    
    def batch_process_ml_signals(self, batch_size: int = 10):
        """Process ML signals for all stocks in batches"""
        self.logger.info(f"Starting ML processing for {len(self.stock_universe)} stocks")
        
        processed_count = 0
        
        # Process in batches using ThreadPoolExecutor
        for i in range(0, len(self.stock_universe), batch_size):
            batch = self.stock_universe[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(self.process_single_stock_ml, symbol): symbol 
                    for symbol in batch
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            self.ml_cache[symbol] = result
                            processed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error in batch processing for {symbol}: {e}")
            
            # Save cache periodically
            if processed_count % (batch_size * 2) == 0:
                self.save_cache()
                self.logger.info(f"Processed {processed_count}/{len(self.stock_universe)} stocks")
            
            # Small delay to avoid overwhelming APIs
            time.sleep(1)
        
        self.save_cache()
        self.logger.info(f"ML processing complete. Processed {processed_count} stocks")
    
    def batch_process_sentiment(self, batch_size: int = 5):
        """Process sentiment for major stocks"""
        # Focus on major stocks for sentiment to avoid rate limits
        major_stocks = NIFTY_50[:20]  # Top 20 stocks for sentiment
        
        self.logger.info(f"Starting sentiment processing for {len(major_stocks)} major stocks")
        
        processed_count = 0
        
        for i in range(0, len(major_stocks), batch_size):
            batch = major_stocks[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_symbol = {
                    executor.submit(self.process_single_stock_sentiment, symbol): symbol 
                    for symbol in batch
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            self.sentiment_cache[symbol] = result
                            processed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error in sentiment processing for {symbol}: {e}")
            
            # Longer delay for news APIs
            time.sleep(5)
        
        self.save_sentiment_cache()
        self.logger.info(f"Sentiment processing complete. Processed {processed_count} stocks")
    
    def run_background_processing(self):
        """Main background processing loop"""
        self.is_running = True
        self.logger.info("Starting background ML processing...")
        
        while self.is_running:
            try:
                # Process ML signals for all stocks
                self.batch_process_ml_signals()
                
                # Process sentiment for major stocks
                self.batch_process_sentiment()
                
                # Update timestamp
                self.update_last_update_time()
                
                # Wait before next cycle (5 minutes for more frequent updates)
                self.logger.info("Processing cycle complete. Waiting 5 minutes...")
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in background processing: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def start_background_processing(self):
        """Start background processing in a separate thread"""
        if not self.is_running:
            processing_thread = threading.Thread(target=self.run_background_processing, daemon=True)
            processing_thread.start()
            self.logger.info("Background processing started")
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.is_running = False
        self.logger.info("Background processing stopped")
    
    def get_ml_signal_for_stock(self, symbol: str) -> Dict:
        """Get cached ML signal for a stock"""
        # Ensure cache is loaded
        if not self.ml_cache:
            self.ml_cache = self.load_cache()
        
        return self.ml_cache.get(symbol, {
            'symbol': symbol,
            'ml_signal': 'Hold',
            'ml_confidence': 0,
            'ml_score': 50,
            'ml_recommendation': 'No data available',
            'technical_signal': 'Hold',
            'current_price': 0,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_sentiment_for_stock(self, symbol: str) -> Dict:
        """Get cached sentiment for a stock"""
        # Ensure cache is loaded
        if not self.sentiment_cache:
            self.sentiment_cache = self.load_sentiment_cache()
        
        return self.sentiment_cache.get(symbol, {
            'symbol': symbol,
            'sentiment_score': 50,
            'sentiment_label': 'neutral',
            'sentiment_confidence': 0,
            'article_count': 0,
            'recent_articles': [],
            'timestamp': datetime.now().isoformat()
        })
    
    def get_market_overview(self) -> Dict:
        """Get market overview from cached data"""
        # Ensure cache is loaded
        if not self.ml_cache:
            self.ml_cache = self.load_cache()
        
        if not self.ml_cache:
            return {
                'total_stocks': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'last_update': 'Never'
            }
        
        buy_count = sum(1 for data in self.ml_cache.values() if data.get('ml_signal') in ['Buy', 'Strong Buy'])
        sell_count = sum(1 for data in self.ml_cache.values() if data.get('ml_signal') in ['Sell', 'Strong Sell'])
        hold_count = len(self.ml_cache) - buy_count - sell_count
        
        # Get latest update time
        latest_time = max(
            (datetime.fromisoformat(data.get('timestamp', '2000-01-01T00:00:00'))
             for data in self.ml_cache.values()),
            default=datetime.now()
        )
        
        return {
            'total_stocks': len(self.ml_cache),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'last_update': latest_time.strftime('%Y-%m-%d %H:%M:%S'),
            'coverage_percent': (len(self.ml_cache) / len(self.stock_universe)) * 100
        }
    
    def get_top_stocks_by_signal(self, signal_type: str = 'Buy', limit: int = 10) -> List[Dict]:
        """Get top stocks by ML signal type"""
        filtered_stocks = [
            data for data in self.ml_cache.values()
            if data.get('ml_signal') == signal_type
        ]
        
        # Sort by ML confidence
        sorted_stocks = sorted(
            filtered_stocks,
            key=lambda x: x.get('ml_confidence', 0),
            reverse=True
        )
        
        return sorted_stocks[:limit]

# Global instance
ml_processor = BackgroundMLProcessor()