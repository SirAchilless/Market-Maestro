"""
Train the price prediction model with historical stock data
This should be run to enable price predictions in the stock screener
"""

import pandas as pd
import numpy as np
from utils.price_predictor import SmartPricePredictor
from utils.stock_lists import NIFTY_50, NIFTY_100
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_price_predictor():
    """Train the price prediction model"""
    try:
        # Initialize the price predictor
        predictor = SmartPricePredictor()
        
        # Use a subset of major stocks for training
        training_stocks = NIFTY_50[:30]  # Use top 30 stocks for training
        logger.info(f"Training price predictor with {len(training_stocks)} stocks")
        
        # Train the model
        training_results = predictor.train_model(training_stocks, days_back=120)
        
        if training_results.get('success', False):
            logger.info("Price predictor training completed successfully!")
            logger.info(f"Model accuracy: {training_results.get('accuracy', 0):.2%}")
            logger.info(f"Mean Absolute Error: ₹{training_results.get('mae', 0):.2f}")
            
            # Test the model with a few stocks
            logger.info("Testing price predictions...")
            test_stocks = ['RELIANCE', 'TCS', 'INFY']
            
            for symbol in test_stocks:
                try:
                    from utils.data_fetcher import get_stock_data
                    stock_data = get_stock_data(symbol, period='6mo')
                    if not stock_data.empty:
                        prediction = predictor.predict_ideal_price(stock_data, symbol)
                        logger.info(f"{symbol}: Ideal ₹{prediction.get('ideal_price', 'N/A')}, "
                                  f"Current ₹{prediction.get('current_price', 'N/A')}, "
                                  f"Confidence {prediction.get('confidence', 0)}%")
                except Exception as e:
                    logger.error(f"Error testing {symbol}: {e}")
                    
        else:
            logger.error("Price predictor training failed!")
            logger.error(f"Error: {training_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in training price predictor: {e}")

if __name__ == "__main__":
    train_price_predictor()