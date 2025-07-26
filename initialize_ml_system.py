#!/usr/bin/env python3
"""
Initialize ML System for Background Processing
Trains the model and starts background processing
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_ml_simple import train_simple_model
from utils.background_ml_processor import ml_processor

def initialize_ml_system():
    """Initialize the complete ML system"""
    
    print("=== Initializing ML System ===")
    print(f"Start time: {datetime.now()}")
    
    # Step 1: Train the ML model
    print("\n1. Training ML model...")
    try:
        success = train_simple_model()
        if success:
            print("✅ ML model training completed successfully")
        else:
            print("⚠️ ML model training failed")
    except Exception as e:
        print(f"❌ Error training ML model: {e}")
    
    # Step 2: Initialize background processor
    print("\n2. Initializing background ML processor...")
    try:
        # Load existing cache if available
        print(f"Stock universe size: {len(ml_processor.stock_universe)}")
        print(f"Cached ML signals: {len(ml_processor.ml_cache)}")
        print(f"Cached sentiment data: {len(ml_processor.sentiment_cache)}")
        
        # Start initial processing if cache is empty or small
        if len(ml_processor.ml_cache) < 10:
            print("Starting initial ML processing...")
            ml_processor.batch_process_ml_signals(batch_size=5)
            
        if len(ml_processor.sentiment_cache) < 5:
            print("Starting initial sentiment processing...")
            ml_processor.batch_process_sentiment(batch_size=3)
        
        print("✅ Background ML processor initialized")
        
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
    
    print(f"\n=== Initialization Complete ===")
    print(f"End time: {datetime.now()}")
    
    # Print final status
    print(f"\nFinal Status:")
    print(f"- ML Cache: {len(ml_processor.ml_cache)} stocks")
    print(f"- Sentiment Cache: {len(ml_processor.sentiment_cache)} stocks")
    print(f"- Models available: {os.path.exists('models/trading_signal_model.joblib')}")

if __name__ == "__main__":
    initialize_ml_system()