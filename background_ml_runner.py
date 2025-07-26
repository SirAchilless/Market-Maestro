#!/usr/bin/env python3
"""
Standalone Background ML Runner
Runs continuously and independently from Streamlit
"""

import time
import logging
import signal
import sys
from datetime import datetime
from utils.background_ml_processor import ml_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BackgroundMLRunner:
    """Standalone background ML runner"""
    
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        ml_processor.stop_background_processing()
        sys.exit(0)
    
    def run(self):
        """Main run loop"""
        logger.info("Starting standalone background ML runner...")
        logger.info(f"Stock universe: {len(ml_processor.stock_universe)} stocks")
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                logger.info(f"=== Starting ML Processing Cycle {cycle} ===")
                
                # Process ML signals for all stocks
                logger.info("Processing ML signals...")
                ml_processor.batch_process_ml_signals(batch_size=8)
                
                # Process sentiment for major stocks
                logger.info("Processing sentiment analysis...")
                ml_processor.batch_process_sentiment(batch_size=4)
                
                # Update timestamp
                ml_processor.update_last_update_time()
                
                # Log status
                overview = ml_processor.get_market_overview()
                logger.info(f"Cycle {cycle} complete:")
                logger.info(f"  - ML signals: {overview['total_stocks']} stocks")
                logger.info(f"  - Buy signals: {overview['buy_signals']}")
                logger.info(f"  - Sell signals: {overview['sell_signals']}")
                logger.info(f"  - Sentiment data: {len(ml_processor.sentiment_cache)} stocks")
                
                # Wait 3 minutes between cycles for continuous updates
                logger.info("Waiting 3 minutes before next cycle...")
                for i in range(180):  # 3 minutes
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in processing cycle {cycle}: {e}")
                time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    runner = BackgroundMLRunner()
    runner.run()