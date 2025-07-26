# Smart Indian Stock Market Screener with AI

## Overview

This is an advanced AI-powered Streamlit-based Indian stock market screening and analysis application that combines machine learning, news sentiment analysis, and traditional technical/fundamental analysis for intelligent algo trading decisions. The application provides real-time stock data analysis, ML-powered buy/sell signals, news sentiment tracking, and comprehensive portfolio management tools specifically tailored for the Indian stock market (NSE).

## Recent Changes

**Latest Major Update (July 26, 2025) - Enhanced UI/UX & PDF Export Color Preservation COMPLETED:**
- ✅ Fixed duplicate navigation icons by removing redundant emoji icons from menu labels
- ✅ Enhanced chart color preservation in PDF exports with vibrant color palettes
- ✅ Updated sector allocation charts with 15 distinct vibrant colors for PDF clarity
- ✅ Improved PDF export quality with higher resolution (2-3x scale) and Kaleido engine settings
- ✅ Enhanced chart conversion functions across all export formats (PDF, ZIP, comprehensive exports)
- ✅ Standardized chart color consistency between web display and exported documents
- ✅ Updated multiple export utilities with enhanced color preservation mechanisms
- ✅ Fixed chart export functions to maintain professional quality with proper backgrounds

**Previous Major Update (July 26, 2025) - Enhanced One-Click PDF Export System & ML Training Fix COMPLETED:**
- ✅ Created comprehensive EnhancedPDFExporter for complete data and chart export
- ✅ Implemented one-click PDF export that captures all data and charts as they appear on screen
- ✅ Added professional PDF report generation with executive summary, risk analysis, and chart gallery
- ✅ Enhanced Portfolio Risk Assessment with complete visual PDF export matching screen layout
- ✅ Integrated chart-to-image conversion for high-quality PDF chart embedding
- ✅ Created structured PDF reports with proper formatting, styling, and professional layout
- ✅ Added comprehensive data capture from session state for complete analysis export
- ✅ Implemented PDF report sections: title page, executive summary, detailed analysis, chart gallery, disclaimers
- ✅ Enhanced export functionality with proper error handling and user feedback
- ✅ Fixed comprehensive export system supporting charts, tables, and formatted reports across all pages
- ✅ Fixed ML Training page NameError by implementing missing `train_comprehensive_price_model` function
- ✅ Trained price prediction model with 3,234 samples achieving R² score of 1.000 using 15 major Indian stocks
- ✅ Updated price predictor model loading logic to work with new trained model structure
- ✅ Fixed import errors in ML Training page for complete functionality
- ✅ Enhanced price prediction interface with training controls, testing features, and performance metrics

**Previous Major Update (July 26, 2025) - Live Market Mood & Enhanced Analytics COMPLETED:**
- ✅ Implemented dynamic LIVE market mood calculation with real-time NSE stock analysis
- ✅ Enhanced market sentiment algorithm using weighted ML signals (70%) and news sentiment (30%)
- ✅ Added responsive mood detection with improved bullish/bearish signal ratio analysis
- ✅ Created comprehensive sector-wise market mood visualization with individual colored boxes
- ✅ Integrated interactive news directly into sector mood boxes with stock-wise news display
- ✅ Enhanced news presentation with sentiment-based color coding and clickable expandable content
- ✅ Added live indicator with pulsing animation and real-time update information
- ✅ Fixed market mood calculation accuracy to properly reflect actual market conditions
- ✅ Moved Market Indices Overview to top of page for prominent display of NIFTY 50, Bank Nifty, IT Index, Auto Index, and Pharma Index
- ✅ Streamlined user interface with sector-specific news access and enhanced user experience

**Previous Major Update (July 26, 2025) - Technical & Fundamental Scoring COMPLETED:**
- ✅ Fixed Technical Score and Fundamental Score calculations in ML Training detailed analysis
- ✅ Implemented comprehensive technical scoring system (0-100) based on RSI, MACD, moving averages, Bollinger Bands, volume, and momentum
- ✅ Enhanced fundamental scoring with PE ratio, ROE, ROCE, debt-to-equity, current ratio, profit margin, and EPS growth analysis
- ✅ Replaced "Calculating..." and "N/A" placeholders with real-time calculated scores
- ✅ Added technical signal generation (Strong Buy, Buy, Hold, Weak Sell, Sell) based on scoring
- ✅ Integrated confidence scoring for both technical and fundamental analysis
- ✅ Updated ML Training page detailed analysis to show complete 4-component breakdown

**Previous Major Update (July 26, 2025) - Portfolio Risk Assessment COMPLETED:**
- ✅ Fixed Portfolio Risk Assessment page functionality completely
- ✅ Resolved Yahoo Finance API data fetching issues with new column structure handling
- ✅ Corrected sector allocation mapping for Indian stocks (IT, Banking, etc.)
- ✅ Fixed Plotly chart duplicate ID errors with unique chart keys
- ✅ Implemented session state portfolio data persistence
- ✅ Enhanced risk analysis with proper stock symbol formatting (.NS suffix)
- ✅ Added comprehensive risk visualization dashboard with radar charts
- ✅ Portfolio metrics calculation working correctly with volatility, Sharpe ratio, beta analysis
- ✅ Sector concentration analysis displaying proper sector breakdown
- ✅ Advanced risk visualization charts without errors

**Previous Major Update (July 26, 2025) - COMPLETED:**
- ✅ Implemented comprehensive AI-powered price prediction system with multiple ML models
- ✅ Enhanced background ML processor to analyze and predict ideal purchase prices for all 195+ NSE stocks
- ✅ Added sophisticated price prediction algorithms using Random Forest, Linear Regression, and ensemble models
- ✅ Updated Stock Screener to display ideal prices, price targets, and specific price recommendations
- ✅ Enhanced main dashboard to show top price-based buying opportunities with discount analysis
- ✅ Created comprehensive ML Training page with both signal prediction and price prediction training
- ✅ Integrated price confidence scores and advanced price recommendation filtering
- ✅ Added price target predictions for 1-week and 1-month timeframes
- ✅ Implemented real-time price vs ideal price comparison for optimal purchase timing
- ✅ Enhanced background processing to continuously learn and update price predictions every 5 minutes

**Previous Major Update (July 26, 2025):**
- ✅ Implemented comprehensive background ML processing system for all NSE stocks
- ✅ Enhanced ML training to use 30 stocks from NIFTY 50 instead of only 5 stocks
- ✅ Created background processor that continuously analyzes 195+ stocks with ML signals
- ✅ Added real-time sentiment analysis for 20 major stocks
- ✅ Enhanced Stock Screener with "Quick ML Screener" mode for all processed stocks
- ✅ Integrated ML service with comprehensive market overview dashboard
- ✅ Added sector-wise ML analysis and market sentiment aggregation
- ✅ Implemented robust caching system for ML signals and sentiment data
- ✅ Enhanced main dashboard with live ML feed showing buy/sell signal counts
- ✅ Fixed UI blocking issues with direct cache loading mechanism
- ✅ Created 5-minute processing cycles for continuous real-time updates
- ✅ Successfully deployed background ML runner processing 195+ stocks automatically

**System Performance:**
- Background ML processor analyzes 195+ NSE stocks every 5 minutes with price predictions
- Live market mood calculation using weighted analysis of ML signals, news sentiment, and market conditions
- Real-time sentiment analysis covers major stocks with sector-wise news integration
- Advanced price prediction system provides ideal purchase prices for optimal investment timing
- ML signals and price predictions cached and updated continuously with 97.5% coverage
- Dynamic sector mood visualization with interactive news access for each sector
- Enhanced Stock Screener displays current prices, ideal prices, price targets, and specific buying recommendations
- Live market sentiment responsive to current trading conditions and signal distributions
- Comprehensive filtering by ML signals, price recommendations, confidence levels, and sector analysis

**Previous Update (July 24, 2025):**
- ✅ Implemented Machine Learning signal prediction system
- ✅ Added real-time news sentiment analysis with multi-source aggregation
- ✅ Enhanced purchase signals with AI + Technical + Fundamental scoring
- ✅ Created ML Training page for continuous model improvement
- ✅ Integrated sentiment analysis in Stock Screener and Technical Analysis pages
- ✅ Added market sentiment dashboard on main page
- ✅ Enhanced signal confidence scoring with news correlation

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page architecture
- **UI Pattern**: Sidebar navigation with main content area
- **State Management**: Session-based state management using `st.session_state`
- **Layout**: Wide layout with responsive column-based design
- **Interactivity**: Real-time charts using Plotly, interactive filters and controls

### Backend Architecture
- **Data Processing**: Python-based with pandas for data manipulation
- **Technical Analysis**: TA-Lib integration for technical indicators
- **Caching Strategy**: Streamlit's `@st.cache_data` for performance optimization
- **Modular Design**: Utility-based architecture with separate modules for different functionalities

## Key Components

### Core Pages
1. **Main Dashboard** (`app.py`): AI market sentiment overview, ML model status, and feature guide
2. **Stock Screener** (`pages/1_Stock_Screener.py`): AI-enhanced filtering with ML signals and news sentiment
3. **Technical Analysis** (`pages/2_Technical_Analysis.py`): ML-powered analysis with sentiment integration
4. **Watchlist Management** (`pages/3_Watchlist.py`): Portfolio tracking with AI recommendations
5. **ML Training** (`pages/4_ML_Training.py`): Train and manage machine learning models for signal prediction

### Utility Modules
1. **Data Fetcher** (`utils/data_fetcher.py`): Yahoo Finance API integration for real-time stock data
2. **Technical Analysis** (`utils/technical_analysis.py`): Enhanced indicators with ML and sentiment integration
3. **Fundamental Analysis** (`utils/fundamental_analysis.py`): Financial ratio calculations and scoring
4. **Chart Generator** (`utils/chart_generator.py`): Interactive Plotly chart creation
5. **Stock Lists** (`utils/stock_lists.py`): Indian market indices and sector classifications
6. **ML Signals** (`utils/ml_signals.py`): Machine learning signal prediction and model training
7. **News Sentiment** (`utils/news_sentiment.py`): Real-time news sentiment analysis and aggregation
8. **Price Predictor** (`utils/price_predictor.py`): Advanced AI-powered price prediction system with multiple ML models
9. **Background ML Processor** (`utils/background_ml_processor.py`): Continuous background processing for ML signals and price predictions
10. **ML Service** (`utils/ml_service.py`): Unified service layer for ML operations and data access

### AI and Technical Features Supported
**Machine Learning Signals:**
- Random Forest and Gradient Boosting classifiers
- Feature engineering from technical, fundamental, and sentiment data
- Continuous learning with historical data
- Signal confidence scoring and recommendation generation

**News Sentiment Analysis:**
- Real-time news aggregation from Google News RSS feeds
- TextBlob-based sentiment analysis with polarity scoring
- Multi-source news correlation for reliability
- Market sentiment tracking across major stocks

**Traditional Technical Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Multiple Moving Averages (SMA/EMA)
- Volume Analysis
- Support/Resistance Detection
- Breakout Detection

## Data Flow

1. **Data Ingestion**: Yahoo Finance API provides real-time stock data for NSE-listed stocks
2. **News Aggregation**: Google News RSS feeds collected and processed for sentiment analysis
3. **Data Processing**: Raw OHLCV data cleaned and processed using pandas
4. **Feature Engineering**: Technical, fundamental, and sentiment features created for ML models
5. **Technical Analysis**: Custom pandas-based calculations for technical indicators
6. **ML Prediction**: Trained models generate buy/sell signals with confidence scores
7. **Sentiment Analysis**: TextBlob processes news text for polarity and subjectivity scoring
8. **Signal Fusion**: AI, technical, and fundamental signals combined for final recommendations
9. **Visualization**: Enhanced charts and metrics rendered through Plotly and Streamlit
10. **State Management**: User selections, watchlists, and ML models stored in session state
11. **Caching**: API responses cached for 5 minutes (stock data), 30 minutes (news), 1 hour (fundamentals)

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API client for stock data
- **scikit-learn**: Machine learning models and preprocessing
- **TextBlob**: Natural language processing for sentiment analysis
- **Plotly**: Interactive charting library
- **joblib**: Model persistence and serialization
- **requests**: HTTP client for news data fetching
- **BeautifulSoup**: Web scraping for news content
- **feedparser**: RSS feed parsing for news aggregation

### Data Sources
- **Yahoo Finance**: Primary data source for stock prices, volume, and fundamental data
- **Google News RSS**: Real-time financial news aggregation for sentiment analysis
- **NSE/BSE**: Indian stock exchange data accessed through Yahoo Finance
- **Multi-source News**: Financial news from various publishers aggregated via Google News
- **Real-time Updates**: 5-minute cache for price data, 30-minute cache for news sentiment

### Market Coverage
- **Indices**: NIFTY 50, NIFTY 100, NIFTY 200, and sector-specific indices
- **Sectors**: IT, Banking, Pharma, FMCG, Auto, Energy, and more
- **Stock Universe**: 500+ Indian listed companies

## Deployment Strategy

### Development Environment
- **Platform**: Streamlit Cloud or local development server
- **Dependencies**: Requirements managed through standard Python package management
- **Configuration**: Environment-specific settings in Streamlit config

### Production Considerations
- **Caching**: Multi-level caching strategy for performance
- **Rate Limiting**: Yahoo Finance API rate limit awareness
- **Error Handling**: Comprehensive error handling for data fetching failures
- **Session Management**: Stateful user experience with watchlist persistence

### Scalability Features
- **Modular Architecture**: Easy to extend with new analysis modules
- **Cacheable Functions**: Efficient data retrieval and processing
- **Responsive Design**: Works across different screen sizes
- **API Integration**: Ready for integration with additional data providers

### Key Architectural Decisions

**Problem**: Real-time stock data access for Indian markets
**Solution**: Yahoo Finance API with NSE symbol formatting (.NS suffix)
**Rationale**: Free, reliable, and comprehensive data source with good coverage of Indian stocks

**Problem**: Technical analysis computation performance
**Solution**: TA-Lib integration with Streamlit caching
**Rationale**: Industry-standard technical analysis library with optimized calculations and built-in caching reduces computation overhead

**Problem**: User experience and interactivity
**Solution**: Streamlit multi-page app with session state management
**Rationale**: Rapid development framework with built-in state management and easy deployment, suitable for financial analysis applications

**Problem**: Chart visualization and analysis
**Solution**: Plotly for interactive charts with multiple indicators
**Rationale**: Professional-grade financial charting capabilities with interactivity suitable for technical analysis workflows