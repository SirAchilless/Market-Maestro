import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import utility functions
from utils.ml_signals import train_ml_model, is_model_trained, get_ml_signal
from utils.stock_lists import get_stocks_by_index, get_all_indices, NIFTY_50, NIFTY_100
from utils.data_fetcher import get_stock_data
from utils.news_sentiment import get_news_sentiment_score, get_market_sentiment
from utils.price_predictor import price_predictor
# Price predictor training is now handled automatically in background

st.set_page_config(page_title="ML Training", page_icon="🤖", layout="wide")

def train_comprehensive_price_model():
    """Train the comprehensive price prediction model"""
    st.subheader("🎯 Advanced Price Prediction Model")
    
    # Price model status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if price_predictor.is_trained:
            st.success("✅ Price Model: Trained")
        else:
            st.warning("⚠️ Price Model: Not Trained")
    
    with col2:
        # Check if model files exist
        import os
        model_files_exist = os.path.exists("models/price_predictor.joblib")
        if model_files_exist:
            st.info("📁 Model Files: Available")
        else:
            st.warning("📁 Model Files: Missing")
    
    with col3:
        if price_predictor.performance_metrics:
            accuracy = price_predictor.performance_metrics.get('r2_score', 0)
            st.metric("Model R² Score", f"{accuracy:.3f}")
        else:
            st.metric("Model Performance", "Not Tested")
    
    st.markdown("---")
    
    # Training controls
    st.markdown("**Train Price Prediction Model:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Price Prediction Features:**
        - Historical price patterns and trends
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volume analysis and momentum
        - Market volatility patterns
        - Fundamental ratios integration
        - Cross-stock correlation analysis
        """)
    
    with col2:
        if st.button("🚀 Train Price Model", type="primary", help="Train advanced price prediction model"):
            with st.spinner("Training price prediction model..."):
                try:
                    # Train the price predictor
                    from utils.stock_lists import NIFTY_50
                    training_stocks = NIFTY_50[:20]  # Use top 20 stocks for training
                    
                    # Train model
                    results = price_predictor.train_model(training_stocks)
                    
                    if results.get('success', False):
                        st.success("✅ Price prediction model trained successfully!")
                        st.info(f"Trained on {len(training_stocks)} stocks")
                        
                        # Show performance metrics
                        if price_predictor.performance_metrics:
                            metrics = price_predictor.performance_metrics
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
                            with col_b:
                                st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                            with col_c:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error training price model: {e}")
        
        # Test price predictions
        if price_predictor.is_trained:
            st.markdown("**Test Price Predictions:**")
            
            # Stock selection for testing
            test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS']
            selected_stock = st.selectbox("Select stock to test:", test_stocks, key="price_test_stock")
            
            if st.button("🔍 Test Price Prediction", help="Get price prediction for selected stock"):
                try:
                    prediction = price_predictor.predict_price(selected_stock)
                    
                    if prediction.get('success', False):
                        st.success("✅ Price prediction generated!")
                        
                        pred_data = prediction['prediction']
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Current Price", f"₹{pred_data.get('current_price', 0):.2f}")
                        with col_b:
                            st.metric("Predicted Price", f"₹{pred_data.get('predicted_price', 0):.2f}")
                        with col_c:
                            confidence = pred_data.get('confidence', 0)
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Show recommendation
                        recommendation = pred_data.get('recommendation', 'Hold')
                        if recommendation == 'Buy':
                            st.success(f"📈 Recommendation: {recommendation}")
                        elif recommendation == 'Sell':
                            st.error(f"📉 Recommendation: {recommendation}")
                        else:
                            st.info(f"⏹️ Recommendation: {recommendation}")
                    else:
                        st.error(f"❌ Prediction failed: {prediction.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error testing prediction: {e}")

def show_model_status():
    """Show detailed model status information"""
    st.subheader("📊 Model Status Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Signal Prediction Model:**")
        if is_model_trained():
            st.success("✅ Ready for signal predictions")
            st.info("Features: Technical indicators, news sentiment, market patterns")
        else:
            st.warning("⚠️ Not trained - train on Signal Prediction tab")
    
    with col2:
        st.markdown("**Price Prediction Model:**")
        if price_predictor.is_trained:
            st.success("✅ Ready for price predictions")
            st.info("Features: Price patterns, volume analysis, correlation data")
        else:
            st.warning("⚠️ Not trained - train on Price Prediction tab")

def test_model_predictions():
    """Test both signal and price prediction models"""
    st.subheader("🧪 Combined Model Testing")
    
    # Stock selection
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS']
    selected_stock = st.selectbox("Select stock for comprehensive analysis:", test_stocks, key="combined_test")
    
    if st.button("🔬 Run Complete Analysis", type="primary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Signal Prediction:**")
            try:
                signal_result = get_ml_signal(selected_stock)
                
                if signal_result:
                    signal = signal_result.get('signal', 'Hold')
                    confidence = signal_result.get('confidence', 0)
                    
                    if signal in ['Strong Buy', 'Buy']:
                        st.success(f"Signal: {signal} (Confidence: {confidence}%)")
                    elif signal in ['Sell', 'Strong Sell']:
                        st.error(f"Signal: {signal} (Confidence: {confidence}%)")
                    else:
                        st.info(f"Signal: {signal} (Confidence: {confidence}%)")
                else:
                    st.warning("Signal prediction not available")
                    
            except Exception as e:
                st.error(f"Signal prediction error: {e}")
        
        with col2:
            st.markdown("**🎯 Price Prediction:**")
            try:
                if price_predictor.is_trained:
                    price_result = price_predictor.predict_price(selected_stock)
                    
                    if price_result.get('success', False):
                        pred_data = price_result['prediction']
                        current_price = pred_data.get('current_price', 0)
                        predicted_price = pred_data.get('predicted_price', 0)
                        price_change = ((predicted_price - current_price) / current_price) * 100
                        
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        st.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{price_change:+.2f}%")
                        
                        recommendation = pred_data.get('recommendation', 'Hold')
                        if recommendation == 'Buy':
                            st.success(f"Price Recommendation: {recommendation}")
                        elif recommendation == 'Sell':
                            st.error(f"Price Recommendation: {recommendation}")
                        else:
                            st.info(f"Price Recommendation: {recommendation}")
                    else:
                        st.error("Price prediction failed")
                else:
                    st.warning("Price model not trained")
                    
            except Exception as e:
                st.error(f"Price prediction error: {e}")

st.title("🤖 Advanced ML Training & Price Prediction")
st.markdown("Train sophisticated AI models for accurate trading signals and ideal purchase price predictions")

# Create tabs for different ML training modules
tab1, tab2, tab3 = st.tabs(["📈 Signal Prediction", "🎯 Price Prediction", "🧪 Test Models"])

with tab1:
    st.subheader("📈 Trading Signal Prediction Model")
    
    # Check model status
    model_status = is_model_trained()
    
    # Display model status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_status:
            st.success("✅ ML Model: Trained")
        else:
            st.warning("⚠️ ML Model: Not Trained")
    
    with col2:
        market_sentiment = get_market_sentiment()
        mood_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
        st.metric("Market Sentiment", 
                 f"{mood_color.get(market_sentiment['market_mood'], '🟡')} {market_sentiment['market_mood'].title()}")
    
    with col3:
        st.metric("Sentiment Score", f"{market_sentiment['market_sentiment_score']}/100")
    
    st.markdown("---")

    # Training Section
    st.subheader("🎯 Model Training")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Enhanced ML Features:**
    - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
    - Fundamental analysis (PE, ROE, Debt/Equity ratios)
    - News sentiment analysis from multiple sources
    - Market timing and seasonal factors
    - Volume and breakout patterns
    """)

with col2:
    training_index = st.selectbox(
        "Select Training Dataset",
        options=get_all_indices(),
        index=0,
        help="Choose stocks for training the ML model"
    )

# Training controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Train New Model", type="primary"):
        training_stocks = get_stocks_by_index(training_index)
        
        if training_stocks:
            with st.spinner("Training ML model... This may take several minutes."):
                success = train_ml_model(training_stocks)
                
                if success:
                    st.success("Model trained successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Model training failed. Please try again.")
        else:
            st.error("No stocks found for the selected index")

with col2:
    if st.button("🔄 Retrain Model"):
        if model_status:
            training_stocks = NIFTY_100  # Use larger dataset for retraining
            
            with st.spinner("Retraining model with updated data..."):
                success = train_ml_model(training_stocks)
                
                if success:
                    st.success("Model retrained successfully!")
                    st.rerun()
                else:
                    st.error("Model retraining failed")
        else:
            st.warning("No existing model to retrain. Please train a new model first.")

with col3:
    st.info(f"Training on {len(get_stocks_by_index(training_index))} stocks")

# Model Performance Section
if model_status:
    st.subheader("📊 Model Performance & Signals")
    
    # Test the model on a sample stock
    test_stock = st.selectbox(
        "Test ML Predictions",
        options=NIFTY_50[:10],  # Top 10 stocks for testing
        help="Select a stock to test ML predictions"
    )
    
    if st.button("🔍 Analyze Stock"):
        with st.spinner(f"Analyzing {test_stock} with ML and sentiment..."):
            # Get stock data
            stock_data = get_stock_data(test_stock, period='6mo')
            
            if not stock_data.empty:
                # Get ML prediction
                ml_result = get_ml_signal(stock_data, test_stock)
                
                # Get news sentiment
                sentiment_result = get_news_sentiment_score(test_stock)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🤖 ML Prediction")
                    signal_color = {
                        'Strong Buy': 'green',
                        'Buy': 'orange',
                        'Hold': 'blue',
                        'Sell': 'red'
                    }.get(ml_result['signal'], 'gray')
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; border-radius: 10px; 
                                background-color: {signal_color}20; border: 2px solid {signal_color}'>
                        <h3 style='color: {signal_color}; margin: 0;'>{ml_result['signal']}</h3>
                        <p style='margin: 5px 0;'>Confidence: {ml_result['confidence']:.1%}</p>
                        <p style='margin: 5px 0;'>ML Score: {ml_result['ml_score']}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📰 News Sentiment")
                    sentiment_label = sentiment_result['sentiment_label']
                    sentiment_score = sentiment_result['sentiment_score']
                    
                    sentiment_color = {
                        'bullish': 'green',
                        'bearish': 'red',
                        'neutral': 'blue'
                    }.get(sentiment_label, 'gray')
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; border-radius: 10px; 
                                background-color: {sentiment_color}20; border: 2px solid {sentiment_color}'>
                        <h3 style='color: {sentiment_color}; margin: 0;'>{sentiment_label.title()}</h3>
                        <p style='margin: 5px 0;'>Score: {sentiment_score}/100</p>
                        <p style='margin: 5px 0;'>Articles: {sentiment_result['article_count']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("### 🎯 Combined Signal")
                    
                    # Calculate combined signal strength
                    ml_strength = (ml_result['ml_score'] - 50) / 50  # -1 to 1
                    sentiment_strength = (sentiment_score - 50) / 50  # -1 to 1
                    
                    combined_strength = (ml_strength + sentiment_strength) / 2
                    combined_score = (combined_strength + 1) * 50  # 0 to 100
                    
                    if combined_score > 70:
                        combined_signal = "Strong Buy"
                        signal_color = "green"
                    elif combined_score > 60:
                        combined_signal = "Buy"
                        signal_color = "orange"
                    elif combined_score < 30:
                        combined_signal = "Sell"
                        signal_color = "red"
                    elif combined_score < 40:
                        combined_signal = "Weak Sell"
                        signal_color = "orange"
                    else:
                        combined_signal = "Hold"
                        signal_color = "blue"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; border-radius: 10px; 
                                background-color: {signal_color}20; border: 2px solid {signal_color}'>
                        <h3 style='color: {signal_color}; margin: 0;'>{combined_signal}</h3>
                        <p style='margin: 5px 0;'>Combined Score: {combined_score:.1f}/100</p>
                        <p style='margin: 5px 0;'>Confidence: High</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed analysis
                st.markdown("### 📋 Detailed Analysis")
                
                # Calculate technical and fundamental scores
                from utils.technical_analysis import get_technical_score, get_technical_signal
                from utils.fundamental_analysis import get_fundamental_data, get_fundamental_score
                
                # Get technical score
                try:
                    tech_score = get_technical_score(stock_data)
                    tech_signal = get_technical_signal(tech_score)
                    tech_confidence = min(tech_score / 100 * 2, 1.0)  # Convert to confidence ratio
                except Exception as e:
                    tech_score = 0
                    tech_signal = "Error"
                    tech_confidence = 0
                
                # Get fundamental score
                try:
                    fundamental_data = get_fundamental_data(test_stock)
                    fund_score = get_fundamental_score(fundamental_data)
                    if fund_score >= 75:
                        fund_signal = "Strong Buy"
                    elif fund_score >= 60:
                        fund_signal = "Buy"
                    elif fund_score >= 45:
                        fund_signal = "Hold"
                    elif fund_score >= 30:
                        fund_signal = "Weak Sell"
                    else:
                        fund_signal = "Sell"
                    fund_confidence = min(fund_score / 100 * 1.5, 1.0)  # Convert to confidence ratio
                except Exception as e:
                    fund_score = 0
                    fund_signal = "Error"
                    fund_confidence = 0
                
                analysis_data = {
                    'Component': ['ML Prediction', 'News Sentiment', 'Technical Score', 'Fundamental Score'],
                    'Signal': [
                        ml_result['signal'],
                        sentiment_result['sentiment_label'].title(),
                        tech_signal,
                        fund_signal
                    ],
                    'Score': [
                        f"{ml_result['ml_score']:.1f}/100",
                        f"{sentiment_result['sentiment_score']:.1f}/100",
                        f"{tech_score:.1f}/100",
                        f"{fund_score:.1f}/100"
                    ],
                    'Confidence': [
                        f"{ml_result['confidence']:.1%}",
                        f"{sentiment_result['confidence']:.1%}",
                        f"{tech_confidence:.1%}",
                        f"{fund_confidence:.1%}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                
                # Recent news articles
                if sentiment_result.get('recent_articles'):
                    st.markdown("### 📰 Recent News Articles")
                    
                    for i, article in enumerate(sentiment_result['recent_articles'][:3]):
                        with st.expander(f"📄 Article {i+1}: {article['title'][:80]}..."):
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Published:** {article['published']}")
                            st.write(f"**Summary:** {article['summary']}")
                            if article['link']:
                                st.markdown(f"[Read Full Article]({article['link']})")
                
                # ML Recommendation
                st.markdown("### 💡 AI Recommendation")
                
                recommendation_text = f"""
                **{test_stock} Analysis Summary:**
                
                - **ML Signal:** {ml_result['signal']} with {ml_result['confidence']:.1%} confidence
                - **News Sentiment:** {sentiment_result['sentiment_label'].title()} based on {sentiment_result['article_count']} articles
                - **Combined Recommendation:** {combined_signal}
                
                **Key Insights:**
                - {ml_result['recommendation']}
                - News analysis shows {sentiment_result['sentiment_label']} sentiment
                - Overall market mood is {market_sentiment['market_mood']}
                
                **Risk Assessment:** {'Low' if combined_score > 60 or combined_score < 40 else 'Medium'}
                """
                
                st.info(recommendation_text)
                
            else:
                st.error(f"Could not fetch data for {test_stock}")

else:
    st.subheader("🎓 Model Training Required")
    
    st.info("""
    **No trained model available.** Please train the ML model to get AI-powered trading signals.
    
    **Training Process:**
    1. Select a stock index for training data
    2. Click "Train New Model" 
    3. Wait for training to complete (5-10 minutes)
    4. Start getting ML-powered predictions!
    
    **What the Model Learns:**
    - Technical pattern recognition
    - News sentiment correlation with price movements
    - Fundamental analysis integration
    - Market timing optimization
    """)

# Model Information Section
st.markdown("---")
st.subheader("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Features Used for Training:**
    - **Technical Indicators:** RSI, MACD, Bollinger Bands, Moving Averages
    - **Price Patterns:** Support/Resistance levels, Breakouts, Volatility
    - **Volume Analysis:** Volume ratios and trends
    - **Fundamental Data:** PE ratio, ROE, Debt/Equity, Profit margins
    - **News Sentiment:** Real-time sentiment from financial news
    - **Market Timing:** Seasonal and cyclical factors
    """)

with col2:
    st.markdown("""
    **Signal Classifications:**
    - **Strong Buy:** High confidence bullish prediction
    - **Buy:** Moderate bullish signal
    - **Hold:** Neutral market conditions
    - **Weak Sell:** Caution advised
    - **Sell:** Bearish signal detected
    
    **Confidence Levels:** Based on model certainty and news sentiment reliability
    """)

# Disclaimer
st.markdown("---")
with tab2:
    # Price Prediction Tab
    train_comprehensive_price_model()
    st.markdown("---")
    show_model_status()

with tab3:
    # Test Models Tab
    st.subheader("🧪 Test Both Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Signal Prediction Test")
        if model_status:
            st.success("Signal model is ready for testing")
        else:
            st.warning("Please train signal model first")
    
    with col2:
        st.markdown("### 🎯 Price Prediction Test")
        if price_predictor.is_trained:
            st.success("Price model is ready for testing")
        else:
            st.warning("Please train price model first")
    
    # Combined testing interface
    if model_status and price_predictor.is_trained:
        test_model_predictions()

# Disclaimer
st.markdown("---")
st.warning("""
**Disclaimer:** These ML models are for educational and analysis purposes only. 
They combine technical analysis with news sentiment but should not be considered as financial advice. 
Always conduct your own research and consult with financial advisors before making investment decisions.
""")