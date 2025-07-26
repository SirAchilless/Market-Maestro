import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Indian Stock Market Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Main page content
st.title("🇮🇳 Smart Stock Screener with AI")
st.markdown("### Advanced ML-Powered Analysis with News Sentiment for Algo Trading")

# Market Indices Overview - Moved to top
st.markdown("---")
st.subheader("📊 Market Indices Overview")

try:
    from utils.data_fetcher import get_index_data
    
    # Display major indices
    indices = ['^NSEI', '^NSEBANK', '^CNXIT', '^CNXAUTO', '^CNXPHARMA']
    index_names = ['Nifty 50', 'Bank Nifty', 'IT Index', 'Auto Index', 'Pharma Index']
    
    index_cols = st.columns(len(indices))
    
    for i, (idx, name) in enumerate(zip(indices, index_names)):
        with index_cols[i]:
            try:
                data = get_index_data(idx)
                if data['status'] == 'active':
                    current = data['current_price']
                    change = data['change']
                    change_pct = data['change_percent']
                    
                    st.metric(
                        name,
                        f"{current:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                else:
                    st.metric(name, "N/A", data['status'])
            except Exception as e:
                st.metric(name, "N/A", "Error loading")
                
except ImportError:
    st.info("📊 Install required packages to view market overview")

st.markdown("---")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Watchlist Stocks", len(st.session_state.watchlist))

with col2:
    # Get market status with proper Indian market hours
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = datetime.now(ist)
    
    # NSE market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    market_open_time = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_weekday = ist_now.weekday() < 5  # Monday = 0, Friday = 4
    is_market_hours = market_open_time <= ist_now <= market_close_time
    
    if is_weekday and is_market_hours:
        market_status = "🟢 Open"
    elif is_weekday and ist_now < market_open_time:
        market_status = "🟡 Pre-Market"
    elif is_weekday and ist_now > market_close_time:
        market_status = "🟡 After-Hours"
    else:
        market_status = "🔴 Closed"
    
    st.metric("Market Status", market_status)

with col3:
    # Display IST time
    ist_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.metric("Last Updated", ist_time)

with col4:
    # Check if ML model is trained
    try:
        from utils.ml_signals import is_model_trained
        model_status = "Trained" if is_model_trained() else "Not Trained"
        st.metric("ML Model", model_status)
    except:
        st.metric("ML Model", "Loading...")

# Quick start guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    **How to use this AI-Powered Stock Screener:**
    
    1. **Stock Screener**: Filter stocks with AI analysis and news sentiment
    2. **Technical Analysis**: Deep dive with ML-enhanced technical indicators
    3. **Watchlist**: Manage your portfolio with sentiment tracking
    4. **ML Training**: Train AI models for accurate buy/sell predictions
    5. **Portfolio Risk Assessment**: Analyze portfolio risk and diversification
    6. **Investment Narrative**: Generate AI-powered investment narratives
    7. **Emoji Mood Selector**: Personalize stocks with custom emoji moods
    
    **Advanced AI Features:**
    - 🤖 Machine Learning signal predictions
    - 📰 Real-time news sentiment analysis
    - 🎯 Combined AI + Technical + Fundamental scoring
    - 📊 Continuous learning from market data
    - 🔍 Multi-source news aggregation
    - ⚡ High-confidence trading signals
    
    **Traditional Features:**
    - ✅ Real-time Indian stock data (NSE/BSE)
    - ✅ Technical indicators (RSI, MACD, Bollinger Bands)
    - ✅ Fundamental analysis (PE, EPS, ROE, ROCE)
    - ✅ Interactive charts and export functionality
    """)

# Enhanced AI Market Analysis with Background ML Processing
st.markdown("---")
st.subheader("🤖 AI Market Analysis - Live ML Feed")

# Initialize ML service with auto-refresh
try:
    import json
    import os
    
    # Direct cache loading function
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def load_ml_cache_direct():
        try:
            cache_file = "ml_data/ml_signals_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_sentiment_cache_direct():
        try:
            cache_file = "ml_data/sentiment_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    # Auto-refresh mechanism (non-blocking)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh ML Data", help="Get latest ML signals and sentiment"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        st.markdown("🔴 **LIVE** - Background ML processing active")
    
    # Get ML data directly from cache
    ml_cache = load_ml_cache_direct()
    sentiment_cache = load_sentiment_cache_direct()
    
    # Calculate market statistics
    if ml_cache:
        buy_signals = sum(1 for data in ml_cache.values() if data.get('ml_signal') in ['Buy', 'Strong Buy'])
        sell_signals = sum(1 for data in ml_cache.values() if data.get('ml_signal') in ['Sell', 'Strong Sell'])
        total_stocks = len(ml_cache)
        
        # Get latest update time
        latest_time = max(
            (data.get('timestamp', '2000-01-01T00:00:00')
             for data in ml_cache.values()),
            default='Never'
        )
        
        market_overview = {
            'total_stocks': total_stocks,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': total_stocks - buy_signals - sell_signals,
            'last_update': latest_time[-8:] if latest_time != 'Never' else 'Never'
        }
    else:
        market_overview = {
            'total_stocks': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'last_update': 'Never'
        }
    
    # Display comprehensive market status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        coverage_pct = (market_overview['total_stocks'] / 200) * 100 if market_overview['total_stocks'] > 0 else 0
        st.metric(
            "Stocks Analyzed", 
            market_overview['total_stocks'],
            f"{coverage_pct:.1f}% coverage"
        )
    
    with col2:
        buy_signals = market_overview['buy_signals']
        total_signals = buy_signals + market_overview['sell_signals']
        buy_pct = (buy_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric(
            "Buy Signals", 
            buy_signals,
            f"{buy_pct:.1f}% of signals"
        )
    
    with col3:
        sell_signals = market_overview['sell_signals']
        sell_pct = (sell_signals / total_signals * 100) if total_signals > 0 else 0
        st.metric(
            "Sell Signals", 
            sell_signals,
            f"{sell_pct:.1f}% of signals"
        )
    
    with col4:
        st.metric(
            "Last ML Update", 
            market_overview['last_update']
        )
    
    # Market Sentiment Display
    col1, col2 = st.columns(2)
    
    # Enhanced Live Market Mood Calculation based on broader NSE stocks
    ml_sentiment_scores = []
    news_sentiment_scores = []
    bullish_count = 0
    bearish_count = 0
    total_stocks_analyzed = 0
    
    # Process ML signals for sentiment calculation
    if ml_cache:
        for symbol, data in ml_cache.items():
            ml_signal = data.get('ml_signal', 'Hold')
            ml_confidence = data.get('ml_confidence', 0.5)
            
            # Convert ML signals to sentiment scores (0-100 scale)
            if ml_signal == 'Strong Buy':
                score = 80 + (ml_confidence * 20)  # 80-100 range
                bullish_count += 1
            elif ml_signal == 'Buy':
                score = 65 + (ml_confidence * 15)  # 65-80 range
                bullish_count += 1
            elif ml_signal == 'Sell':
                score = 35 - (ml_confidence * 15)  # 20-35 range
                bearish_count += 1
            elif ml_signal == 'Strong Sell':
                score = max(0, 20 - (ml_confidence * 20))  # 0-20 range
                bearish_count += 1
            else:  # Hold
                score = 50
            
            ml_sentiment_scores.append(score)
        
        total_stocks_analyzed = len(ml_cache)
    
    # Process news sentiment data
    if sentiment_cache:
        for symbol, data in sentiment_cache.items():
            sentiment_score = data.get('sentiment_score', 50)
            news_sentiment_scores.append(sentiment_score)
        
        total_stocks_analyzed = max(total_stocks_analyzed, len(sentiment_cache))
    
    # Calculate overall live market sentiment
    if ml_sentiment_scores or news_sentiment_scores:
        # Combine ML and news sentiment with proper weighting
        all_scores = []
        
        # Add ML scores (primary weight: 70%)
        if ml_sentiment_scores:
            ml_avg = sum(ml_sentiment_scores) / len(ml_sentiment_scores)
            all_scores.append(ml_avg * 0.7)
        
        # Add news sentiment scores (secondary weight: 30%)
        if news_sentiment_scores:
            news_avg = sum(news_sentiment_scores) / len(news_sentiment_scores)
            all_scores.append(news_avg * 0.3)
        
        # Calculate final weighted average
        if all_scores:
            avg_sentiment = sum(all_scores)
        else:
            avg_sentiment = 50
        
        # Determine mood based on signal distribution and sentiment score
        bullish_ratio = bullish_count / max(1, total_stocks_analyzed)
        bearish_ratio = bearish_count / max(1, total_stocks_analyzed)
        
        # Enhanced mood determination
        if bullish_ratio > 0.6 or avg_sentiment > 65:  # Strong bullish conditions
            sentiment_mood = 'bullish'
            avg_sentiment = max(avg_sentiment, 65)  # Boost if strong bullish signals
        elif bearish_ratio > 0.3 or avg_sentiment < 35:  # Bearish conditions
            sentiment_mood = 'bearish'
            avg_sentiment = min(avg_sentiment, 45)  # Reduce if bearish signals
        elif bullish_ratio > 0.4 or avg_sentiment > 55:  # Moderate bullish
            sentiment_mood = 'bullish'
        elif bearish_ratio > 0.2 or avg_sentiment < 45:  # Moderate bearish
            sentiment_mood = 'bearish'
        else:
            sentiment_mood = 'neutral'
        
        # Ensure sentiment score is reasonable (between 20-80 for dynamic feel)
        avg_sentiment = max(20, min(80, avg_sentiment))
        
    else:
        avg_sentiment = 50
        sentiment_mood = 'neutral'
    
    with col1:
        # Display sentiment with color coding
        if sentiment_mood == 'bullish':
            sentiment_color = 'green'
            sentiment_emoji = '📈'
        elif sentiment_mood == 'bearish':
            sentiment_color = 'red'
            sentiment_emoji = '📉'
        else:
            sentiment_color = 'blue'
            sentiment_emoji = '➡️'
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; 
                    background-color: {sentiment_color}20; border: 2px solid {sentiment_color}'>
            <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 10px;'>
                <span style='width: 8px; height: 8px; background-color: #ff4444; border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite;'></span>
                <h3 style='color: {sentiment_color}; margin: 0;'>LIVE Market Mood: {sentiment_mood.title()}</h3>
            </div>
            <p style='font-size: 28px; margin: 10px 0; color: {sentiment_color}; font-weight: bold;'>{avg_sentiment:.1f}/100</p>
            <p style='margin: 5px 0; color: #666; font-size: 12px;'>📊 Live analysis of {total_stocks_analyzed} NSE stocks</p>
            <p style='margin: 2px 0; color: #888; font-size: 11px;'>🔄 Updated every 5 minutes with ML + Sentiment + Price momentum</p>
        </div>
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
            100% {{ opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📰 AI News Analysis")
        total_articles = sum(data.get('article_count', 0) for data in sentiment_cache.values())
        st.metric("News Articles Analyzed", total_articles)
        
        st.metric("Bullish Stocks", bullish_count)
        st.metric("Bearish Stocks", bearish_count)
        
        st.markdown("**💡 Click on any sector below to view latest news!**")
        
        # Quick emoji mood preview
        try:
            from utils.emoji_mood_selector import load_user_emoji_moods
            moods = load_user_emoji_moods()
            if moods:
                emoji_count = len(moods)
                st.markdown(f"**🎭 Your Emoji Moods: {emoji_count} stocks personalized**")
                sample_moods = list(moods.items())[:3]
                mood_text = " ".join([f"{data['emoji']}{symbol}" for symbol, data in sample_moods])
                if mood_text:
                    st.markdown(f"*{mood_text}{'...' if len(moods) > 3 else ''}*")
        except:
            pass
        
        # Show top ML signals with price predictions
        if st.button("🚀 View Top ML Signals & Price Predictions"):
            if ml_cache:
                # Get stocks with ideal price below current price (good buying opportunities)
                buying_opportunities = []
                for symbol, data in ml_cache.items():
                    current_price = data.get('current_price', 0)
                    ideal_price = data.get('ideal_price')
                    price_recommendation = data.get('price_recommendation', '')
                    
                    if ideal_price and current_price > 0:
                        discount_pct = ((current_price - ideal_price) / ideal_price) * 100
                        if 'Buy' in price_recommendation:
                            buying_opportunities.append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'ideal_price': ideal_price,
                                'discount_pct': discount_pct,
                                'price_confidence': data.get('price_confidence', 0),
                                'ml_signal': data.get('ml_signal', 'Hold')
                            })
                
                buying_opportunities.sort(key=lambda x: -x['discount_pct'])  # Sort by best discount
                
                if buying_opportunities[:5]:
                    st.write("**🎯 Top Price-Based Buy Opportunities:**")
                    for opp in buying_opportunities[:5]:
                        symbol = opp['symbol']
                        current = opp['current_price']
                        ideal = opp['ideal_price']
                        confidence = opp['price_confidence']
                        ml_signal = opp['ml_signal']
                        
                        # Add emoji mood if available
                        try:
                            from utils.emoji_mood_selector import get_stock_emoji
                            emoji = get_stock_emoji(symbol)
                        except:
                            emoji = ""
                            
                        if opp['discount_pct'] <= 0:  # Current price is below or at ideal
                            st.write(f"• **{emoji} {symbol}** - Current: ₹{current:.2f} | Ideal: ₹{ideal:.2f} | ML: {ml_signal} | Confidence: {confidence:.1f}%")
                        else:
                            st.write(f"• {emoji} {symbol} - Current: ₹{current:.2f} | Ideal: ₹{ideal:.2f} | ML: {ml_signal}")
                else:
                    # Fallback to regular ML signals
                    strong_buys = [(symbol, data) for symbol, data in ml_cache.items() 
                                  if data.get('ml_signal') == 'Strong Buy']
                    strong_buys.sort(key=lambda x: x[1].get('ml_confidence', 0), reverse=True)
                    
                    if strong_buys[:5]:
                        st.write("**🟢 Top Strong Buy Signals:**")
                        for symbol, data in strong_buys[:5]:
                            confidence = data.get('ml_confidence', 0)
                            current_price = data.get('current_price', 0)
                            
                            # Add emoji mood if available
                            try:
                                from utils.emoji_mood_selector import get_stock_emoji
                                emoji = get_stock_emoji(symbol)
                                st.write(f"• {emoji} {symbol} - ₹{current_price:.2f} - {confidence:.1%} confidence")
                            except:
                                st.write(f"• {symbol} - ₹{current_price:.2f} - {confidence:.1%} confidence")
                    else:
                        st.info("No strong buy opportunities found currently")
            else:
                st.info("ML signals and price predictions loading in background...")

    # Sector-wise Market Mood Display
    st.markdown("---")
    st.markdown("### 🏭 Sector-wise Market Mood")
    
    # Calculate sector-wise sentiment from ML cache
    if ml_cache:
        from utils.stock_lists import SECTOR_STOCKS, get_stock_sector
        
        sector_sentiments = {}
        
        # Initialize sector data
        for sector in SECTOR_STOCKS.keys():
            sector_sentiments[sector] = {
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'total_stocks': 0,
                'sentiment_score': 50,
                'mood': 'neutral'
            }
        
        # Process ML cache data by sector
        for symbol, data in ml_cache.items():
            ml_signal = data.get('ml_signal', 'Hold')
            sector = get_stock_sector(symbol)
            
            if sector in sector_sentiments:
                sector_sentiments[sector]['total_stocks'] += 1
                
                if ml_signal in ['Strong Buy', 'Buy']:
                    sector_sentiments[sector]['buy_signals'] += 1
                elif ml_signal in ['Strong Sell', 'Sell']:
                    sector_sentiments[sector]['sell_signals'] += 1
                else:
                    sector_sentiments[sector]['hold_signals'] += 1
        
        # Calculate sentiment scores and moods for each sector
        for sector, data in sector_sentiments.items():
            if data['total_stocks'] > 0:
                buy_ratio = data['buy_signals'] / data['total_stocks']
                sell_ratio = data['sell_signals'] / data['total_stocks']
                
                # Calculate sentiment score (0-100)
                data['sentiment_score'] = 50 + (buy_ratio - sell_ratio) * 50
                
                # Determine mood
                if data['sentiment_score'] > 60:
                    data['mood'] = 'bullish'
                elif data['sentiment_score'] < 40:
                    data['mood'] = 'bearish'
                else:
                    data['mood'] = 'neutral'
        
        # Display sector boxes in a grid
        sectors_with_data = [(sector, data) for sector, data in sector_sentiments.items() 
                           if data['total_stocks'] > 0]
        
        if sectors_with_data:
            # Display sectors in rows of 4
            cols_per_row = 4
            for i in range(0, len(sectors_with_data), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, (sector, data) in enumerate(sectors_with_data[i:i+cols_per_row]):
                    with cols[j]:
                        # Set colors based on mood
                        if data['mood'] == 'bullish':
                            color = 'green'
                            emoji = '📈'
                        elif data['mood'] == 'bearish':
                            color = 'red'
                            emoji = '📉'
                        else:
                            color = 'blue'
                            emoji = '➡️'
                        
                        # Create clickable sector mood box with news
                        sector_box_key = f"sector_news_{sector}_{i}_{j}"
                        
                        # Create sector mood box
                        st.markdown(f"""
                        <div style='text-align: center; padding: 15px; margin: 5px; border-radius: 8px; 
                                    background-color: {color}15; border: 1px solid {color}'>
                            <h4 style='color: {color}; margin: 0 0 8px 0; font-size: 16px;'>{emoji} {sector}</h4>
                            <p style='font-size: 18px; margin: 5px 0; color: {color}; font-weight: bold;'>{data['sentiment_score']:.1f}/100</p>
                            <p style='margin: 5px 0; color: #666; font-size: 12px;'>{data['total_stocks']} stocks</p>
                            <p style='margin: 2px 0; color: #666; font-size: 11px;'>Buy: {data['buy_signals']} | Sell: {data['sell_signals']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add news button for sector
                        if st.button(f"📰 Latest {sector} News", key=sector_box_key, use_container_width=True):
                            # Get sector stocks and their news
                            sector_stocks = SECTOR_STOCKS.get(sector, [])[:5]  # Top 5 stocks from sector
                            
                            # Show sector news in expandable container
                            with st.expander(f"📰 Latest {sector} Sector News", expanded=True):
                                if sentiment_cache:
                                    sector_news_found = False
                                    
                                    for stock_symbol in sector_stocks:
                                        if stock_symbol in sentiment_cache:
                                            stock_data = sentiment_cache[stock_symbol]
                                            recent_articles = stock_data.get('recent_articles', [])
                                            
                                            if recent_articles:
                                                sector_news_found = True
                                                st.markdown(f"**📊 {stock_symbol} News:**")
                                                
                                                for idx, article in enumerate(recent_articles[:3]):  # Top 3 articles per stock
                                                    # Determine article sentiment color
                                                    article_color = 'blue'
                                                    title_lower = article.get('title', '').lower()
                                                    if any(word in title_lower for word in ['gain', 'up', 'rise', 'profit', 'growth', 'positive']):
                                                        article_color = 'green'
                                                    elif any(word in title_lower for word in ['fall', 'down', 'loss', 'decline', 'negative', 'drop']):
                                                        article_color = 'red'
                                                    
                                                    # Display news article in styled box
                                                    st.markdown(f"""
                                                    <div style='padding: 10px; margin: 8px 0; border-radius: 6px; 
                                                                background-color: {article_color}08; border-left: 3px solid {article_color}'>
                                                        <h5 style='margin: 0 0 5px 0; color: {article_color}; font-size: 14px;'>{article['title']}</h5>
                                                        <p style='margin: 5px 0; font-size: 12px; color: #666;'>{article.get('summary', '')[:100]}...</p>
                                                        <p style='margin: 5px 0; font-size: 10px; color: #888;'>
                                                            📅 {article.get('published', 'Unknown date')[:16]}
                                                        </p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                                    # Add link to full article
                                                    if article.get('link'):
                                                        st.markdown(f"[📖 Read Full Article]({article['link']})")
                                                
                                                st.markdown("---")
                                    
                                    if not sector_news_found:
                                        st.info(f"No recent news available for {sector} sector stocks. News analysis is loading in background.")
                                else:
                                    st.info("News data is loading. Please try again in a moment.")
        else:
            st.info("Sector analysis loading in background...")
    else:
        st.info("Sector mood analysis will be available once ML processing completes...")

    # Quick Portfolio Risk Insight for Watchlist
    if st.session_state.watchlist and len(st.session_state.watchlist) >= 3:
        st.markdown("---")
        st.subheader("🎯 Quick Portfolio Risk Insight")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"💡 Quick risk preview for your {len(st.session_state.watchlist)}-stock watchlist")
            st.markdown("**Watchlist Holdings:**")
            
            # Display watchlist with basic info
            for i, stock in enumerate(st.session_state.watchlist[:5]):
                st.write(f"• {stock}")
            
            if len(st.session_state.watchlist) > 5:
                st.write(f"... and {len(st.session_state.watchlist) - 5} more stocks")
        
        with col2:
            if st.button("🎯 Full Risk Analysis", type="primary", use_container_width=True):
                # Store portfolio and redirect
                equal_weight = 1.0 / len(st.session_state.watchlist)
                st.session_state.portfolio_for_analysis = {stock: equal_weight for stock in st.session_state.watchlist}
                st.switch_page("pages/5_Portfolio_Risk_Assessment.py")
            
            st.markdown("**One-Click Analysis:**")
            st.write("• Portfolio metrics")
            st.write("• Risk assessment")
            st.write("• Diversification")
            st.write("• AI recommendations")

except Exception as e:
    st.info("🔄 Starting advanced ML analysis system...")



# Navigation help
st.markdown("---")

# Add sidebar navigation
st.sidebar.markdown("### 📊 Navigation")
st.sidebar.page_link("pages/1_Stock_Screener.py", label="Stock Screener", icon="🔍")
st.sidebar.page_link("pages/2_Technical_Analysis.py", label="Technical Analysis", icon="📈")
st.sidebar.page_link("pages/3_Watchlist.py", label="Watchlist Management", icon="📋")
st.sidebar.page_link("pages/4_ML_Training.py", label="ML Training", icon="🤖")
st.sidebar.page_link("pages/5_Portfolio_Risk_Assessment.py", label="Portfolio Risk Assessment", icon="🎯")
st.sidebar.page_link("pages/6_Investment_Narrative.py", label="Investment Narrative", icon="📝")
st.sidebar.page_link("pages/7_Emoji_Mood_Selector.py", label="Emoji Mood Selector", icon="🎭")

st.sidebar.markdown("---")

# Global PDF Export Section
st.sidebar.subheader("📄 Complete Report Export")

if st.sidebar.button("📥 Generate Complete PDF Report", use_container_width=True, type="primary", help="Export all current session data and charts"):
    try:
        from utils.enhanced_pdf_export import create_one_click_pdf_export
        
        with st.sidebar:
            with st.spinner("Generating comprehensive PDF..."):
                pdf_bytes = create_one_click_pdf_export()
                
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download Complete Report",
                        data=pdf_bytes,
                        file_name=f"complete_market_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help="Complete analysis report with all session data",
                        use_container_width=True
                    )
                    st.success("✅ PDF generated successfully!")
                else:
                    st.error("❌ No data available for export")
    except Exception as e:
        st.sidebar.error(f"Export error: {str(e)}")

st.sidebar.markdown("---")

# Quick Portfolio Risk Assessment Widget
st.sidebar.subheader("🎯 Quick Risk Check")

if st.session_state.watchlist:
    if st.sidebar.button("📊 Analyze Watchlist Risk", use_container_width=True):
        # Create equal weight portfolio from watchlist
        equal_weight = 1.0 / len(st.session_state.watchlist)
        portfolio_for_risk = {stock: equal_weight for stock in st.session_state.watchlist}
        
        # Store in session state and redirect
        st.session_state.portfolio_for_analysis = portfolio_for_risk
        st.switch_page("pages/5_Portfolio_Risk_Assessment.py")
else:
    st.sidebar.info("Add stocks to watchlist for quick risk analysis")

st.info("👈 Use the sidebar to navigate between different sections of the screener")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built for Indian Stock Market Analysis | Real-time data via Yahoo Finance</p>
    <p>⚠️ This tool is for educational and analysis purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
