import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import io

# Import utility functions
from utils.data_fetcher import get_multiple_stocks_data, get_stock_info, format_market_cap
from utils.technical_analysis import calculate_all_indicators, get_purchase_signal, get_trend_status
from utils.fundamental_analysis import get_fundamental_data, format_fundamental_value, get_fundamental_score
from utils.stock_lists import (
    get_stocks_by_index, get_stocks_by_sector, get_all_sectors, 
    get_all_indices, get_stock_sector, get_stock_market_cap_category,
    search_stock, get_all_stocks, get_stock_display_name
)
from utils.ml_service import ml_service

st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")

# Initialize session state
if 'display_format_radio' not in st.session_state:
    st.session_state.display_format_radio = "Summary View"

if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None

if 'screening_df' not in st.session_state:
    st.session_state.screening_df = None

st.title("🔍 Enhanced AI Stock Screener")
st.markdown("Filter and analyze Indian stocks with real-time ML signals and sentiment analysis")

# Quick ML Overview
col1, col2, col3 = st.columns(3)
try:
    market_overview = ml_service.get_market_overview()
    with col1:
        st.metric("Stocks with ML Signals", market_overview['total_stocks'])
    with col2:
        st.metric("Buy Signals", market_overview['buy_signals'])
    with col3:
        st.metric("Sell Signals", market_overview['sell_signals'])
except:
    st.info("🔄 Background ML processing starting...")

# Screening Mode Selection
screening_mode = st.selectbox(
    "🎯 Select Screening Mode",
    ["Quick ML Screener (All NSE Stocks)", "Traditional Screener (Custom Filters)"],
    help="Choose between AI-powered screening of all processed stocks or custom filtering"
)

if screening_mode == "Quick ML Screener (All NSE Stocks)":
    st.markdown("---")
    st.subheader("🤖 AI-Powered Stock Screening")
    st.markdown("Real-time ML signals for all NSE stocks processed in background")
    
    # ML Screening filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ml_signal_filter = st.selectbox(
            "ML Signal Filter",
            ["All Signals", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
            index=0
        )
    
    with col2:
        min_confidence = st.slider(
            "Min ML Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Filter by ML signal confidence"
        )
    
    with col3:
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            ["All", "Bullish", "Neutral", "Bearish"],
            index=0
        )
    
    with col4:
        sort_by = st.selectbox(
            "Sort By",
            ["ML Confidence", "Price Confidence", "ML Score", "Change %", "Volume"],
            index=0
        )
    
    # Additional price-based filter
    col5, col6 = st.columns(2)
    with col5:
        price_filter = st.selectbox(
            "Price Recommendation Filter",
            ["All", "Strong Buy - Below ideal", "Buy - Near ideal", "Hold - Slightly above ideal", "Wait - Above ideal"],
            index=0,
            help="Filter by AI price recommendations"
        )
    
    with col6:
        show_price_targets = st.checkbox(
            "Show Price Targets",
            value=True,
            help="Display 1-week and 1-month price targets"
        )
    
    # Get and display ML screening results
    if st.button("🔍 Screen All Stocks with ML", type="primary"):
        with st.spinner("Fetching ML signals for all processed stocks..."):
            try:
                # Use ML service to get filtered results
                signal_filter = ml_signal_filter if ml_signal_filter != "All Signals" else None
                if signal_filter:
                    ml_results_df = ml_service.search_stocks_by_signal(
                        signal_type=signal_filter,
                        min_confidence=min_confidence
                    )
                else:
                    ml_results_df = ml_service.get_all_ml_data()
                
                # Safety check: ensure we have valid data
                if ml_results_df is None or (isinstance(ml_results_df, pd.DataFrame) and ml_results_df.empty):
                    st.warning("No ML data available yet. The background system is still processing stocks.")
                    st.stop()
                
                # Ensure we have a DataFrame to work with
                if isinstance(ml_results_df, pd.DataFrame) and not ml_results_df.empty:
                    # Clean up all numeric columns to handle None values
                    numeric_columns = ['Sentiment Score', 'ML Confidence', 'ML Score', 'Price Confidence', 
                                     'Current Price', 'Ideal Price', 'Price Target 1W', 'Price Target 1M', 
                                     'Change %', 'Volume', 'Support Level', 'Resistance Level']
                    
                    for col in numeric_columns:
                        if col in ml_results_df.columns:
                            ml_results_df[col] = pd.to_numeric(ml_results_df[col], errors='coerce').fillna(
                                50 if col == 'Sentiment Score' else 0
                            )
                    
                    # Apply sentiment filter (with comprehensive error handling)
                    try:
                        if sentiment_filter != "All":
                            if sentiment_filter == "Bullish":
                                ml_results_df = ml_results_df[ml_results_df['Sentiment Score'] > 60]
                            elif sentiment_filter == "Bearish":
                                ml_results_df = ml_results_df[ml_results_df['Sentiment Score'] < 40]
                            else:  # Neutral
                                ml_results_df = ml_results_df[
                                    (ml_results_df['Sentiment Score'] >= 40) & 
                                    (ml_results_df['Sentiment Score'] <= 60)
                                ]
                    except Exception as sentiment_error:
                        st.warning(f"Sentiment filtering issue, showing all results: {str(sentiment_error)}")
                    
                    # Apply price recommendation filter (with error handling)
                    try:
                        if price_filter != "All":
                            if price_filter == "Strong Buy - Below ideal":
                                ml_results_df = ml_results_df[ml_results_df['Price Recommendation'].str.contains('Strong Buy', na=False)]
                            elif price_filter == "Buy - Near ideal":
                                ml_results_df = ml_results_df[ml_results_df['Price Recommendation'].str.contains('Buy', na=False) & 
                                                            ~ml_results_df['Price Recommendation'].str.contains('Strong Buy', na=False)]
                            elif price_filter == "Hold - Slightly above ideal":
                                ml_results_df = ml_results_df[ml_results_df['Price Recommendation'].str.contains('Hold', na=False)]
                            elif price_filter == "Wait - Above ideal":
                                ml_results_df = ml_results_df[ml_results_df['Price Recommendation'].str.contains('Wait', na=False)]
                    except Exception as price_error:
                        st.warning(f"Price filtering issue, showing all results: {str(price_error)}")
                    
                    # Sort results (with error handling)
                    try:
                        if sort_by == "ML Confidence":
                            ml_results_df = ml_results_df.sort_values('ML Confidence', ascending=False)
                        elif sort_by == "Price Confidence":
                            ml_results_df = ml_results_df.sort_values('Price Confidence', ascending=False)
                        elif sort_by == "ML Score":
                            ml_results_df = ml_results_df.sort_values('ML Score', ascending=False)
                        elif sort_by == "Change %":
                            ml_results_df = ml_results_df.sort_values('Change %', ascending=False)
                        elif sort_by == "Volume":
                            ml_results_df = ml_results_df.sort_values('Volume', ascending=False)
                    except Exception as sort_error:
                        st.warning(f"Sorting issue, displaying unsorted results: {str(sort_error)}")
                    
                    st.success(f"Found {len(ml_results_df)} stocks matching your criteria")
                    
                    # Display results with color coding
                    def color_ml_signals(val):
                        if val in ['Strong Buy']:
                            return 'background-color: #90EE90'
                        elif val in ['Buy']:
                            return 'background-color: #ADD8E6'
                        elif val in ['Sell']:
                            return 'background-color: #FFB6C1'
                        elif val in ['Strong Sell']:
                            return 'background-color: #FFA07A'
                        else:
                            return ''
                    
                    # Format display with price predictions and add emoji moods
                    display_df = ml_results_df.copy()
                    
                    # Add emoji mood column
                    try:
                        from utils.emoji_mood_selector import get_stock_emoji
                        display_df['Emoji'] = display_df['Symbol'].apply(get_stock_emoji)
                        # Combine emoji with symbol
                        display_df['Symbol'] = display_df['Emoji'] + ' ' + display_df['Symbol']
                        display_df = display_df.drop('Emoji', axis=1)
                    except:
                        pass  # Continue without emoji if there's an error
                    
                    # Format numeric columns for display
                    if 'ML Confidence' in display_df.columns:
                        display_df['ML Confidence'] = display_df['ML Confidence'].apply(lambda x: f"{x:.1%}")
                    if 'Price Confidence' in display_df.columns:
                        display_df['Price Confidence'] = display_df['Price Confidence'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
                    if 'Change %' in display_df.columns:
                        display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")
                    if 'Current Price' in display_df.columns:
                        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
                    if 'Ideal Price' in display_df.columns:
                        display_df['Ideal Price'] = display_df['Ideal Price'].apply(lambda x: f"₹{x:.2f}" if x > 0 else "N/A")
                    if 'Price Target 1W' in display_df.columns:
                        display_df['Price Target 1W'] = display_df['Price Target 1W'].apply(lambda x: f"₹{x:.2f}" if x > 0 else "N/A")
                    if 'Price Target 1M' in display_df.columns:
                        display_df['Price Target 1M'] = display_df['Price Target 1M'].apply(lambda x: f"₹{x:.2f}" if x > 0 else "N/A")
                    if 'Volume' in display_df.columns:
                        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
                    
                    # Reorder columns based on user preference for price targets
                    if show_price_targets:
                        key_columns = ['Symbol', 'Current Price', 'Ideal Price', 'Price Target 1W', 'Price Target 1M', 
                                     'Price Recommendation', 'Price Confidence', 'ML Signal', 'ML Confidence', 
                                     'Change %', 'Volume', 'Sentiment Score', 'Sentiment Label']
                    else:
                        key_columns = ['Symbol', 'Current Price', 'Ideal Price', 'Price Recommendation', 'Price Confidence', 
                                     'ML Signal', 'ML Confidence', 'Change %', 'Volume', 'Sentiment Score', 'Sentiment Label']
                    
                    # Only include columns that exist in the dataframe
                    available_columns = [col for col in key_columns if col in display_df.columns]
                    display_df = display_df[available_columns]
                    
                    # Style the dataframe
                    styled_df = display_df.style.applymap(color_ml_signals, subset=['ML Signal'])
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Add watchlist functionality
                    st.markdown("### 📋 Quick Actions")
                    selected_stocks = st.multiselect(
                        "Add to Watchlist",
                        options=display_df['Symbol'].tolist(),
                        help="Select stocks to add to your watchlist"
                    )
                    
                    if selected_stocks and st.button("➕ Add to Watchlist"):
                        for stock in selected_stocks:
                            if stock not in st.session_state.watchlist:
                                st.session_state.watchlist.append(stock)
                        st.success(f"Added {len(selected_stocks)} stocks to watchlist")
                        st.rerun()
                    
                    # Download functionality
                    csv_buffer = io.StringIO()
                    display_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"ml_screening_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No stocks match your criteria. Try adjusting the filters.")
                    
            except Exception as e:
                # Handle any remaining data comparison or processing errors
                error_message = str(e)
                if "not supported between instances" in error_message or "NoneType" in error_message:
                    st.error("Error in ML screening: Data comparison issue with None values")
                    st.info("The background ML system is still processing data. Some stocks may have incomplete information.")
                    st.info("Try using different filters or wait a moment for more data to be processed.")
                else:
                    st.error(f"Error in ML screening: {error_message}")
                    st.info("The background ML system may still be processing stocks. Please try again in a few minutes.")

else:
    # Traditional screening mode - keep existing sidebar filters
    st.sidebar.header("📊 Screening Filters")

# Stock Search Box
st.sidebar.subheader("🔍 Stock Search")
search_query = st.sidebar.text_input(
    "Search by Symbol or Company Name",
    placeholder="e.g., RELIANCE, TCS, Infosys",
    help="Type to search from 500+ Indian stocks"
)

# Show search results
if search_query:
    matching_stocks = search_stock(search_query)
    if matching_stocks:
        st.sidebar.write(f"Found {len(matching_stocks)} matches:")
        for stock in matching_stocks[:10]:  # Show top 10
            display_name = get_stock_display_name(stock)
            st.sidebar.write(f"• **{stock}** - {display_name}")
    else:
        st.sidebar.warning("No matching stocks found")

st.sidebar.markdown("---")

# Index/Market Cap selection
index_selection = st.sidebar.selectbox(
    "Select Market Segment",
    options=get_all_indices(),
    index=0
)

# Sector selection
sector_selection = st.sidebar.multiselect(
    "Select Sectors",
    options=get_all_sectors(),
    default=[]
)

# Technical filters
st.sidebar.subheader("🔧 Technical Filters")

rsi_range = st.sidebar.slider(
    "RSI Range",
    min_value=0,
    max_value=100,
    value=(0, 100),
    help="Filter stocks by RSI values"
)

volume_filter = st.sidebar.checkbox(
    "High Volume (Above Average)",
    help="Show only stocks with above-average volume"
)

breakout_filter = st.sidebar.checkbox(
    "Bullish Breakout Only",
    help="Show only stocks with recent bullish breakouts"
)

# Fundamental filters
st.sidebar.subheader("💰 Fundamental Filters")

pe_range = st.sidebar.slider(
    "PE Ratio Range",
    min_value=0,
    max_value=100,
    value=(0, 50),
    help="Filter stocks by PE ratio"
)

market_cap_filter = st.sidebar.selectbox(
    "Market Cap Category",
    options=["All", "Large Cap", "Mid Cap", "Small Cap"],
    index=0
)

# Analysis period
analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=3
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data", help="Reload stock data"):
    st.cache_data.clear()
    st.rerun()

# Main content
if st.button("🚀 Start Screening", type="primary"):
    # Get stocks to analyze
    stocks_to_analyze = get_stocks_by_index(index_selection)
    
    # Filter by sectors if selected
    if sector_selection:
        sector_stocks = []
        for sector in sector_selection:
            sector_stocks.extend(get_stocks_by_sector(sector))
        # Remove duplicates from sector stocks first
        sector_stocks = list(set(sector_stocks))
        # Intersection of index and sector stocks
        stocks_to_analyze = list(set(stocks_to_analyze) & set(sector_stocks))
    
    # Remove any remaining duplicates from final list
    stocks_to_analyze = list(set(stocks_to_analyze))
    
    if not stocks_to_analyze:
        st.error("No stocks found matching the selected criteria")
        st.stop()
    
    st.info(f"Analyzing {len(stocks_to_analyze)} stocks...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results storage
    screened_stocks = []
    
    # Analyze stocks in batches for better performance
    batch_size = 10
    for i in range(0, len(stocks_to_analyze), batch_size):
        batch = stocks_to_analyze[i:i+batch_size]
        
        status_text.text(f"Processing batch {i//batch_size + 1}/{(len(stocks_to_analyze)-1)//batch_size + 1}")
        
        # Get data for batch
        stocks_data = get_multiple_stocks_data(batch, period=analysis_period)
        
        for j, symbol in enumerate(batch):
            try:
                data = stocks_data.get(symbol, pd.DataFrame())
                
                if data.empty or len(data) < 50:
                    continue
                
                # Calculate technical indicators
                indicators = calculate_all_indicators(data)
                if not indicators:
                    continue
                
                # Get fundamental data
                fundamentals = get_fundamental_data(symbol)
                
                # Apply filters
                # RSI filter
                rsi = indicators.get('RSI', 50)
                if not (rsi_range[0] <= rsi <= rsi_range[1]):
                    continue
                
                # Volume filter
                if volume_filter:
                    volume_ratio = indicators.get('Volume_Ratio', 1)
                    if volume_ratio <= 1.2:  # At least 20% above average
                        continue
                
                # Breakout filter
                if breakout_filter:
                    bullish_breakout = indicators.get('Bullish_Breakout', False)
                    if not bullish_breakout:
                        continue
                
                # PE filter
                pe_ratio = fundamentals.get('PE Ratio', 0)
                if pe_ratio > 0 and not (pe_range[0] <= pe_ratio <= pe_range[1]):
                    continue
                
                # Market cap filter
                if market_cap_filter != "All":
                    stock_category = get_stock_market_cap_category(symbol)
                    if stock_category != market_cap_filter:
                        continue
                
                # Get additional info
                purchase_signal, signal_details = get_purchase_signal(data, symbol)
                trend_status = get_trend_status(indicators)
                sector = get_stock_sector(symbol)
                fundamental_score = get_fundamental_score(fundamentals)
                
                # Get ML signal if available
                ml_signal = "Not Available"
                ml_confidence = 0
                try:
                    from utils.ml_signals import get_ml_signal, is_model_trained
                    if is_model_trained():
                        ml_result = get_ml_signal(data, symbol)
                        ml_signal = ml_result.get('signal', 'Hold')
                        ml_confidence = ml_result.get('confidence', 0)
                except:
                    pass
                
                # Get news sentiment
                news_sentiment = "Neutral"
                sentiment_score = 50
                try:
                    from utils.news_sentiment import get_news_sentiment_score
                    sentiment_data = get_news_sentiment_score(symbol)
                    news_sentiment = sentiment_data.get('sentiment_label', 'neutral').title()
                    sentiment_score = sentiment_data.get('sentiment_score', 50)
                except:
                    pass
                
                # Add to results
                stock_result = {
                    'Symbol': symbol,
                    'Sector': sector,
                    'Current Price': indicators.get('Current_Price', 0),
                    'Price Change %': indicators.get('Price_Change_Pct', 0),
                    'RSI': indicators.get('RSI', 0),
                    'MACD': indicators.get('MACD', 0),
                    'MA44': indicators.get('MA44', 0),
                    'Volume Ratio': indicators.get('Volume_Ratio', 0),
                    'PE Ratio': fundamentals.get('PE Ratio', 0),
                    'EPS': fundamentals.get('EPS', 0),
                    'Market Cap': fundamentals.get('Market Cap', 0),
                    'ROE': fundamentals.get('ROE', 0),
                    'ROCE': fundamentals.get('ROCE', 0),
                    'Debt to Equity': fundamentals.get('Debt to Equity', 0),
                    'Purchase Signal': purchase_signal,
                    'Signal Details': signal_details,
                    'Trend': trend_status,
                    'Fundamental Score': fundamental_score,
                    'Support': indicators.get('Support', 0),
                    'Resistance': indicators.get('Resistance', 0),
                    'Bullish Breakout': indicators.get('Bullish_Breakout', False),
                    'ML Signal': ml_signal,
                    'ML Confidence': ml_confidence,
                    'News Sentiment': news_sentiment,
                    'Sentiment Score': sentiment_score
                }
                
                screened_stocks.append(stock_result)
                
            except Exception as e:
                st.warning(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Update progress
        progress = min((i + batch_size) / len(stocks_to_analyze), 1.0)
        progress_bar.progress(progress)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if not screened_stocks:
        st.warning("No stocks found matching all criteria. Try adjusting your filters.")
        st.stop()
    
    # Convert to DataFrame
    df = pd.DataFrame(screened_stocks)
    
    # Remove duplicates based on Symbol (this handles any remaining duplicates)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    
    # Sort by fundamental score and purchase signals
    df['Signal_Score'] = df['Purchase Signal'].replace({
        'Strong Buy': 4, 'Buy': 3, 'Watch': 2, 'Hold': 1, 'Error': 0
    }).fillna(0)
    
    df = df.sort_values(['Signal_Score', 'Fundamental Score'], ascending=[False, False])
    
    # Store results in session state
    st.session_state.screening_results = screened_stocks
    st.session_state.screening_df = df
    
    st.success(f"Found {len(df)} stocks matching your criteria!")

# Display results section (outside button handler to prevent refresh issues)
if st.session_state.screening_df is not None:
    df = st.session_state.screening_df
    
    # Display results
    st.subheader("📈 Screening Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strong_buy_count = len(df[df['Purchase Signal'] == 'Strong Buy'])
        st.metric("Strong Buy Signals", strong_buy_count)
    
    with col2:
        buy_count = len(df[df['Purchase Signal'] == 'Buy'])
        st.metric("Buy Signals", buy_count)
    
    with col3:
        high_score_count = len(df[df['Fundamental Score'] >= 70])
        st.metric("High Fundamental Score", high_score_count)
    
    with col4:
        breakout_count = len(df[df['Bullish Breakout'] == True])
        st.metric("Bullish Breakouts", breakout_count)

    # Filter and display options
    st.subheader("🎯 Top Recommendations")
    
    # Display format selection
    display_format = st.radio(
        "Select Display Format",
        options=["Summary View", "Detailed View", "Technical Focus", "Fundamental Focus", "AI Analysis"],
        horizontal=True,
        key="display_format_radio"
    )
    
    if display_format == "Summary View":
        display_columns = [
            'Symbol', 'Sector', 'Current Price', 'Price Change %', 'RSI', 
            'Purchase Signal', 'Trend', 'Fundamental Score'
        ]
    elif display_format == "Technical Focus":
        display_columns = [
            'Symbol', 'Current Price', 'Price Change %', 'RSI', 'MACD', 'MA44',
            'Volume Ratio', 'Support', 'Resistance', 'Purchase Signal', 'Bullish Breakout'
        ]
    elif display_format == "Fundamental Focus":
        display_columns = [
            'Symbol', 'Current Price', 'PE Ratio', 'EPS', 'ROE', 'ROCE',
            'Debt to Equity', 'Market Cap', 'Fundamental Score'
        ]
    elif display_format == "AI Analysis":
        display_columns = [
            'Symbol', 'Current Price', 'Purchase Signal', 'ML Signal', 'ML Confidence',
            'News Sentiment', 'Sentiment Score', 'Trend', 'Fundamental Score'
        ]
    else:  # Detailed View
        display_columns = list(df.columns)
        display_columns.remove('Signal_Score')  # Hide internal scoring column
    
    # Format the display dataframe
    display_df = df[display_columns].copy()
    
    # Format numerical columns safely with proper pandas Series operations
    try:
        # Ensure we're working with a proper DataFrame
        display_df = pd.DataFrame(display_df)
        
        if 'Current Price' in display_df.columns:
            display_df['Current Price'] = [f"₹{float(x):.2f}" for x in display_df['Current Price']]
        
        if 'Price Change %' in display_df.columns:
            display_df['Price Change %'] = [f"{float(x):+.2f}%" for x in display_df['Price Change %']]
        
        if 'RSI' in display_df.columns:
            display_df['RSI'] = [f"{float(x):.1f}" if str(x) != 'N/A' and pd.notna(x) else "N/A" for x in display_df['RSI']]
        
        if 'Volume Ratio' in display_df.columns:
            display_df['Volume Ratio'] = [f"{float(x):.2f}x" for x in display_df['Volume Ratio']]
        
        if 'Market Cap' in display_df.columns:
            display_df['Market Cap'] = [format_market_cap(x) for x in display_df['Market Cap']]
        
        if 'ROE' in display_df.columns:
            display_df['ROE'] = [f"{float(x)*100:.1f}%" if float(x) != 0 else "N/A" for x in display_df['ROE']]
        
        if 'ROCE' in display_df.columns:
            display_df['ROCE'] = [f"{float(x):.1f}%" if float(x) != 0 else "N/A" for x in display_df['ROCE']]
        
        if 'Support' in display_df.columns:
            display_df['Support'] = [f"₹{float(x):.2f}" if float(x) != 0 else "N/A" for x in display_df['Support']]
        
        if 'Resistance' in display_df.columns:
            display_df['Resistance'] = [f"₹{float(x):.2f}" if float(x) != 0 else "N/A" for x in display_df['Resistance']]
            
    except Exception as e:
        st.warning(f"Warning: Error formatting display columns: {str(e)}")
        # Fallback to basic display
        pass
    
    # Color-code certain columns
    def highlight_signals(val):
        if val == 'Strong Buy':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Buy':
            return 'background-color: #fff3cd; color: #856404'
        elif val == 'Watch':
            return 'background-color: #e2e3e5; color: #383d41'
        return ''
    
    def highlight_trend(val):
        if val == 'Bullish':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Bearish':
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    # Apply styling
    styled_df = display_df.style
    
    if 'Purchase Signal' in display_df.columns:
        styled_df = styled_df.applymap(highlight_signals, subset=['Purchase Signal'])
    
    if 'Trend' in display_df.columns:
        styled_df = styled_df.applymap(highlight_trend, subset=['Trend'])
    
    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600
    )
    
    # Watchlist management
    st.subheader("⭐ Watchlist Management")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_for_watchlist = st.multiselect(
            "Select stocks to add to watchlist",
            options=df['Symbol'].tolist(),
            help="Choose stocks for your algo trading watchlist"
        )
    
    with col2:
        if st.button("➕ Add to Watchlist"):
            if selected_for_watchlist:
                for symbol in selected_for_watchlist:
                    if symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(symbol)
                st.success(f"Added {len(selected_for_watchlist)} stocks to watchlist")
                st.rerun()
            else:
                st.warning("Please select stocks to add")
    
    # Comprehensive Export functionality
    st.subheader("📤 Comprehensive Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Comprehensive PDF Export
        if st.button("📄 Export Complete PDF Report"):
            try:
                from utils.comprehensive_export import ComprehensiveExporter
                
                exporter = ComprehensiveExporter()
                
                # Prepare data for PDF
                pdf_data = {}
                
                # Main screening results
                pdf_data['Screening Results'] = df.copy()
                
                # Summary statistics
                summary_stats = {
                    'Total Stocks Analyzed': len(df),
                    'Strong Buy Signals': len(df[df['Purchase Signal'] == 'Strong Buy']),
                    'Buy Signals': len(df[df['Purchase Signal'] == 'Buy']),
                    'Bullish Trends': len(df[df['Trend'] == 'Bullish']),
                    'Average RSI': df['RSI'].mean() if 'RSI' in df.columns else 0,
                    'Average PE Ratio': df['PE Ratio'].mean() if 'PE Ratio' in df.columns else 0,
                    'Screening Criteria': f"Index: {index_selection}, Sectors: {', '.join(sector_selection) if sector_selection else 'All'}"
                }
                pdf_data['Summary Statistics'] = summary_stats
                
                # Generate PDF
                pdf_bytes = exporter.create_pdf_report(pdf_data)
                
                st.download_button(
                    label="📥 Download Complete PDF Report",
                    data=pdf_bytes,
                    file_name=f"stock_screening_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    help="Comprehensive PDF report with all screening data and analysis"
                )
                
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                # Fallback to CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download CSV (Fallback)",
                    data=csv_data,
                    file_name=f"stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Comprehensive Excel Export
        if st.button("📊 Export Complete Excel Report"):
            try:
                from utils.comprehensive_export import ComprehensiveExporter
                
                exporter = ComprehensiveExporter()
                
                # Prepare data for Excel with multiple tabs
                excel_data = {}
                
                # Main results
                excel_data['Screening Results'] = df.copy()
                
                # Technical analysis tab
                technical_columns = ['Symbol', 'Sector', 'Current Price', 'Price Change %', 'RSI', 'MACD Signal', 
                                   'Volume Ratio', 'Purchase Signal', 'Trend', 'Support', 'Resistance']
                technical_df = df[[col for col in technical_columns if col in df.columns]].copy()
                excel_data['Technical Analysis'] = technical_df
                
                # Fundamental analysis tab
                fundamental_columns = ['Symbol', 'Sector', 'Category', 'Current Price', 'PE Ratio', 'Market Cap', 
                                     'ROE', 'ROCE', 'Fundamental Score', 'Purchase Signal']
                fundamental_df = df[[col for col in fundamental_columns if col in df.columns]].copy()
                excel_data['Fundamental Analysis'] = fundamental_df
                
                # Buy signals tab
                buy_signals_df = df[df['Purchase Signal'].isin(['Strong Buy', 'Buy'])].copy()
                if not buy_signals_df.empty:
                    excel_data['Buy Signals'] = buy_signals_df
                
                # Summary statistics
                summary_data = []
                summary_data.append({'Metric': 'Total Stocks Analyzed', 'Value': len(df)})
                summary_data.append({'Metric': 'Strong Buy Signals', 'Value': len(df[df['Purchase Signal'] == 'Strong Buy'])})
                summary_data.append({'Metric': 'Buy Signals', 'Value': len(df[df['Purchase Signal'] == 'Buy'])})
                summary_data.append({'Metric': 'Bullish Trends', 'Value': len(df[df['Trend'] == 'Bullish'])})
                if 'RSI' in df.columns:
                    summary_data.append({'Metric': 'Average RSI', 'Value': df['RSI'].mean()})
                if 'PE Ratio' in df.columns:
                    summary_data.append({'Metric': 'Average PE Ratio', 'Value': df['PE Ratio'].mean()})
                
                excel_data['Summary'] = pd.DataFrame(summary_data)
                
                # Generate Excel
                excel_bytes = exporter.create_excel_report(excel_data)
                
                st.download_button(
                    label="📥 Download Complete Excel Report",
                    data=excel_bytes,
                    file_name=f"stock_screening_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Comprehensive Excel report with multiple tabs: Results, Technical, Fundamental, Buy Signals, Summary"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {e}")
                # Fallback to basic Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Screened_Stocks', index=False)
                excel_buffer.seek(0)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Excel (Fallback)",
                    data=excel_data,
                    file_name=f"stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col3:
        # Trading signals export
        if st.button("🎯 Export Trading Signals"):
            try:
                # Create trading-focused export
                buy_signals = df[df['Purchase Signal'].isin(['Strong Buy', 'Buy'])].copy()
                
                if not buy_signals.empty:
                    # Create trading list
                    trading_data = []
                    for _, stock in buy_signals.iterrows():
                        trading_data.append({
                            'Symbol': stock['Symbol'],
                            'Signal': stock['Purchase Signal'],
                            'Current Price': stock['Current Price'],
                            'Target Price': stock.get('Resistance', 'N/A'),
                            'Support Level': stock.get('Support', 'N/A'),
                            'RSI': stock.get('RSI', 'N/A'),
                            'Trend': stock.get('Trend', 'N/A'),
                            'Sector': stock.get('Sector', 'N/A')
                        })
                    
                    trading_df = pd.DataFrame(trading_data)
                    
                    # Export as Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        trading_df.to_excel(writer, sheet_name='Trading_Signals', index=False)
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Trading Signals",
                        data=excel_buffer.getvalue(),
                        file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Export only buy signals with trading-specific information"
                    )
                else:
                    st.info("No buy signals found to export")
                    
            except Exception as e:
                st.error(f"Error generating trading signals: {e}")

else:
    # Show information about the screener
    st.info("👆 Click 'Start Screening' to analyze stocks based on your selected criteria")
    
    # Display sample filters explanation
    with st.expander("ℹ️ How to use the Stock Screener"):
        st.markdown("""
        **Step 1:** Select your preferred market segment (Nifty 50, Nifty 100, etc.)
        
        **Step 2:** Choose specific sectors to focus on (optional)
        
        **Step 3:** Set technical filters:
        - **RSI Range**: Filter stocks by RSI values (30-70 for moderate signals)
        - **High Volume**: Show stocks with above-average trading volume
        - **Bullish Breakout**: Show stocks breaking out of resistance levels
        
        **Step 4:** Set fundamental filters:
        - **PE Ratio Range**: Filter by valuation (10-25 is often considered reasonable)
        - **Market Cap Category**: Focus on Large, Mid, or Small cap stocks
        
        **Step 5:** Click 'Start Screening' to analyze the stocks
        
        **Features:**
        - Purchase range suggestions based on support/resistance
        - Watchlist management for algo trading
        - Export results to CSV/Excel
        - Multiple display formats for different analysis needs
        """)

# Display current watchlist in sidebar
if st.session_state.watchlist:
    st.sidebar.subheader("⭐ Current Watchlist")
    for i, symbol in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.text(symbol)
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()
