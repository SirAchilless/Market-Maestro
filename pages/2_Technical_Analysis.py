import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import utility functions
from utils.data_fetcher import get_stock_data, get_stock_info
from utils.technical_analysis import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_moving_averages, calculate_volume_sma, detect_breakout,
    calculate_support_resistance, get_purchase_signal, calculate_all_indicators
)
from utils.chart_generator import (
    create_candlestick_chart, create_rsi_chart, create_volume_analysis_chart
)
from utils.stock_lists import search_stock, get_stock_display_name

st.set_page_config(page_title="Technical Analysis", page_icon="📊", layout="wide")

st.title("📊 Technical Analysis")
st.markdown("Deep dive into individual stock technical indicators and charts")

# Sidebar for stock selection and parameters
st.sidebar.header("🎯 Stock Selection")

# Stock search and selection with comprehensive database
search_query = st.sidebar.text_input(
    "Search Stock by Symbol or Company Name", 
    placeholder="e.g., RELIANCE, TCS, Infosys",
    help="Search from 500+ Indian stocks"
)

if search_query:
    matching_stocks = search_stock(search_query)
    if matching_stocks:
        # Create display options with company names
        stock_options = []
        for stock in matching_stocks:
            display_name = get_stock_display_name(stock)
            stock_options.append(f"{stock} - {display_name}")
            
        selected_option = st.sidebar.selectbox("Select Stock", options=stock_options)
        selected_stock = selected_option.split(" - ")[0] if selected_option else None
    else:
        st.sidebar.warning("No matching stocks found")
        selected_stock = search_query.upper()
else:
    # Default popular stocks with names
    popular_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'ITC', 'LT']
    stock_options = [f"{stock} - {get_stock_display_name(stock)}" for stock in popular_stocks]
    selected_option = st.sidebar.selectbox("Select Popular Stock", options=stock_options)
    selected_stock = selected_option.split(" - ")[0] if selected_option else None

# Time period selection
time_period = st.sidebar.selectbox(
    "Analysis Period",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

# Chart timeframe
chart_interval = st.sidebar.selectbox(
    "Chart Interval",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="Daily, Weekly, or Monthly candlesticks"
)

# Technical indicator parameters
st.sidebar.subheader("🔧 Indicator Parameters")

rsi_period = st.sidebar.number_input("RSI Period", value=14, min_value=5, max_value=50)
macd_fast = st.sidebar.number_input("MACD Fast", value=12, min_value=5, max_value=30)
macd_slow = st.sidebar.number_input("MACD Slow", value=26, min_value=15, max_value=50)
macd_signal = st.sidebar.number_input("MACD Signal", value=9, min_value=5, max_value=20)
bb_period = st.sidebar.number_input("Bollinger Bands Period", value=20, min_value=10, max_value=50)

# Main content
if selected_stock:
    # Get stock data
    with st.spinner(f"Loading data for {selected_stock}..."):
        stock_data = get_stock_data(selected_stock, period=time_period, interval=chart_interval)
        stock_info = get_stock_info(selected_stock)
    
    if stock_data.empty:
        st.error(f"Could not fetch data for {selected_stock}. Please check the symbol.")
        st.stop()
    
    # Display stock info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        st.metric(
            "Current Price",
            f"₹{current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        high_52w = stock_info.get('fifty_two_week_high', 0)
        low_52w = stock_info.get('fifty_two_week_low', 0)
        st.metric("52W High", f"₹{high_52w:.2f}" if high_52w else "N/A")
        st.metric("52W Low", f"₹{low_52w:.2f}" if low_52w else "N/A")
    
    with col3:
        market_cap = stock_info.get('market_cap', 0)
        pe_ratio = stock_info.get('pe_ratio', 0)
        from utils.data_fetcher import format_market_cap
        st.metric("Market Cap", format_market_cap(market_cap))
        st.metric("PE Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    
    with col4:
        sector = stock_info.get('sector', 'N/A')
        beta = stock_info.get('beta', 0)
        st.metric("Sector", sector)
        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    
    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        # RSI
        rsi_values = calculate_rsi(stock_data, period=rsi_period)
        
        # MACD
        macd_line, macd_signal_line, macd_histogram = calculate_macd(
            stock_data, fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(stock_data, period=bb_period)
        
        # Moving Averages
        ma_values = calculate_moving_averages(stock_data, periods=[20, 44, 50, 200])
        
        # Volume analysis
        volume_sma = calculate_volume_sma(stock_data)
        
        # Support and Resistance
        support, resistance = calculate_support_resistance(stock_data)
        
        # Breakout detection
        bullish_breakout, bearish_breakout = detect_breakout(stock_data)
        
        # Purchase signal with ML and sentiment
        purchase_signal, signal_details = get_purchase_signal(stock_data, selected_stock)
    
    # Technical indicators summary
    st.subheader("📈 Technical Indicators Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]) else 0
        rsi_signal = "Oversold" if current_rsi <= 30 else "Overbought" if current_rsi >= 70 else "Neutral"
        rsi_color = "green" if current_rsi <= 30 else "red" if current_rsi >= 70 else "blue"
        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
    
    with col2:
        current_macd = macd_line[-1] if len(macd_line) > 0 and not np.isnan(macd_line[-1]) else 0
        current_macd_signal = macd_signal_line[-1] if len(macd_signal_line) > 0 and not np.isnan(macd_signal_line[-1]) else 0
        macd_trend = "Bullish" if current_macd > current_macd_signal else "Bearish"
        st.metric("MACD", f"{current_macd:.3f}", macd_trend)
    
    with col3:
        current_bb_position = "Above Upper" if current_price > bb_upper[-1] else "Below Lower" if current_price < bb_lower[-1] else "Within Bands"
        st.metric("Bollinger Position", current_bb_position)
    
    with col4:
        ma20 = ma_values.get('MA20', np.array([]))
        current_ma20 = ma20[-1] if len(ma20) > 0 and not np.isnan(ma20[-1]) else 0
        ma_trend = "Above" if current_price > current_ma20 else "Below" if current_ma20 > 0 else "N/A"
        st.metric("MA20 Position", ma_trend)
    
    with col5:
        current_volume = stock_data['Volume'].iloc[-1]
        avg_volume = volume_sma[-1] if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        volume_status = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low"
        st.metric("Volume Status", f"{volume_ratio:.2f}x", volume_status)
    
    # AI-Enhanced Trading Analysis
    st.subheader("🤖 AI-Enhanced Trading Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # ML Signal Analysis
    with col1:
        st.markdown("#### 🤖 ML Prediction")
        try:
            from utils.ml_signals import get_ml_signal, is_model_trained
            
            if is_model_trained():
                ml_result = get_ml_signal(stock_data, selected_stock)
                ml_signal = ml_result['signal']
                ml_confidence = ml_result['confidence']
                ml_score = ml_result['ml_score']
                
                signal_color = {
                    'Strong Buy': 'green',
                    'Buy': 'orange', 
                    'Hold': 'blue',
                    'Sell': 'red'
                }.get(ml_signal, 'gray')
                
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; border-radius: 10px; 
                            background-color: {signal_color}20; border: 2px solid {signal_color}'>
                    <h4 style='color: {signal_color}; margin: 0;'>{ml_signal}</h4>
                    <p style='margin: 5px 0;'>Confidence: {ml_confidence:.1%}</p>
                    <p style='margin: 5px 0;'>ML Score: {ml_score}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**AI Recommendation:** {ml_result['recommendation']}")
            else:
                st.info("🎯 Train the ML model in the ML Training page to get AI predictions")
                
        except Exception as e:
            st.warning("ML analysis temporarily unavailable")
    
    # News Sentiment Analysis
    with col2:
        st.markdown("#### 📰 News Sentiment")
        try:
            from utils.news_sentiment import get_news_sentiment_score
            
            sentiment_data = get_news_sentiment_score(selected_stock)
            sentiment_score = sentiment_data['sentiment_score']
            sentiment_label = sentiment_data['sentiment_label']
            sentiment_confidence = sentiment_data['confidence']
            article_count = sentiment_data['article_count']
            
            sentiment_color = {
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'blue'
            }.get(sentiment_label, 'gray')
            
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; border-radius: 10px; 
                        background-color: {sentiment_color}20; border: 2px solid {sentiment_color}'>
                <h4 style='color: {sentiment_color}; margin: 0;'>{sentiment_label.title()}</h4>
                <p style='margin: 5px 0;'>Score: {sentiment_score}/100</p>
                <p style='margin: 5px 0;'>Confidence: {sentiment_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Articles Analyzed:** {article_count}")
            
            # Show recent news headlines
            if sentiment_data.get('recent_articles'):
                with st.expander("📄 Recent Headlines"):
                    for article in sentiment_data['recent_articles'][:3]:
                        st.write(f"• {article['title'][:60]}...")
                        
        except Exception as e:
            st.warning("News sentiment analysis temporarily unavailable")
    
    # Combined Signal
    with col3:
        st.markdown("#### 🎯 Combined Signal")
        
        signal_color = {
            'Strong Buy': 'green',
            'Buy': 'orange', 
            'Watch': 'blue',
            'Hold': 'gray',
            'Weak Sell': 'orange',
            'Sell': 'red',
            'Error': 'red'
        }.get(purchase_signal, 'gray')
        
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; border-radius: 10px; 
                    background-color: {signal_color}20; border: 2px solid {signal_color}'>
            <h4 style='color: {signal_color}; margin: 0;'>{purchase_signal}</h4>
            <p style='margin: 5px 0; font-size: 12px;'>{signal_details[:50]}...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Signal Components:**")
        components = signal_details.split(" | ")
        for component in components[:3]:  # Show first 3 components
            st.write(f"• {component}")
    
    # Traditional Technical Signal Details
    st.subheader("💡 Technical Signal Details")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        signal_color = {
            'Strong Buy': 'green',
            'Buy': 'orange', 
            'Watch': 'blue',
            'Hold': 'gray',
            'Error': 'red'
        }.get(purchase_signal, 'gray')
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {signal_color}20; border: 2px solid {signal_color}'>
            <h3 style='color: {signal_color}; margin: 0;'>{purchase_signal}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Signal Details:**")
        st.write(signal_details)
        
        if support and resistance:
            st.write("**Key Levels:**")
            st.write(f"Support: ₹{support:.2f}")
            st.write(f"Resistance: ₹{resistance:.2f}")
            
            # Calculate risk-reward
            if purchase_signal in ['Strong Buy', 'Buy']:
                stop_loss = support * 0.98  # 2% below support
                target = resistance * 1.02   # 2% above resistance
                risk = current_price - stop_loss
                reward = target - current_price
                rr_ratio = reward / risk if risk > 0 else 0
                
                st.write(f"**Risk-Reward Analysis:**")
                st.write(f"Entry: ₹{current_price:.2f}")
                st.write(f"Stop Loss: ₹{stop_loss:.2f}")
                st.write(f"Target: ₹{target:.2f}")
                st.write(f"R:R Ratio: 1:{rr_ratio:.2f}")
    
    # Breakout information
    if bullish_breakout or bearish_breakout:
        breakout_type = "Bullish Breakout 📈" if bullish_breakout else "Bearish Breakout 📉"
        breakout_color = "green" if bullish_breakout else "red"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {breakout_color}20; border: 1px solid {breakout_color}'>
            <strong style='color: {breakout_color};'>{breakout_type} Detected!</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Main candlestick chart
    st.subheader("📊 Price Chart with Technical Indicators")
    
    # Prepare indicators for chart
    chart_indicators = {
        'MA20': ma_values.get('MA20', np.array([])),
        'MA44': ma_values.get('MA44', np.array([])),
        'MA200': ma_values.get('MA200', np.array([])),
        'BB_Upper': bb_upper,
        'BB_Middle': bb_middle,
        'BB_Lower': bb_lower,
        'MACD': macd_line,
        'MACD_Signal': macd_signal_line,
        'MACD_Histogram': macd_histogram
    }
    
    # Create and display the main chart
    main_chart = create_candlestick_chart(stock_data, selected_stock, chart_indicators, height=700)
    st.plotly_chart(main_chart, use_container_width=True)
    
    # Additional indicator charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RSI Analysis")
        if len(rsi_values) > 0:
            rsi_chart = create_rsi_chart(stock_data, rsi_values, selected_stock)
            st.plotly_chart(rsi_chart, use_container_width=True)
        else:
            st.warning("Unable to calculate RSI")
    
    with col2:
        st.subheader("Volume Analysis")
        if len(volume_sma) > 0:
            volume_chart = create_volume_analysis_chart(stock_data, volume_sma, selected_stock)
            st.plotly_chart(volume_chart, use_container_width=True)
        else:
            st.warning("Unable to calculate volume indicators")
    
    # Detailed technical analysis table
    st.subheader("📋 Detailed Technical Analysis")
    
    # Calculate all indicators for detailed view
    all_indicators = calculate_all_indicators(stock_data)
    
    if all_indicators:
        analysis_data = {
            'Indicator': [
                'Current Price', 'RSI (14)', 'MACD', 'MACD Signal', 'MACD Histogram',
                'MA 20', 'MA 44', 'MA 50', 'MA 200', 'Bollinger Upper', 'Bollinger Middle',
                'Bollinger Lower', 'Support Level', 'Resistance Level', 'Volume Ratio'
            ],
            'Value': [
                f"₹{all_indicators.get('Current_Price', 0):.2f}",
                f"{all_indicators.get('RSI', 0):.2f}",
                f"{all_indicators.get('MACD', 0):.4f}",
                f"{all_indicators.get('MACD_Signal', 0):.4f}",
                f"{all_indicators.get('MACD_Histogram', 0):.4f}",
                f"₹{all_indicators.get('MA20', 0):.2f}",
                f"₹{all_indicators.get('MA44', 0):.2f}",
                f"₹{all_indicators.get('MA50', 0):.2f}",
                f"₹{all_indicators.get('MA200', 0):.2f}",
                f"₹{all_indicators.get('BB_Upper', 0):.2f}",
                f"₹{all_indicators.get('BB_Middle', 0):.2f}",
                f"₹{all_indicators.get('BB_Lower', 0):.2f}",
                f"₹{all_indicators.get('Support', 0):.2f}",
                f"₹{all_indicators.get('Resistance', 0):.2f}",
                f"{all_indicators.get('Volume_Ratio', 0):.2f}x"
            ],
            'Signal': [
                'Current Market Price',
                'Oversold' if all_indicators.get('RSI', 50) <= 30 else 'Overbought' if all_indicators.get('RSI', 50) >= 70 else 'Neutral',
                'Bullish' if all_indicators.get('MACD', 0) > all_indicators.get('MACD_Signal', 0) else 'Bearish',
                'Reference Line',
                'Momentum Indicator',
                'Above' if all_indicators.get('Current_Price', 0) > all_indicators.get('MA20', 0) else 'Below',
                'Above' if all_indicators.get('Current_Price', 0) > all_indicators.get('MA44', 0) else 'Below',
                'Above' if all_indicators.get('Current_Price', 0) > all_indicators.get('MA50', 0) else 'Below',
                'Above' if all_indicators.get('Current_Price', 0) > all_indicators.get('MA200', 0) else 'Below',
                'Resistance Level',
                'Mean Reversion Level',
                'Support Level',
                'Key Support Zone',
                'Key Resistance Zone',
                'High' if all_indicators.get('Volume_Ratio', 1) > 1.5 else 'Normal' if all_indicators.get('Volume_Ratio', 1) > 0.8 else 'Low'
            ]
        }
        
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)
    
    # Watchlist actions
    st.subheader("⭐ Watchlist Actions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if selected_stock in st.session_state.watchlist:
            st.success(f"{selected_stock} is already in your watchlist")
        else:
            st.info(f"Add {selected_stock} to your watchlist for tracking")
    
    with col2:
        if selected_stock not in st.session_state.watchlist:
            if st.button("➕ Add to Watchlist"):
                st.session_state.watchlist.append(selected_stock)
                st.success(f"Added {selected_stock} to watchlist")
                st.rerun()
        else:
            if st.button("➖ Remove from Watchlist"):
                st.session_state.watchlist.remove(selected_stock)
                st.success(f"Removed {selected_stock} from watchlist")
                st.rerun()
    
    # Historical performance
    st.subheader("📈 Historical Performance")
    
    periods = ['1W', '1M', '3M', '6M', '1Y']
    returns = []
    
    for period_days in [7, 30, 90, 180, 365]:
        if len(stock_data) > period_days:
            start_price = stock_data['Close'].iloc[-period_days-1]
            end_price = stock_data['Close'].iloc[-1]
            period_return = ((end_price - start_price) / start_price) * 100
            returns.append(period_return)
        else:
            returns.append(0)
    
    performance_cols = st.columns(len(periods))
    
    for i, (period, ret) in enumerate(zip(periods, returns)):
        with performance_cols[i]:
            st.metric(period, f"{ret:+.2f}%")
    
    # Comprehensive Export Section
    st.subheader("📤 Export Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced PDF Export
        if st.button("📄 Export Complete PDF Report", type="primary", help="Generate complete PDF with all technical analysis data and charts"):
            try:
                from utils.enhanced_pdf_export import EnhancedPDFExporter
                
                # Store technical analysis data in session state for export
                st.session_state.technical_analysis = {
                    'Stock Symbol': selected_stock,
                    'Current Price': f"₹{all_indicators.get('Current_Price', 0):.2f}",
                    'Purchase Signal': purchase_signal,
                    'Trend Status': trend_status,
                    'RSI': f"{all_indicators.get('RSI', 0):.1f}",
                    'MACD Signal': 'Bullish' if all_indicators.get('MACD', 0) > all_indicators.get('MACD_Signal', 0) else 'Bearish',
                    'Support Level': f"₹{all_indicators.get('Support', 0):.2f}",
                    'Resistance Level': f"₹{all_indicators.get('Resistance', 0):.2f}",
                    'Volume Status': 'High' if all_indicators.get('Volume_Ratio', 1) > 1.5 else 'Normal'
                }
                
                # Create enhanced exporter
                exporter = EnhancedPDFExporter()
                
                # Generate comprehensive PDF with all current data
                pdf_bytes = exporter.create_comprehensive_pdf_report(
                    portfolio_data=None,
                    risk_data=None,
                    charts=None,
                    technical_data=st.session_state.technical_analysis,
                    watchlist_data=None,
                    ml_signals=None
                )
                
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download Complete Report",
                        data=pdf_bytes,
                        file_name=f"technical_analysis_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help=f"Complete technical analysis report for {selected_stock}",
                        type="primary"
                    )
                    st.success("✅ PDF report generated successfully!")
                else:
                    st.error("❌ Failed to generate PDF report")
                
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
    
    with col2:
        # Excel Export
        if st.button("📊 Export Excel Report"):
            try:
                from utils.comprehensive_export import ComprehensiveExporter
                
                exporter = ComprehensiveExporter()
                
                # Prepare data for Excel
                excel_data = {}
                
                # Technical indicators
                indicators_data = []
                for key, value in all_indicators.items():
                    indicators_data.append({
                        'Indicator': key.replace('_', ' ').title(),
                        'Value': f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    })
                excel_data['Technical Indicators'] = pd.DataFrame(indicators_data)
                
                # Historical performance
                performance_data = []
                for period, ret in zip(periods, returns):
                    performance_data.append({'Period': period, 'Return (%)': ret})
                excel_data['Performance'] = pd.DataFrame(performance_data)
                
                # Price data sample (last 30 days)
                recent_data = stock_data.tail(30).copy()
                recent_data['Date'] = recent_data.index.strftime('%Y-%m-%d')
                excel_data['Price Data'] = recent_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True)
                
                # Analysis summary
                summary_data = [{
                    'Stock': selected_stock,
                    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Purchase Signal': purchase_signal,
                    'Trend': trend_status,
                    'Current Price': all_indicators.get('Current_Price', 0),
                    'RSI': all_indicators.get('RSI', 0),
                    'Support': all_indicators.get('Support', 0),
                    'Resistance': all_indicators.get('Resistance', 0)
                }]
                excel_data['Summary'] = pd.DataFrame(summary_data)
                
                # Generate Excel
                excel_bytes = exporter.create_excel_report(excel_data)
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_bytes,
                    file_name=f"technical_analysis_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Comprehensive Excel report with multiple tabs for {selected_stock}"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {e}")
    
    with col3:
        # Chart Export
        if st.button("📊 Export Chart"):
            try:
                # Export the main price chart as PNG
                import zipfile
                
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    # Create and export main chart
                    chart = create_enhanced_candlestick_chart(
                        stock_data, selected_stock, all_indicators, period, interval,
                        show_volume, show_ma, show_bb, show_rsi, show_macd
                    )
                    
                    # Convert chart to PNG
                    img_bytes = chart.to_image(format="png", width=1200, height=800)
                    zip_file.writestr(f"{selected_stock}_technical_chart.png", img_bytes)
                    
                    # Create summary text file
                    summary_text = f"""Technical Analysis Summary - {selected_stock}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Price: ₹{all_indicators.get('Current_Price', 0):.2f}
Purchase Signal: {purchase_signal}
Trend Status: {trend_status}
RSI: {all_indicators.get('RSI', 0):.1f}
Support: ₹{all_indicators.get('Support', 0):.2f}
Resistance: ₹{all_indicators.get('Resistance', 0):.2f}

Performance Returns:
{chr(10).join([f'{period}: {ret:+.2f}%' for period, ret in zip(periods, returns)])}
"""
                    zip_file.writestr(f"{selected_stock}_analysis_summary.txt", summary_text)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Chart & Summary",
                    data=zip_buffer.getvalue(),
                    file_name=f"technical_analysis_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    help=f"Chart image and analysis summary for {selected_stock}"
                )
                
            except Exception as e:
                st.error(f"Error generating chart export: {e}")

else:
    st.info("👆 Please select a stock symbol to start technical analysis")
    
    # Show help information
    with st.expander("ℹ️ How to use Technical Analysis"):
        st.markdown("""
        **Step 1:** Search and select a stock symbol
        
        **Step 2:** Choose analysis period and chart interval
        
        **Step 3:** Adjust technical indicator parameters if needed
        
        **Available Indicators:**
        - **RSI**: Relative Strength Index (momentum oscillator)
        - **MACD**: Moving Average Convergence Divergence
        - **Bollinger Bands**: Price volatility bands
        - **Moving Averages**: Trend indicators (20, 44, 50, 200 periods)
        - **Volume Analysis**: Trading volume patterns
        - **Support/Resistance**: Key price levels
        
        **Trading Signals:**
        - **Strong Buy**: Multiple bullish indicators align
        - **Buy**: Some bullish signals present
        - **Watch**: Mixed signals, monitor closely
        - **Hold**: No clear directional bias
        
        **Chart Features:**
        - Interactive candlestick charts
        - Technical indicator overlays
        - Volume analysis
        - Zoom and pan capabilities
        """)

# Display current watchlist in sidebar
if st.session_state.watchlist:
    st.sidebar.subheader("⭐ Current Watchlist")
    for i, symbol in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(symbol, key=f"watchlist_goto_{i}"):
                # This would navigate to the stock if we implement navigation
                st.experimental_set_query_params(stock=symbol)
        with col2:
            if st.button("❌", key=f"remove_watchlist_{i}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()
