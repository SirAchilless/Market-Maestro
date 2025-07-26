import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import io

# Import utility functions
from utils.data_fetcher import get_multiple_stocks_data, get_stock_info, get_current_price, format_market_cap
from utils.technical_analysis import calculate_all_indicators, get_purchase_signal, get_trend_status
from utils.fundamental_analysis import get_fundamental_data, get_fundamental_score, get_fundamental_rating
from utils.chart_generator import create_comparison_chart, create_heatmap
from utils.stock_lists import get_stock_sector, get_stock_market_cap_category, search_stock, get_stock_display_name
from utils.stock_validator import parse_imported_stocks, validate_stock_symbol

st.set_page_config(page_title="Watchlist", page_icon="⭐", layout="wide")

st.title("⭐ Stock Watchlist")
st.markdown("Monitor your selected stocks and prepare for algo trading")

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'watchlist_data' not in st.session_state:
    st.session_state.watchlist_data = {}

if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None

# Sidebar for watchlist management
st.sidebar.header("📋 Watchlist Management")

# Add new stock
st.sidebar.subheader("➕ Add Stock")
new_stock_input = st.sidebar.text_input(
    "Search Stock by Symbol or Company Name", 
    placeholder="e.g., RELIANCE, TCS, Infosys",
    help="Search from 500+ Indian stocks"
)

if new_stock_input:
    search_results = search_stock(new_stock_input)
    if search_results:
        # Create display options with company names
        stock_options = []
        for stock in search_results:
            display_name = get_stock_display_name(stock)
            stock_options.append(f"{stock} - {display_name}")
            
        selected_option = st.sidebar.selectbox("Select from matches", options=stock_options)
        selected_new_stock = selected_option.split(" - ")[0] if selected_option else None
    else:
        selected_new_stock = new_stock_input.upper()
    
    if st.sidebar.button("Add to Watchlist"):
        if selected_new_stock and selected_new_stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(selected_new_stock)
            st.sidebar.success(f"Added {selected_new_stock}")
            st.rerun()
        elif selected_new_stock in st.session_state.watchlist:
            st.sidebar.warning("Stock already in watchlist")
        else:
            st.sidebar.error("Please enter a valid stock symbol")

# Quick add popular stocks
st.sidebar.subheader("🔥 Quick Add")
popular_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'ITC', 'LT']
available_popular = [stock for stock in popular_stocks if stock not in st.session_state.watchlist]

if available_popular:
    quick_add_stock = st.sidebar.selectbox("Popular Stocks", options=["Select..."] + available_popular)
    if quick_add_stock != "Select..." and st.sidebar.button("Quick Add"):
        st.session_state.watchlist.append(quick_add_stock)
        st.sidebar.success(f"Added {quick_add_stock}")
        st.rerun()

# Bulk operations
st.sidebar.subheader("🔧 Bulk Operations")
if st.sidebar.button("Clear All"):
    if st.session_state.watchlist:
        st.session_state.watchlist.clear()
        st.session_state.watchlist_data.clear()
        st.sidebar.success("Watchlist cleared")
        st.rerun()

# Export watchlist
if st.session_state.watchlist:
    watchlist_text = "\n".join(st.session_state.watchlist)
    st.sidebar.download_button(
        "📤 Export Watchlist",
        data=watchlist_text,
        file_name=f"watchlist_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# Import watchlist
st.sidebar.subheader("📥 Import Watchlist")
uploaded_file = st.sidebar.file_uploader(
    "Upload Stock List", 
    type=['txt', 'csv'], 
    help="Upload a text or CSV file with stock symbols (one per line or comma-separated)"
)

if uploaded_file is not None:
    try:
        # Read and parse file content
        content = str(uploaded_file.read(), "utf-8")
        parse_result = parse_imported_stocks(content, uploaded_file.name)
        
        # Display file info
        st.sidebar.write(f"📄 **File:** {uploaded_file.name}")
        st.sidebar.write(f"📊 **Analysis:**")
        st.sidebar.write(f"• Total entries: {parse_result['total_found']}")
        st.sidebar.write(f"• Valid stocks: {parse_result['valid_stocks']}")
        
        # Show warnings for problematic stocks
        if parse_result['delisted_stocks']:
            st.sidebar.warning(f"⚠️ Skipped {len(parse_result['delisted_stocks'])} delisted stocks")
            with st.sidebar.expander("🚫 Delisted stocks skipped"):
                for stock in parse_result['delisted_stocks'][:10]:
                    st.sidebar.write(f"• {stock}")
        
        if parse_result['invalid_stocks']:
            st.sidebar.info(f"ℹ️ Skipped {len(parse_result['invalid_stocks'])} invalid entries")
        
        # Process valid stocks
        imported_stocks = parse_result['stocks']
        
        if imported_stocks:
            # Show preview of valid stocks
            with st.sidebar.expander("🔍 Preview valid stocks"):
                for stock in imported_stocks[:10]:
                    display_name = get_stock_display_name(stock)
                    st.sidebar.write(f"• **{stock}** - {display_name}")
                if len(imported_stocks) > 10:
                    st.sidebar.write(f"... and {len(imported_stocks) - 10} more")
            
            # Check for new stocks (not already in watchlist)
            new_stocks = [stock for stock in imported_stocks if stock not in st.session_state.watchlist]
            
            if new_stocks:
                st.sidebar.success(f"✅ {len(new_stocks)} new stocks ready to import")
                
                if st.sidebar.button("📥 Import Stocks", type="primary"):
                    st.session_state.watchlist.extend(new_stocks)
                    st.sidebar.balloons()
                    st.sidebar.success(f"🎉 Successfully imported {len(new_stocks)} stocks!")
                    st.rerun()
            else:
                st.sidebar.info("ℹ️ No new stocks to import (all already in watchlist)")
        else:
            st.sidebar.error("❌ No valid stock symbols found in file")
            st.sidebar.write("💡 **Tips:**")
            st.sidebar.write("• Use NSE symbols (e.g., RELIANCE, TCS)")
            st.sidebar.write("• One symbol per line in text files")
            st.sidebar.write("• First column for CSV files")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error processing file: {str(e)}")
        st.sidebar.write("💡 Try uploading a simple text file with stock symbols")

# Main content
if not st.session_state.watchlist:
    st.info("📝 Your watchlist is empty. Add stocks using the sidebar to get started.")
    
    # Show sample watchlist suggestion
    with st.expander("💡 Suggested Starter Watchlist"):
        st.markdown("""
        **Large Cap Diversified Portfolio:**
        - **RELIANCE** - Energy & Petrochemicals
        - **TCS** - Information Technology
        - **HDFCBANK** - Banking
        - **INFY** - Information Technology
        - **ITC** - FMCG
        - **LT** - Capital Goods
        - **MARUTI** - Automotive
        - **SUNPHARMA** - Pharmaceuticals
        
        **Benefits of a Watchlist:**
        - Monitor multiple stocks simultaneously
        - Track performance and trends
        - Get alerts for trading opportunities
        - Export data for algo trading systems
        - Compare relative performance
        """)
        
        if st.button("🚀 Add Starter Watchlist"):
            starter_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC', 'LT', 'MARUTI', 'SUNPHARMA']
            st.session_state.watchlist.extend(starter_stocks)
            st.success("Added starter watchlist!")
            st.rerun()

else:
    # Data refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📊 Monitoring {len(st.session_state.watchlist)} Stocks")
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh (5min)", value=False)
    
    with col3:
        if st.button("🔄 Refresh Now") or auto_refresh:
            with st.spinner("Refreshing watchlist data..."):
                # Clear cache to get fresh data
                st.cache_data.clear()
                # Fetch fresh data for all watchlist stocks with more historical data
                fresh_data = get_multiple_stocks_data(st.session_state.watchlist, period='6mo')
                st.session_state.watchlist_data = fresh_data
                st.session_state.last_updated = datetime.now()
            st.success("Data refreshed!")
            time.sleep(1)
            st.rerun()
    
    # Last updated info
    if st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get watchlist data
    if not st.session_state.watchlist_data:
        with st.spinner("Loading watchlist data..."):
            watchlist_data = get_multiple_stocks_data(st.session_state.watchlist, period='6mo')
            st.session_state.watchlist_data = watchlist_data
            st.session_state.last_updated = datetime.now()
    else:
        watchlist_data = st.session_state.watchlist_data
    
    # Process data for display
    watchlist_summary = []
    
    # Debug information
    with st.expander("🔍 Debug Information", expanded=False):
        for symbol in st.session_state.watchlist:
            data = watchlist_data.get(symbol, pd.DataFrame())
            if not data.empty:
                st.write(f"**{symbol}**: {len(data)} data points available")
            else:
                st.write(f"**{symbol}**: No data available")
    
    for symbol in st.session_state.watchlist:
        try:
            data = watchlist_data.get(symbol, pd.DataFrame())
            
            if not data.empty and len(data) >= 20:  # Reduced from 50 to 20
                # Calculate indicators
                indicators = calculate_all_indicators(data)
                purchase_signal, signal_details = get_purchase_signal(data)
                trend_status = get_trend_status(indicators)
                
                # Get fundamental data
                fundamentals = get_fundamental_data(symbol)
                fundamental_score = get_fundamental_score(fundamentals)
                
                # Get stock info
                sector = get_stock_sector(symbol)
                market_cap_category = get_stock_market_cap_category(symbol)
                
                stock_summary = {
                    'Symbol': symbol,
                    'Sector': sector,
                    'Category': market_cap_category,
                    'Current Price': indicators.get('Current_Price', 0),
                    'Price Change': indicators.get('Price_Change', 0),
                    'Price Change %': indicators.get('Price_Change_Pct', 0),
                    'RSI': indicators.get('RSI', 50),
                    'MACD Signal': 'Bullish' if indicators.get('MACD', 0) > indicators.get('MACD_Signal', 0) else 'Bearish',
                    'Volume Ratio': indicators.get('Volume_Ratio', 0),
                    'Purchase Signal': purchase_signal,
                    'Trend': trend_status,
                    'PE Ratio': fundamentals.get('PE Ratio', 0),
                    'Market Cap': fundamentals.get('Market Cap', 0),
                    'Fundamental Score': fundamental_score,
                    'Support': indicators.get('Support', 0),
                    'Resistance': indicators.get('Resistance', 0),
                    '1W Return': ((data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100) if len(data) >= 6 else 0,
                    '1M Return': ((data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100) if len(data) >= 21 else 0,
                    'Data Available': True
                }
                
                watchlist_summary.append(stock_summary)
            
            elif not data.empty and len(data) >= 5:  # Handle stocks with some data
                # Get basic price information
                current_price = data['Close'].iloc[-1] if len(data) > 0 else 0
                price_change = ((current_price / data['Close'].iloc[0] - 1) * 100) if len(data) > 1 else 0
                
                # Simple RSI calculation for limited data
                if len(data) >= 14:
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                else:
                    current_rsi = 50
                
                # Simple trend analysis
                if len(data) >= 5:
                    recent_trend = 'Bullish' if data['Close'].iloc[-1] > data['Close'].iloc[-5] else 'Bearish'
                    simple_signal = 'Buy' if recent_trend == 'Bullish' and current_rsi < 70 else 'Hold'
                else:
                    recent_trend = 'Neutral'
                    simple_signal = 'Hold'
                
                # Get fundamental data
                fundamentals = get_fundamental_data(symbol)
                fundamental_score = get_fundamental_score(fundamentals)
                
                # Get stock info
                sector = get_stock_sector(symbol)
                market_cap_category = get_stock_market_cap_category(symbol)
                
                stock_summary = {
                    'Symbol': symbol,
                    'Sector': sector,
                    'Category': market_cap_category,
                    'Current Price': current_price,
                    'Price Change': price_change,
                    'Price Change %': price_change,
                    'RSI': current_rsi,
                    'MACD Signal': recent_trend,
                    'Volume Ratio': data['Volume'].iloc[-1] / data['Volume'].mean() if len(data) > 1 else 1.0,
                    'Purchase Signal': simple_signal,
                    'Trend': recent_trend,
                    'PE Ratio': fundamentals.get('PE Ratio', 0),
                    'Market Cap': fundamentals.get('Market Cap', 0),
                    'Fundamental Score': fundamental_score,
                    'Support': data['Close'].min() if len(data) > 0 else 0,
                    'Resistance': data['Close'].max() if len(data) > 0 else 0,
                    '1W Return': ((data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100) if len(data) >= 6 else 0,
                    '1M Return': 0,  # Not enough data for monthly return
                    'Data Available': True
                }
                
                watchlist_summary.append(stock_summary)
            
            else:
                # Handle stocks with very limited or no data
                current_price = get_current_price(symbol)
                
                # Get stock info
                sector = get_stock_sector(symbol)
                market_cap_category = get_stock_market_cap_category(symbol)
                
                # Try to get some basic info even with limited data
                fundamentals = get_fundamental_data(symbol)
                
                stock_summary = {
                    'Symbol': symbol,
                    'Sector': sector,
                    'Category': market_cap_category,
                    'Current Price': current_price if current_price > 0 else 0,
                    'Price Change': 0,
                    'Price Change %': 0,
                    'RSI': 50,  # Neutral RSI
                    'MACD Signal': 'Neutral',
                    'Volume Ratio': 1.0,
                    'Purchase Signal': 'Watch' if current_price > 0 else 'No Data',
                    'Trend': 'Neutral',
                    'PE Ratio': fundamentals.get('PE Ratio', 0),
                    'Market Cap': fundamentals.get('Market Cap', 0),
                    'Fundamental Score': get_fundamental_score(fundamentals),
                    'Support': current_price * 0.95 if current_price > 0 else 0,  # Estimate 5% below current
                    'Resistance': current_price * 1.05 if current_price > 0 else 0,  # Estimate 5% above current
                    '1W Return': 0,
                    '1M Return': 0,
                    'Data Available': False
                }
                watchlist_summary.append(stock_summary)
        
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not watchlist_summary:
        st.error("Unable to load data for any stocks in your watchlist.")
        st.stop()
    
    # Convert to DataFrame
    df = pd.DataFrame(watchlist_summary)
    
    # Performance overview
    st.subheader("📈 Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        positive_stocks = len(df[df['Price Change %'] > 0])
        st.metric("Gainers", f"{positive_stocks}/{len(df)}")
    
    with col2:
        strong_buy_count = len(df[df['Purchase Signal'] == 'Strong Buy'])
        st.metric("Strong Buy Signals", strong_buy_count)
    
    with col3:
        bullish_count = len(df[df['Trend'] == 'Bullish'])
        st.metric("Bullish Trends", bullish_count)
    
    with col4:
        avg_fundamental_score = df['Fundamental Score'].mean()
        st.metric("Avg Fundamental Score", f"{avg_fundamental_score:.1f}")
    
    with col5:
        high_volume_count = len(df[df['Volume Ratio'] > 1.5])
        st.metric("High Volume", high_volume_count)
    
    # Display options
    st.subheader("📊 Watchlist Data")
    
    # View selection
    view_option = st.radio(
        "Select View",
        options=["Overview", "Technical", "Fundamental", "Performance", "Detailed"],
        horizontal=True
    )
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        signal_filter = st.selectbox(
            "Filter by Signal",
            options=["All", "Strong Buy", "Buy", "Watch", "Hold"],
            index=0
        )
    
    with col2:
        trend_filter = st.selectbox(
            "Filter by Trend",
            options=["All", "Bullish", "Bearish", "Neutral"],
            index=0
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if signal_filter != "All":
        filtered_df = filtered_df[filtered_df['Purchase Signal'] == signal_filter]
    
    if trend_filter != "All":
        filtered_df = filtered_df[filtered_df['Trend'] == trend_filter]
    
    # Select columns based on view
    if view_option == "Overview":
        display_columns = ['Symbol', 'Current Price', 'Price Change %', 'Purchase Signal', 'Trend', 'RSI']
    elif view_option == "Technical":
        display_columns = ['Symbol', 'Current Price', 'Price Change %', 'RSI', 'MACD Signal', 'Volume Ratio', 'Support', 'Resistance']
    elif view_option == "Fundamental":
        display_columns = ['Symbol', 'PE Ratio', 'Market Cap', 'Fundamental Score', 'Sector', 'Category']
    elif view_option == "Performance":
        display_columns = ['Symbol', 'Current Price', 'Price Change %', '1W Return', '1M Return', 'Volume Ratio']
    else:  # Detailed
        display_columns = ['Symbol', 'Sector', 'Current Price', 'Price Change %', 'RSI', 'Purchase Signal', 
                          'Trend', 'PE Ratio', 'Fundamental Score', '1W Return', '1M Return']
    
    # Format the display
    display_df = filtered_df[display_columns].copy()
    
    # Format numerical columns
    if 'Current Price' in display_df.columns:
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
    
    if 'Price Change %' in display_df.columns:
        display_df['Price Change %'] = display_df['Price Change %'].apply(lambda x: f"{x:+.2f}%")
    
    if 'RSI' in display_df.columns:
        display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x > 0 else "N/A")
    
    if 'PE Ratio' in display_df.columns:
        display_df['PE Ratio'] = display_df['PE Ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x > 0 else "N/A")
    
    if 'Market Cap' in display_df.columns:
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: format_market_cap(x))
    
    if 'Volume Ratio' in display_df.columns:
        display_df['Volume Ratio'] = display_df['Volume Ratio'].apply(lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) and x > 0 else "N/A")
    
    if 'Support' in display_df.columns:
        display_df['Support'] = display_df['Support'].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) and x > 0 else "N/A")
    
    if 'Resistance' in display_df.columns:
        display_df['Resistance'] = display_df['Resistance'].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) and x > 0 else "N/A")
    
    if '1W Return' in display_df.columns:
        display_df['1W Return'] = display_df['1W Return'].apply(lambda x: f"{x:+.2f}%")
    
    if '1M Return' in display_df.columns:
        display_df['1M Return'] = display_df['1M Return'].apply(lambda x: f"{x:+.2f}%")
    
    # Style the dataframe
    def highlight_signals(val):
        if val == 'Strong Buy':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Buy':
            return 'background-color: #fff3cd; color: #856404'
        elif val == 'Watch':
            return 'background-color: #e2e3e5; color: #383d41'
        return ''
    
    def highlight_trends(val):
        if val == 'Bullish':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Bearish':
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_df = display_df.style
    
    if 'Purchase Signal' in display_df.columns:
        styled_df = styled_df.applymap(highlight_signals, subset=['Purchase Signal'])
    
    if 'Trend' in display_df.columns:
        styled_df = styled_df.applymap(highlight_trends, subset=['Trend'])
    
    # Display the table with action buttons
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Individual stock actions
    st.subheader("🎯 Stock Actions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stock_for_action = st.selectbox(
            "Select stock for detailed analysis",
            options=st.session_state.watchlist
        )
    
    with col2:
        if st.button("📊 Analyze"):
            st.session_state.selected_stock = selected_stock_for_action
            st.info(f"Navigate to Technical Analysis page for {selected_stock_for_action}")
    
    # Remove stocks section
    st.subheader("🗑️ Remove Stocks")
    
    stocks_to_remove = st.multiselect(
        "Select stocks to remove from watchlist",
        options=st.session_state.watchlist
    )
    
    if stocks_to_remove:
        if st.button("🗑️ Remove Selected"):
            for stock in stocks_to_remove:
                st.session_state.watchlist.remove(stock)
                if stock in st.session_state.watchlist_data:
                    del st.session_state.watchlist_data[stock]
            st.success(f"Removed {len(stocks_to_remove)} stocks from watchlist")
            st.rerun()
    
    # Performance charts
    st.subheader("📊 Performance Visualization")
    
    chart_type = st.radio(
        "Select Chart Type",
        options=["Comparison Chart", "Performance Heatmap"],
        horizontal=True
    )
    
    if chart_type == "Comparison Chart" and len(st.session_state.watchlist) > 1:
        # Create comparison chart for available stocks
        available_data = {symbol: data for symbol, data in watchlist_data.items() 
                         if not data.empty and len(data) > 20}
        
        if available_data:
            comparison_chart = create_comparison_chart(
                available_data, 
                list(available_data.keys()), 
                period='1mo'
            )
            st.plotly_chart(comparison_chart, use_container_width=True)
        else:
            st.warning("Insufficient data for comparison chart")
    
    elif chart_type == "Performance Heatmap":
        # Create performance heatmap
        if len(df) > 0:
            heatmap_data = {}
            for _, stock in df.iterrows():
                heatmap_data[stock['Symbol']] = {'Price_Change_Pct': stock['Price Change %']}
            
            if heatmap_data:
                # Convert percentage strings back to numbers for heatmap
                for symbol in heatmap_data:
                    pct_str = heatmap_data[symbol]['Price_Change_Pct']
                    if isinstance(pct_str, str) and '%' in pct_str:
                        heatmap_data[symbol]['Price_Change_Pct'] = float(pct_str.replace('%', '').replace('+', ''))
                
                heatmap_chart = create_heatmap(heatmap_data, 'Price_Change_Pct')
                st.plotly_chart(heatmap_chart, use_container_width=True)
        else:
            st.warning("No data available for heatmap")
    
    # Algo trading preparation
    st.subheader("🤖 Algo Trading Preparation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Formats for Trading Systems:**")
        
        # Comprehensive Excel export with multiple tabs
        def create_excel_export():
            """Create Excel file with multiple tabs for different data views"""
            output = io.BytesIO()
            
            # Create a clean copy of the data with raw numeric values for Excel
            clean_df = df.copy()
            
            # Convert formatted strings back to numbers for Excel export
            for col in ['Current Price', 'PE Ratio', 'Support', 'Resistance']:
                if col in clean_df.columns:
                    clean_df[col] = pd.to_numeric(clean_df[col].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce')
            
            for col in ['Price Change %', '1W Return', '1M Return']:
                if col in clean_df.columns:
                    clean_df[col] = pd.to_numeric(clean_df[col].astype(str).str.replace('%', '').str.replace('+', ''), errors='coerce')
            
            if 'Volume Ratio' in clean_df.columns:
                clean_df['Volume Ratio'] = pd.to_numeric(clean_df['Volume Ratio'].astype(str).str.replace('x', ''), errors='coerce')
            
            if 'RSI' in clean_df.columns:
                clean_df['RSI'] = pd.to_numeric(clean_df['RSI'], errors='coerce')
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Overview Tab
                overview_columns = ['Symbol', 'Sector', 'Category', 'Current Price', 'Price Change %', 
                                  'Purchase Signal', 'Trend', 'RSI', 'Data Available']
                overview_df = clean_df[[col for col in overview_columns if col in clean_df.columns]].copy()
                overview_df.to_excel(writer, sheet_name='Overview', index=False)
                
                # Technical Tab
                technical_columns = ['Symbol', 'Current Price', 'Price Change', 'Price Change %', 
                                   'RSI', 'MACD Signal', 'Volume Ratio', 'Support', 'Resistance', 
                                   'Purchase Signal', 'Trend']
                technical_df = clean_df[[col for col in technical_columns if col in clean_df.columns]].copy()
                technical_df.to_excel(writer, sheet_name='Technical', index=False)
                
                # Fundamental Tab
                fundamental_columns = ['Symbol', 'Sector', 'Category', 'PE Ratio', 'Market Cap', 
                                     'Fundamental Score', 'Current Price', 'Purchase Signal']
                fundamental_df = clean_df[[col for col in fundamental_columns if col in clean_df.columns]].copy()
                fundamental_df.to_excel(writer, sheet_name='Fundamental', index=False)
                
                # Performance Tab
                performance_columns = ['Symbol', 'Current Price', 'Price Change', 'Price Change %', 
                                     '1W Return', '1M Return', 'Volume Ratio', 'Trend']
                performance_df = clean_df[[col for col in performance_columns if col in clean_df.columns]].copy()
                performance_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Detailed Tab (All data)
                detailed_df = clean_df.copy()
                detailed_df.to_excel(writer, sheet_name='Details', index=False)
                
                # Summary Tab with statistics
                price_change_numeric = clean_df['Price Change %']
                one_week_return_numeric = clean_df['1W Return']
                one_month_return_numeric = clean_df['1M Return']
                rsi_numeric = clean_df['RSI']
                volume_ratio_numeric = clean_df['Volume Ratio']
                
                summary_data = {
                    'Metric': [
                        'Total Stocks',
                        'Gainers',
                        'Losers',
                        'Strong Buy Signals',
                        'Buy Signals',
                        'Hold Signals',
                        'Bullish Trends',
                        'Bearish Trends',
                        'Neutral Trends',
                        'Average RSI',
                        'Average Fundamental Score',
                        'High Volume Stocks (>1.5x)',
                        'Average 1W Return %',
                        'Average 1M Return %'
                    ],
                    'Value': [
                        len(clean_df),
                        len(clean_df[price_change_numeric > 0]),
                        len(clean_df[price_change_numeric < 0]),
                        len(clean_df[clean_df['Purchase Signal'] == 'Strong Buy']),
                        len(clean_df[clean_df['Purchase Signal'] == 'Buy']),
                        len(clean_df[clean_df['Purchase Signal'] == 'Hold']),
                        len(clean_df[clean_df['Trend'] == 'Bullish']),
                        len(clean_df[clean_df['Trend'] == 'Bearish']),
                        len(clean_df[clean_df['Trend'] == 'Neutral']),
                        round(rsi_numeric.mean(), 1) if not rsi_numeric.isna().all() else 0,
                        round(clean_df['Fundamental Score'].mean(), 1),
                        len(clean_df[volume_ratio_numeric > 1.5]),
                        round(one_week_return_numeric.mean(), 2) if not one_week_return_numeric.isna().all() else 0,
                        round(one_month_return_numeric.mean(), 2) if not one_month_return_numeric.isna().all() else 0
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            return output.getvalue()
        
        # Excel export button
        excel_data = create_excel_export()
        st.download_button(
            "📈 Export Complete Excel (All Tabs)",
            data=excel_data,
            file_name=f"watchlist_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Exports all data views in separate Excel tabs: Overview, Technical, Fundamental, Performance, Details, and Summary"
        )
        
        # JSON export for APIs
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "📄 Export as JSON",
            data=json_data,
            file_name=f"watchlist_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # CSV export
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            "📊 Export as CSV",
            data=csv_data,
            file_name=f"watchlist_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.write("**Trading Signal Summary:**")
        
        signal_summary = df['Purchase Signal'].value_counts()
        for signal, count in signal_summary.items():
            st.write(f"• {signal}: {count} stocks")
        
        # Generate trading list
        strong_buy_stocks = df[df['Purchase Signal'] == 'Strong Buy']['Symbol'].tolist()
        buy_stocks = df[df['Purchase Signal'] == 'Buy']['Symbol'].tolist()
        
        if strong_buy_stocks or buy_stocks:
            trading_list = strong_buy_stocks + buy_stocks
            trading_text = "\n".join(trading_list)
            
            st.download_button(
                "🎯 Export Buy Signals",
                data=trading_text,
                file_name=f"buy_signals_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    # Alerts and notifications
    st.subheader("🔔 Alerts & Notifications")
    
    # Price alerts
    with st.expander("⚠️ Current Alerts"):
        alerts = []
        
        for _, stock in df.iterrows():
            symbol = stock['Symbol']
            current_price = stock['Current Price']
            
            # Extract numeric value from formatted price
            if isinstance(current_price, str):
                price_value = float(current_price.replace('₹', '').replace(',', ''))
            else:
                price_value = current_price
            
            # RSI alerts
            rsi = stock['RSI']
            if isinstance(rsi, (int, float)) and rsi > 0:
                if rsi <= 30:
                    alerts.append(f"🟢 {symbol}: RSI Oversold ({rsi:.1f})")
                elif rsi >= 70:
                    alerts.append(f"🔴 {symbol}: RSI Overbought ({rsi:.1f})")
            
            # Volume alerts
            volume_ratio = stock['Volume Ratio']
            if isinstance(volume_ratio, (int, float)) and volume_ratio > 2:
                alerts.append(f"📈 {symbol}: High Volume ({volume_ratio:.2f}x)")
            
            # Signal alerts
            if stock['Purchase Signal'] == 'Strong Buy':
                alerts.append(f"🚀 {symbol}: Strong Buy Signal")
        
        if alerts:
            for alert in alerts:
                st.write(alert)
        else:
            st.info("No active alerts")
    
    # Auto-refresh setup
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()

# Footer with watchlist statistics
if st.session_state.watchlist:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Stocks", len(st.session_state.watchlist))
    
    with col2:
        if watchlist_summary:
            sectors = [stock['Sector'] for stock in watchlist_summary]
            unique_sectors = len(set(sectors))
            st.metric("Sectors Covered", unique_sectors)
    
    with col3:
        if st.session_state.last_updated:
            time_diff = datetime.now() - st.session_state.last_updated
            minutes_ago = int(time_diff.total_seconds() / 60)
            st.metric("Data Age", f"{minutes_ago} min ago")

