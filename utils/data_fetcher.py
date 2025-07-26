import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period='1y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE.NS')
        period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        # Ensure symbol has .NS suffix for NSE stocks
        if not symbol.endswith('.NS') and not symbol.startswith('^'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            st.warning(f"No data found for symbol: {symbol}")
            return pd.DataFrame()
        
        # Clean column names
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        data = data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol):
    """
    Get stock information and fundamental data
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Stock information
    """
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'book_value': info.get('bookValue', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'current_price': info.get('currentPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'currency': info.get('currency', 'INR')
        }
    
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {str(e)}")
        return {}

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_index_data(index_symbol, period='1d'):
    """
    Get index data for market overview
    
    Args:
        index_symbol (str): Index symbol (e.g., '^NSEI')
        period (str): Data period
    
    Returns:
        pd.DataFrame: Index data
    """
    try:
        ticker = yf.Ticker(index_symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_multiple_stocks_data(symbols, period='1y'):
    """
    Fetch data for multiple stocks efficiently
    
    Args:
        symbols (list): List of stock symbols
        period (str): Data period
    
    Returns:
        dict: Dictionary with symbol as key and dataframe as value
    """
    stock_data = {}
    
    # Add .NS suffix if not present
    ns_symbols = []
    for symbol in symbols:
        if not symbol.endswith('.NS'):
            ns_symbols.append(f"{symbol}.NS")
        else:
            ns_symbols.append(symbol)
    
    try:
        # Use yfinance to download multiple tickers at once
        tickers = yf.download(ns_symbols, period=period, group_by='ticker', progress=False)
        
        if tickers is not None and not tickers.empty:
            if hasattr(tickers, 'columns') and isinstance(tickers.columns, pd.MultiIndex):
                # Multiple stocks
                for i, symbol in enumerate(ns_symbols):
                    try:
                        if symbol in tickers.columns.levels[0]:
                            stock_data[symbols[i]] = tickers[symbol].dropna()
                        else:
                            stock_data[symbols[i]] = pd.DataFrame()
                    except:
                        stock_data[symbols[i]] = pd.DataFrame()
            else:
                # Single stock
                if len(symbols) == 1:
                    stock_data[symbols[0]] = tickers.dropna()
        else:
            # If tickers is None or empty, fallback to individual requests
            for symbol in symbols:
                stock_data[symbol] = get_stock_data(symbol, period)
    
    except Exception as e:
        st.error(f"Error fetching multiple stocks data: {str(e)}")
        # Fallback to individual requests
        for symbol in symbols:
            stock_data[symbol] = get_stock_data(symbol, period)
    
    return stock_data

def format_market_cap(market_cap):
    """Format market cap in Indian number system"""
    if market_cap == 0:
        return "N/A"
    
    if market_cap >= 10000000000000:  # 10 Lakh Crores
        return f"₹{market_cap/10000000000000:.2f}L Cr"
    elif market_cap >= 100000000000:  # 1000 Crores
        return f"₹{market_cap/10000000000:.2f}K Cr"
    elif market_cap >= 10000000000:  # 100 Crores
        return f"₹{market_cap/10000000000:.0f} Cr"
    else:
        return f"₹{market_cap/10000000:.0f} L"

def get_current_price(symbol):
    """Get current price of a stock"""
    try:
        if not symbol.endswith('.NS') and not symbol.startswith('^'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0
    except:
        return 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_index_data(symbol):
    """Get index data including current price and change"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='2d')  # Get 2 days to calculate change
        
        if data.empty or len(data) < 1:
            return {
                'current_price': 'N/A',
                'change': 'N/A',
                'change_percent': 'N/A',
                'status': 'Data unavailable'
            }
        
        current_price = data['Close'].iloc[-1]
        
        # Calculate change if we have previous day data
        if len(data) >= 2:
            previous_price = data['Close'].iloc[-2]
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
        else:
            change = 0
            change_percent = 0
        
        return {
            'current_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'status': 'active'
        }
    
    except Exception as e:
        return {
            'current_price': 'N/A',
            'change': 'N/A', 
            'change_percent': 'N/A',
            'status': 'Data unavailable'
        }
