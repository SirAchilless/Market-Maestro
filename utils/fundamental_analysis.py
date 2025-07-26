import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fundamental_data(symbol):
    """
    Get fundamental data for a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Fundamental data
    """
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract fundamental metrics
        fundamentals = {
            'PE Ratio': info.get('trailingPE', 0),
            'Forward PE': info.get('forwardPE', 0),
            'PEG Ratio': info.get('pegRatio', 0),
            'Price to Book': info.get('priceToBook', 0),
            'EPS': info.get('trailingEps', 0),
            'Forward EPS': info.get('forwardEps', 0),
            'Book Value': info.get('bookValue', 0),
            'Market Cap': info.get('marketCap', 0),
            'Enterprise Value': info.get('enterpriseValue', 0),
            'Revenue': info.get('totalRevenue', 0),
            'Revenue Per Share': info.get('revenuePerShare', 0),
            'Profit Margin': info.get('profitMargins', 0),
            'Operating Margin': info.get('operatingMargins', 0),
            'ROA': info.get('returnOnAssets', 0),
            'ROE': info.get('returnOnEquity', 0),
            'ROCE': calculate_roce(info),
            'Debt to Equity': info.get('debtToEquity', 0),
            'Current Ratio': info.get('currentRatio', 0),
            'Quick Ratio': info.get('quickRatio', 0),
            'Interest Coverage': info.get('interestCoverage', 0),
            'Dividend Yield': info.get('dividendYield', 0),
            'Payout Ratio': info.get('payoutRatio', 0),
            'Beta': info.get('beta', 0),
            'Float Shares': info.get('floatShares', 0),
            'Shares Outstanding': info.get('sharesOutstanding', 0),
            'Held by Insiders': info.get('heldPercentInsiders', 0),
            'Held by Institutions': info.get('heldPercentInstitutions', 0),
            '52 Week High': info.get('fiftyTwoWeekHigh', 0),
            '52 Week Low': info.get('fiftyTwoWeekLow', 0),
            'Price to Sales': info.get('priceToSalesTrailing12Months', 0),
            'Enterprise to Revenue': info.get('enterpriseToRevenue', 0),
            'Enterprise to EBITDA': info.get('enterpriseToEbitda', 0)
        }
        
        return fundamentals
    
    except Exception as e:
        st.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
        return {}

def calculate_roce(info):
    """Calculate Return on Capital Employed"""
    try:
        ebit = info.get('ebitda', 0) - info.get('depreciation', 0) if info.get('depreciation') else info.get('ebitda', 0)
        total_assets = info.get('totalAssets', 0)
        current_liabilities = info.get('totalCurrentLiabilities', 0)
        
        if total_assets > 0 and current_liabilities >= 0:
            capital_employed = total_assets - current_liabilities
            if capital_employed > 0:
                return (ebit / capital_employed) * 100
        return 0
    except:
        return 0

def format_fundamental_value(key, value):
    """Format fundamental values for display"""
    try:
        if value is None or value == 0:
            return "N/A"
        
        # Percentage values
        percentage_keys = ['Profit Margin', 'Operating Margin', 'ROA', 'ROE', 'ROCE', 
                          'Dividend Yield', 'Payout Ratio', 'Held by Insiders', 'Held by Institutions']
        if any(k in key for k in percentage_keys):
            return f"{value*100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"
        
        # Currency values (in crores for Indian context)
        currency_keys = ['Market Cap', 'Enterprise Value', 'Revenue', 'Book Value']
        if any(k in key for k in currency_keys):
            if value >= 10000000000:  # 1000 crores
                return f"₹{value/10000000000:.2f}K Cr"
            elif value >= 100000000:  # 10 crores
                return f"₹{value/10000000:.0f} Cr"
            else:
                return f"₹{value/10000000:.2f} L"
        
        # Ratio values
        ratio_keys = ['PE Ratio', 'Forward PE', 'PEG Ratio', 'Price to Book', 'Debt to Equity', 
                     'Current Ratio', 'Quick Ratio', 'Price to Sales', 'Enterprise to Revenue', 
                     'Enterprise to EBITDA']
        if any(k in key for k in ratio_keys):
            return f"{value:.2f}x"
        
        # EPS and per share values
        if 'EPS' in key or 'Per Share' in key or key in ['52 Week High', '52 Week Low']:
            return f"₹{value:.2f}"
        
        # Coverage ratio
        if 'Coverage' in key:
            return f"{value:.2f}x"
        
        # Share counts
        if 'Shares' in key:
            if value >= 1000000000:  # 100 crores
                return f"{value/10000000:.0f} Cr"
            elif value >= 10000000:  # 1 crore
                return f"{value/10000000:.2f} Cr"
            else:
                return f"{value/100000:.2f} L"
        
        # Beta
        if key == 'Beta':
            return f"{value:.2f}"
        
        # Default formatting
        return f"{value:.2f}"
    
    except:
        return "N/A"

def get_fundamental_score(fundamentals):
    """Calculate a fundamental strength score (0-100)"""
    try:
        if not fundamentals:
            return 0
        
        score = 0
        max_score = 0
        
        # PE Ratio scoring (lower is better, but not too low)
        pe = fundamentals.get('PE Ratio', 0)
        if 10 <= pe <= 25:
            score += 15
        elif 5 <= pe < 10 or 25 < pe <= 35:
            score += 10
        elif pe > 0:
            score += 5
        max_score += 15
        
        # ROE scoring (higher is better)
        roe = fundamentals.get('ROE', 0)
        if roe >= 0.20:  # 20%+
            score += 20
        elif roe >= 0.15:  # 15%+
            score += 15
        elif roe >= 0.10:  # 10%+
            score += 10
        elif roe > 0:
            score += 5
        max_score += 20
        
        # ROCE scoring
        roce = fundamentals.get('ROCE', 0)
        if roce >= 20:
            score += 15
        elif roce >= 15:
            score += 12
        elif roce >= 10:
            score += 8
        elif roce > 0:
            score += 4
        max_score += 15
        
        # Debt to Equity scoring (lower is better)
        debt_equity = fundamentals.get('Debt to Equity', 0)
        if debt_equity <= 0.3:
            score += 10
        elif debt_equity <= 0.6:
            score += 8
        elif debt_equity <= 1.0:
            score += 5
        elif debt_equity <= 2.0:
            score += 2
        max_score += 10
        
        # Current Ratio scoring
        current_ratio = fundamentals.get('Current Ratio', 0)
        if 1.5 <= current_ratio <= 3.0:
            score += 10
        elif 1.0 <= current_ratio < 1.5 or 3.0 < current_ratio <= 5.0:
            score += 7
        elif current_ratio >= 1.0:
            score += 4
        max_score += 10
        
        # Profit Margin scoring
        profit_margin = fundamentals.get('Profit Margin', 0)
        if profit_margin >= 0.15:  # 15%+
            score += 15
        elif profit_margin >= 0.10:  # 10%+
            score += 12
        elif profit_margin >= 0.05:  # 5%+
            score += 8
        elif profit_margin > 0:
            score += 4
        max_score += 15
        
        # EPS Growth potential (basic check)
        eps = fundamentals.get('EPS', 0)
        forward_eps = fundamentals.get('Forward EPS', 0)
        if eps > 0 and forward_eps > eps:
            score += 15
        elif eps > 0:
            score += 8
        max_score += 15
        
        # Calculate final score
        final_score = (score / max_score) * 100 if max_score > 0 else 0
        return round(final_score, 1)
    
    except Exception as e:
        return 0

def get_fundamental_rating(score):
    """Get rating based on fundamental score"""
    if score >= 80:
        return "Excellent", "🟢"
    elif score >= 65:
        return "Good", "🟡"
    elif score >= 50:
        return "Average", "🟠"
    elif score >= 35:
        return "Below Average", "🔴"
    else:
        return "Poor", "⚫"

def compare_with_industry_avg(symbol, metric_name, metric_value):
    """Compare metric with industry average (simplified)"""
    # This is a simplified comparison - in production, you'd use industry data
    industry_benchmarks = {
        'PE Ratio': 20,
        'ROE': 0.15,
        'ROCE': 15,
        'Debt to Equity': 0.6,
        'Current Ratio': 2.0,
        'Profit Margin': 0.10
    }
    
    benchmark = industry_benchmarks.get(metric_name)
    if benchmark is None or metric_value == 0:
        return "N/A"
    
    if metric_name in ['PE Ratio', 'Debt to Equity']:
        # Lower is better
        if metric_value <= benchmark * 0.8:
            return "Better"
        elif metric_value <= benchmark * 1.2:
            return "Similar"
        else:
            return "Worse"
    else:
        # Higher is better
        if metric_value >= benchmark * 1.2:
            return "Better"
        elif metric_value >= benchmark * 0.8:
            return "Similar"
        else:
            return "Worse"
