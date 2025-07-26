import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from openai import OpenAI

# Import analysis utilities
from utils.technical_analysis import get_technical_score, get_technical_signal, calculate_all_indicators
from utils.fundamental_analysis import get_fundamental_data, get_fundamental_score, get_fundamental_rating
from utils.news_sentiment import get_news_sentiment_score
from utils.ml_signals import get_ml_signal, is_model_trained

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def generate_investment_narrative(symbol, stock_data, period="6mo"):
    """
    Generate comprehensive AI-powered investment narrative using all analysis components
    
    Args:
        symbol (str): Stock symbol
        stock_data (pd.DataFrame): Historical stock data
        period (str): Analysis period
    
    Returns:
        dict: Complete investment narrative with sections and recommendations
    """
    try:
        if stock_data.empty:
            return {"error": "No stock data available for analysis"}
        
        # Gather all analysis components
        analysis_components = gather_analysis_data(symbol, stock_data)
        
        # Generate AI narrative using OpenAI
        if OPENAI_API_KEY:
            narrative = generate_ai_narrative(symbol, analysis_components, period)
        else:
            narrative = generate_structured_narrative(symbol, analysis_components, period)
        
        return {
            "success": True,
            "narrative": narrative,
            "components": analysis_components,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "period": period
        }
    
    except Exception as e:
        return {
            "error": f"Failed to generate investment narrative: {str(e)}",
            "success": False
        }

def gather_analysis_data(symbol, stock_data):
    """Gather all analysis components for narrative generation"""
    components = {}
    
    # Basic stock information
    current_price = stock_data['Close'].iloc[-1]
    price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
    price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
    
    components['basic_info'] = {
        'symbol': symbol,
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volume': stock_data['Volume'].iloc[-1],
        'high_52w': stock_data['High'].max(),
        'low_52w': stock_data['Low'].min()
    }
    
    # Technical analysis
    try:
        tech_score = get_technical_score(stock_data)
        tech_signal = get_technical_signal(tech_score)
        tech_indicators = calculate_all_indicators(stock_data)
        
        components['technical'] = {
            'score': tech_score,
            'signal': tech_signal,
            'indicators': tech_indicators,
            'rsi': tech_indicators.get('RSI', 50),
            'macd': tech_indicators.get('MACD', 0),
            'trend': determine_trend(tech_indicators)
        }
    except Exception as e:
        components['technical'] = {'error': str(e)}
    
    # Fundamental analysis
    try:
        fundamental_data = get_fundamental_data(symbol)
        fund_score = get_fundamental_score(fundamental_data)
        fund_rating, fund_emoji = get_fundamental_rating(fund_score)
        
        components['fundamental'] = {
            'score': fund_score,
            'rating': fund_rating,
            'data': fundamental_data,
            'key_metrics': extract_key_fundamentals(fundamental_data)
        }
    except Exception as e:
        components['fundamental'] = {'error': str(e)}
    
    # ML analysis
    try:
        if is_model_trained():
            ml_result = get_ml_signal(stock_data, symbol)
            components['ml'] = {
                'signal': ml_result.get('signal', 'Hold'),
                'confidence': ml_result.get('confidence', 0),
                'ml_score': ml_result.get('ml_score', 50),
                'recommendation': ml_result.get('recommendation', 'No recommendation available')
            }
        else:
            components['ml'] = {'error': 'ML model not trained'}
    except Exception as e:
        components['ml'] = {'error': str(e)}
    
    # News sentiment
    try:
        sentiment_result = get_news_sentiment_score(symbol)
        components['sentiment'] = {
            'score': sentiment_result.get('sentiment_score', 50),
            'label': sentiment_result.get('sentiment_label', 'neutral'),
            'confidence': sentiment_result.get('confidence', 0),
            'article_count': sentiment_result.get('article_count', 0),
            'summary': sentiment_result.get('summary', 'No news analysis available')
        }
    except Exception as e:
        components['sentiment'] = {'error': str(e)}
    
    return components

def determine_trend(indicators):
    """Determine overall price trend from technical indicators"""
    current_price = indicators.get('Current_Price', 0)
    ma_20 = indicators.get('MA_20', 0)
    ma_50 = indicators.get('MA_50', 0)
    
    if current_price > ma_20 > ma_50:
        return "Strong Uptrend"
    elif current_price > ma_20:
        return "Uptrend"
    elif current_price < ma_20 < ma_50:
        return "Downtrend"
    elif current_price < ma_20:
        return "Weak Downtrend"
    else:
        return "Sideways"

def extract_key_fundamentals(fund_data):
    """Extract key fundamental metrics for narrative"""
    return {
        'pe_ratio': fund_data.get('PE Ratio', 0),
        'roe': fund_data.get('ROE', 0),
        'debt_equity': fund_data.get('Debt to Equity', 0),
        'current_ratio': fund_data.get('Current Ratio', 0),
        'profit_margin': fund_data.get('Profit Margin', 0),
        'market_cap': fund_data.get('Market Cap', 0)
    }

def generate_ai_narrative(symbol, components, period):
    """Generate AI-powered investment narrative using OpenAI"""
    try:
        openai = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare comprehensive analysis data for AI
        analysis_summary = prepare_analysis_for_ai(components)
        
        prompt = f"""
        You are an expert financial analyst creating a comprehensive investment narrative for {symbol}.
        
        Analysis Data:
        {analysis_summary}
        
        Create a professional, well-structured investment narrative with the following sections:
        
        1. Executive Summary (2-3 sentences)
        2. Technical Analysis Insights (3-4 sentences)
        3. Fundamental Analysis Review (3-4 sentences)
        4. Market Sentiment & News Impact (2-3 sentences)
        5. Risk Assessment (2-3 sentences)
        6. Investment Recommendation (3-4 sentences with specific action items)
        7. Key Price Levels & Targets (2-3 sentences)
        
        Make it professional, actionable, and specific to the data provided. 
        Use financial terminology appropriately and provide clear reasoning for recommendations.
        Include specific numbers and percentages from the analysis.
        
        Return the response in JSON format with each section as a separate field.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior financial analyst with expertise in Indian stock markets. "
                    + "Provide professional, data-driven investment narratives based on comprehensive analysis."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        narrative_json = json.loads(content) if content else {}
        return narrative_json
    
    except Exception as e:
        # Fallback to structured narrative if AI fails
        return generate_structured_narrative(symbol, components, period)

def prepare_analysis_for_ai(components):
    """Prepare analysis data in readable format for AI processing"""
    summary = []
    
    # Basic info
    basic = components.get('basic_info', {})
    summary.append(f"Current Price: ₹{basic.get('current_price', 0):.2f}")
    summary.append(f"Price Change: {basic.get('price_change_pct', 0):.2f}%")
    
    # Technical analysis
    tech = components.get('technical', {})
    if 'score' in tech:
        summary.append(f"Technical Score: {tech['score']}/100 ({tech['signal']})")
        summary.append(f"RSI: {tech.get('rsi', 50):.1f}")
        summary.append(f"Trend: {tech.get('trend', 'Unknown')}")
    
    # Fundamental analysis
    fund = components.get('fundamental', {})
    if 'score' in fund:
        summary.append(f"Fundamental Score: {fund['score']}/100 ({fund['rating']})")
        key_metrics = fund.get('key_metrics', {})
        summary.append(f"PE Ratio: {key_metrics.get('pe_ratio', 0):.2f}")
        summary.append(f"ROE: {key_metrics.get('roe', 0)*100:.1f}%")
    
    # ML analysis
    ml = components.get('ml', {})
    if 'signal' in ml:
        summary.append(f"ML Signal: {ml['signal']} (Confidence: {ml['confidence']:.1%})")
    
    # Sentiment analysis
    sentiment = components.get('sentiment', {})
    if 'score' in sentiment:
        summary.append(f"News Sentiment: {sentiment['label']} ({sentiment['score']}/100)")
        summary.append(f"Articles Analyzed: {sentiment['article_count']}")
    
    return "\n".join(summary)

def generate_structured_narrative(symbol, components, period):
    """Generate structured narrative without AI (fallback method)"""
    narrative = {}
    
    # Executive Summary
    basic = components.get('basic_info', {})
    tech = components.get('technical', {})
    fund = components.get('fundamental', {})
    
    price_direction = "higher" if basic.get('price_change_pct', 0) > 0 else "lower"
    
    narrative['executive_summary'] = (
        f"{symbol} is trading at ₹{basic.get('current_price', 0):.2f}, "
        f"{abs(basic.get('price_change_pct', 0)):.2f}% {price_direction} from the previous session. "
        f"Our comprehensive analysis indicates a {tech.get('signal', 'Hold').lower()} signal "
        f"based on technical indicators and fundamental metrics."
    )
    
    # Technical Analysis
    narrative['technical_insights'] = (
        f"Technical analysis shows a score of {tech.get('score', 0)}/100, "
        f"indicating a {tech.get('signal', 'neutral')} bias. "
        f"The RSI reading of {tech.get('rsi', 50):.1f} suggests "
        f"{'overbought' if tech.get('rsi', 50) > 70 else 'oversold' if tech.get('rsi', 50) < 30 else 'neutral'} conditions. "
        f"Current trend analysis indicates {tech.get('trend', 'sideways')} price movement."
    )
    
    # Fundamental Analysis
    key_metrics = fund.get('key_metrics', {})
    narrative['fundamental_review'] = (
        f"Fundamental analysis yields a score of {fund.get('score', 0)}/100, "
        f"rated as {fund.get('rating', 'Average')}. "
        f"Key metrics include PE ratio of {key_metrics.get('pe_ratio', 0):.2f} "
        f"and ROE of {key_metrics.get('roe', 0)*100:.1f}%. "
        f"The company's financial health appears {'strong' if fund.get('score', 0) > 60 else 'moderate' if fund.get('score', 0) > 40 else 'weak'}."
    )
    
    # Market Sentiment
    sentiment = components.get('sentiment', {})
    narrative['market_sentiment'] = (
        f"News sentiment analysis of {sentiment.get('article_count', 0)} recent articles "
        f"indicates {sentiment.get('label', 'neutral')} market mood with a score of {sentiment.get('score', 50)}/100. "
        f"This sentiment analysis provides {'strong' if sentiment.get('confidence', 0) > 0.7 else 'moderate' if sentiment.get('confidence', 0) > 0.4 else 'limited'} confidence in market direction."
    )
    
    # Risk Assessment
    overall_score = (tech.get('score', 0) + fund.get('score', 0)) / 2
    narrative['risk_assessment'] = (
        f"Overall investment risk is assessed as {'LOW' if overall_score > 65 else 'MEDIUM' if overall_score > 45 else 'HIGH'} "
        f"based on combined technical and fundamental analysis. "
        f"Volatility levels and market conditions suggest {'conservative' if overall_score < 45 else 'moderate' if overall_score < 65 else 'aggressive'} position sizing is appropriate."
    )
    
    # Investment Recommendation
    ml = components.get('ml', {})
    combined_signal = determine_combined_signal(tech, fund, ml, sentiment)
    
    narrative['investment_recommendation'] = (
        f"Based on our comprehensive analysis, we recommend {combined_signal['action']} for {symbol}. "
        f"The combined analysis confidence is {combined_signal['confidence']:.1%}. "
        f"Investors should {combined_signal['specific_action']} and monitor key technical levels closely."
    )
    
    # Price Levels
    current_price = basic.get('current_price', 0)
    support_level = current_price * 0.95  # 5% below current
    resistance_level = current_price * 1.05  # 5% above current
    
    narrative['price_targets'] = (
        f"Key support level identified at ₹{support_level:.2f} with resistance at ₹{resistance_level:.2f}. "
        f"A break above resistance could target ₹{resistance_level * 1.03:.2f}, "
        f"while a break below support may lead to ₹{support_level * 0.97:.2f}."
    )
    
    return narrative

def determine_combined_signal(tech, fund, ml, sentiment):
    """Determine combined investment signal from all components"""
    scores = []
    
    # Weight the scores
    if 'score' in tech:
        scores.append(tech['score'] * 0.3)  # 30% weight
    if 'score' in fund:
        scores.append(fund['score'] * 0.3)  # 30% weight
    if 'ml_score' in ml:
        scores.append(ml['ml_score'] * 0.25)  # 25% weight
    if 'score' in sentiment:
        scores.append(sentiment['score'] * 0.15)  # 15% weight
    
    combined_score = sum(scores) / len(scores) if scores else 50
    confidence = min(len(scores) / 4, 1.0)  # Higher confidence with more components
    
    if combined_score >= 70:
        return {
            'action': 'BUY',
            'confidence': confidence,
            'specific_action': 'consider accumulating on dips'
        }
    elif combined_score >= 55:
        return {
            'action': 'HOLD/WATCH',
            'confidence': confidence,
            'specific_action': 'maintain current position and watch for breakout'
        }
    elif combined_score <= 35:
        return {
            'action': 'SELL/AVOID',
            'confidence': confidence,
            'specific_action': 'consider reducing exposure or avoiding new positions'
        }
    else:
        return {
            'action': 'HOLD',
            'confidence': confidence,
            'specific_action': 'maintain cautious stance with tight stop-losses'
        }

def format_narrative_for_display(narrative_data):
    """Format narrative data for Streamlit display"""
    if not narrative_data.get('success'):
        return f"Error: {narrative_data.get('error', 'Unknown error')}"
    
    narrative = narrative_data['narrative']
    formatted_sections = []
    
    section_titles = {
        'executive_summary': '📊 Executive Summary',
        'technical_insights': '📈 Technical Analysis Insights',
        'fundamental_review': '💼 Fundamental Analysis Review',
        'market_sentiment': '📰 Market Sentiment & News Impact',
        'risk_assessment': '⚠️ Risk Assessment',
        'investment_recommendation': '💡 Investment Recommendation',
        'price_targets': '🎯 Key Price Levels & Targets'
    }
    
    for key, title in section_titles.items():
        if key in narrative:
            formatted_sections.append(f"### {title}\n{narrative[key]}\n")
    
    return "\n".join(formatted_sections)

def save_narrative_to_session(symbol, narrative_data):
    """Save generated narrative to session state for later access"""
    if 'investment_narratives' not in st.session_state:
        st.session_state.investment_narratives = {}
    
    st.session_state.investment_narratives[symbol] = narrative_data

def get_saved_narrative(symbol):
    """Retrieve saved narrative from session state"""
    if 'investment_narratives' not in st.session_state:
        return None
    
    return st.session_state.investment_narratives.get(symbol)