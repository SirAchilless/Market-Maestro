import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import utility functions
from utils.data_fetcher import get_stock_data, get_stock_info
from utils.stock_lists import search_stock, get_stock_display_name, NIFTY_50
from utils.investment_narrative import (
    generate_investment_narrative, 
    format_narrative_for_display,
    save_narrative_to_session,
    get_saved_narrative
)

def create_download_content(narrative_data, symbol):
    """Create downloadable text content from narrative data"""
    content = []
    content.append(f"INVESTMENT NARRATIVE REPORT")
    content.append(f"Stock Symbol: {symbol}")
    content.append(f"Generated: {narrative_data.get('generated_at', 'Unknown')}")
    content.append("=" * 50)
    content.append("")
    
    narrative = narrative_data.get('narrative', {})
    
    sections = {
        'executive_summary': 'EXECUTIVE SUMMARY',
        'technical_insights': 'TECHNICAL ANALYSIS',
        'fundamental_review': 'FUNDAMENTAL ANALYSIS',
        'market_sentiment': 'MARKET SENTIMENT',
        'risk_assessment': 'RISK ASSESSMENT',
        'investment_recommendation': 'INVESTMENT RECOMMENDATION',
        'price_targets': 'PRICE TARGETS'
    }
    
    for key, title in sections.items():
        if key in narrative:
            content.append(f"{title}")
            content.append("-" * len(title))
            content.append(narrative[key])
            content.append("")
    
    content.append("ANALYSIS COMPONENTS SUMMARY")
    content.append("-" * 30)
    
    components = narrative_data.get('components', {})
    if 'technical' in components:
        tech = components['technical']
        content.append(f"Technical Score: {tech.get('score', 0)}/100 ({tech.get('signal', 'Hold')})")
    
    if 'fundamental' in components:
        fund = components['fundamental']
        content.append(f"Fundamental Score: {fund.get('score', 0)}/100 ({fund.get('rating', 'Average')})")
    
    if 'ml' in components:
        ml = components['ml']
        content.append(f"ML Signal: {ml.get('signal', 'Hold')} ({ml.get('confidence', 0):.1%} confidence)")
    
    if 'sentiment' in components:
        sentiment = components['sentiment']
        content.append(f"News Sentiment: {sentiment.get('score', 50)}/100 ({sentiment.get('label', 'neutral')})")
    
    content.append("")
    content.append("DISCLAIMER")
    content.append("-" * 10)
    content.append("This report is for informational purposes only. Investment decisions should be made based on your own research and risk tolerance.")
    
    return "\n".join(content)

st.set_page_config(page_title="Investment Narrative", page_icon="📝", layout="wide")

st.title("📝 AI-Powered Investment Narrative Generator")
st.markdown("Generate comprehensive, intelligent investment narratives using advanced AI analysis")

# Sidebar for stock selection
st.sidebar.header("🎯 Stock Selection")

# Stock search and selection
search_query = st.sidebar.text_input(
    "Search Stock by Symbol or Company Name", 
    placeholder="e.g., RELIANCE, TCS, Infosys",
    help="Search from 500+ Indian stocks"
)

if search_query:
    matching_stocks = search_stock(search_query)
    if matching_stocks:
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
    # Default to NIFTY 50 stocks with names
    stock_options = [f"{stock} - {get_stock_display_name(stock)}" for stock in NIFTY_50[:20]]
    selected_option = st.sidebar.selectbox("Select from NIFTY 50", options=stock_options)
    selected_stock = selected_option.split(" - ")[0] if selected_option else None

# Analysis period selection
analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=2,
    help="Historical data period for analysis"
)

# Narrative settings
st.sidebar.subheader("📋 Narrative Settings")

narrative_style = st.sidebar.selectbox(
    "Narrative Style",
    options=["Professional", "Detailed Technical", "Conservative", "Aggressive"],
    help="Choose the tone and focus of the narrative"
)

include_price_targets = st.sidebar.checkbox("Include Price Targets", value=True)
include_risk_analysis = st.sidebar.checkbox("Include Risk Analysis", value=True)
include_sector_context = st.sidebar.checkbox("Include Sector Context", value=True)

# Check for OpenAI API key
has_openai_key = st.session_state.get('OPENAI_API_KEY')
if not has_openai_key:
    # Try to get from secrets if they exist
    try:
        has_openai_key = st.secrets.get('OPENAI_API_KEY')
    except:
        # Secrets file doesn't exist, which is fine
        pass

if not has_openai_key:
    st.info("💡 For enhanced AI-powered narratives, please provide your OpenAI API key in the secrets.")

# Main content area
if selected_stock:
    # Stock header information
    st.markdown(f"## 📊 Investment Narrative for {selected_stock}")
    
    # Fetch stock data
    with st.spinner(f"Fetching data for {selected_stock}..."):
        stock_data = get_stock_data(selected_stock, period=analysis_period)
        stock_info = get_stock_info(selected_stock)
    
    if not stock_data.empty:
        # Display basic stock info
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = stock_data['Close'].iloc[-1]
        previous_price = stock_data['Close'].iloc[-2] if len(stock_data) >= 2 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
        
        with col1:
            st.metric(
                "Current Price", 
                f"₹{current_price:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            company_name = get_stock_display_name(selected_stock)
            sector = stock_info.get('sector', 'N/A')
            st.metric("Company", company_name)
            st.caption(f"Sector: {sector}")
        
        with col3:
            market_cap = stock_info.get('market_cap', 0)
            if market_cap > 0:
                market_cap_cr = market_cap / 10000000  # Convert to crores
                st.metric("Market Cap", f"₹{market_cap_cr:,.0f} Cr")
            else:
                st.metric("Market Cap", "N/A")
        
        with col4:
            volume = stock_data['Volume'].iloc[-1]
            avg_volume = stock_data['Volume'].mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
        
        # Generate narrative button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Generate Investment Narrative", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive investment narrative..."):
                    # Generate the narrative
                    narrative_result = generate_investment_narrative(
                        selected_stock, 
                        stock_data, 
                        period=analysis_period
                    )
                    
                    # Save to session state
                    save_narrative_to_session(selected_stock, narrative_result)
                    
                    st.success("✅ Investment narrative generated successfully!")
                    st.rerun()
        
        # Display saved narrative if available
        saved_narrative = get_saved_narrative(selected_stock)
        
        if saved_narrative and saved_narrative.get('success'):
            st.markdown("---")
            st.markdown("### 📖 Investment Narrative Report")
            
            # Display generation timestamp
            generation_time = saved_narrative.get('generated_at', 'Unknown')
            st.caption(f"Generated on: {generation_time}")
            
            # Display the formatted narrative
            narrative_content = format_narrative_for_display(saved_narrative)
            st.markdown(narrative_content)
            
            # Additional analysis components
            if 'components' in saved_narrative:
                st.markdown("---")
                st.markdown("### 📊 Analysis Components Summary")
                
                components = saved_narrative['components']
                
                # Create summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    tech_score = components.get('technical', {}).get('score', 0)
                    tech_signal = components.get('technical', {}).get('signal', 'Hold')
                    st.metric("Technical Score", f"{tech_score:.0f}/100", delta=tech_signal)
                
                with col2:
                    fund_score = components.get('fundamental', {}).get('score', 0)
                    fund_rating = components.get('fundamental', {}).get('rating', 'Average')
                    st.metric("Fundamental Score", f"{fund_score:.0f}/100", delta=fund_rating)
                
                with col3:
                    ml_signal = components.get('ml', {}).get('signal', 'Hold')
                    ml_confidence = components.get('ml', {}).get('confidence', 0)
                    st.metric("ML Signal", ml_signal, delta=f"{ml_confidence:.1%} confidence")
                
                with col4:
                    sentiment_score = components.get('sentiment', {}).get('score', 50)
                    sentiment_label = components.get('sentiment', {}).get('label', 'neutral')
                    st.metric("News Sentiment", f"{sentiment_score:.0f}/100", delta=sentiment_label.title())
                
                # Detailed component analysis
                with st.expander("🔍 Detailed Component Analysis"):
                    
                    # Technical analysis details
                    st.markdown("#### 📈 Technical Analysis Details")
                    tech_comp = components.get('technical', {})
                    if 'indicators' in tech_comp:
                        indicators = tech_comp['indicators']
                        
                        tech_col1, tech_col2 = st.columns(2)
                        with tech_col1:
                            st.write(f"**RSI:** {indicators.get('RSI', 0):.1f}")
                            st.write(f"**MACD:** {indicators.get('MACD', 0):.4f}")
                            st.write(f"**Trend:** {tech_comp.get('trend', 'Unknown')}")
                        
                        with tech_col2:
                            st.write(f"**MA 20:** ₹{indicators.get('MA_20', 0):.2f}")
                            st.write(f"**MA 50:** ₹{indicators.get('MA_50', 0):.2f}")
                            st.write(f"**Volume Ratio:** {indicators.get('Volume_Ratio', 1):.2f}x")
                    
                    # Fundamental analysis details
                    st.markdown("#### 💼 Fundamental Analysis Details")
                    fund_comp = components.get('fundamental', {})
                    if 'key_metrics' in fund_comp:
                        metrics = fund_comp['key_metrics']
                        
                        fund_col1, fund_col2 = st.columns(2)
                        with fund_col1:
                            pe_ratio = metrics.get('pe_ratio', 0)
                            roe = metrics.get('roe', 0) * 100
                            st.write(f"**PE Ratio:** {pe_ratio:.2f}")
                            st.write(f"**ROE:** {roe:.1f}%")
                        
                        with fund_col2:
                            debt_equity = metrics.get('debt_equity', 0)
                            current_ratio = metrics.get('current_ratio', 0)
                            st.write(f"**Debt/Equity:** {debt_equity:.2f}")
                            st.write(f"**Current Ratio:** {current_ratio:.2f}")
                
                # Download narrative button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📄 Download Narrative Report", use_container_width=True):
                        # Create downloadable content
                        download_content = create_download_content(saved_narrative, selected_stock)
                        st.download_button(
                            label="💾 Download as Text File",
                            data=download_content,
                            file_name=f"{selected_stock}_investment_narrative_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            key="download_narrative"
                        )
        
        elif saved_narrative and not saved_narrative.get('success'):
            st.error(f"Error generating narrative: {saved_narrative.get('error')}")
    
    else:
        st.error(f"Unable to fetch data for {selected_stock}. Please try a different stock symbol.")

else:
    # Welcome message when no stock selected
    st.markdown("""
    ### 🎯 Welcome to AI-Powered Investment Narrative Generator
    
    **What This Tool Does:**
    - Generates comprehensive investment narratives using advanced AI analysis
    - Combines technical, fundamental, ML, and sentiment analysis
    - Provides professional-grade investment reports
    - Offers actionable insights and recommendations
    
    **How to Use:**
    1. **Select a Stock** from the sidebar (search or choose from NIFTY 50)
    2. **Choose Analysis Period** (1 month to 2 years)
    3. **Configure Narrative Settings** for personalized reports
    4. **Generate Narrative** with one click
    5. **Review and Download** your comprehensive investment report
    
    **Key Features:**
    - 🤖 **AI-Powered Analysis** using GPT-4 for intelligent narratives
    - 📊 **Multi-Component Analysis** combining 4 different analytical approaches
    - 📈 **Professional Reports** suitable for investment presentations
    - 🎯 **Actionable Insights** with specific buy/sell recommendations
    - 💾 **Downloadable Reports** for offline analysis and sharing
    
    **Analysis Components:**
    - **Technical Analysis:** RSI, MACD, Moving Averages, Bollinger Bands
    - **Fundamental Analysis:** PE ratio, ROE, Debt/Equity, Financial Health
    - **ML Predictions:** AI-powered signal prediction with confidence scores
    - **News Sentiment:** Real-time sentiment analysis from financial news
    
    Select a stock from the sidebar to get started!
    """)

# Footer information
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool provides analytical insights for educational and informational purposes only. 
Investment decisions should always be made based on your own research and risk tolerance. 
Past performance does not guarantee future results.
""")