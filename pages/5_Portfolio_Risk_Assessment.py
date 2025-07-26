"""
Portfolio Risk Assessment Page for Indian Stock Market Screener
One-click comprehensive portfolio risk analysis with actionable insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from utils.portfolio_risk_analyzer import PortfolioRiskAnalyzer
from utils.stock_lists import get_all_stocks

# Initialize session state
if 'portfolio_for_analysis' not in st.session_state:
    st.session_state.portfolio_for_analysis = {}

if 'risk_analysis_results' not in st.session_state:
    st.session_state.risk_analysis_results = None

# Page configuration
st.set_page_config(page_title="Portfolio Risk Assessment", page_icon="🎯", layout="wide")

st.title("🎯 One-Click Portfolio Risk Assessment")
st.markdown("Comprehensive risk analysis of your stock portfolio with AI-powered insights")

# Initialize the risk analyzer
risk_analyzer = PortfolioRiskAnalyzer()

# Portfolio Input Section
st.subheader("📊 Portfolio Setup")

input_method = st.radio(
    "Choose Portfolio Input Method:",
    ["Manual Entry", "Upload CSV", "Use Current Watchlist", "Sample Portfolio"],
    horizontal=True,
    help="Select how you want to input your portfolio holdings"
)

# Initialize portfolio data from session state or empty
portfolio_data = st.session_state.get('current_portfolio_data', {})

# Check if portfolio was passed from main dashboard
if hasattr(st.session_state, 'portfolio_for_analysis') and st.session_state.portfolio_for_analysis:
    portfolio_data = st.session_state.portfolio_for_analysis.copy()
    st.session_state.current_portfolio_data = portfolio_data
    st.success(f"✅ Loaded portfolio from watchlist with {len(portfolio_data)} stocks")
    
    # Clear the session state to avoid reloading
    del st.session_state.portfolio_for_analysis

if input_method == "Manual Entry":
    st.markdown("### 📝 Enter Your Holdings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_stocks = st.number_input("Number of stocks in portfolio", min_value=1, max_value=50, value=5)
    
    with col2:
        portfolio_value = st.number_input("Total Portfolio Value (₹)", min_value=10000, value=1000000, step=10000)
    
    st.markdown("#### Stock Holdings:")
    
    for i in range(num_stocks):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            stock_symbol = st.text_input(
                f"Stock {i+1} Symbol", 
                key=f"stock_{i}",
                placeholder="e.g., RELIANCE, TCS, INFY",
                help="Enter stock symbol without .NS suffix"
            )
        
        with col2:
            weight_percent = st.number_input(
                f"Weight %", 
                min_value=0.1, 
                max_value=100.0, 
                value=20.0,
                step=0.1,
                key=f"weight_{i}"
            )
        
        with col3:
            investment_amount = (weight_percent / 100) * portfolio_value
            st.metric("Amount ₹", f"{investment_amount:,.0f}")
        
        if stock_symbol:
            portfolio_data[stock_symbol.upper()] = weight_percent / 100
            st.session_state.current_portfolio_data = portfolio_data

elif input_method == "Upload CSV":
    st.markdown("### 📁 Upload Portfolio CSV")
    
    st.info("CSV format: Stock Symbol, Weight (%) or Amount (₹)")
    
    uploaded_file = st.file_uploader(
        "Upload your portfolio CSV file",
        type=['csv'],
        help="Columns: Symbol, Weight or Symbol, Amount"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Process the CSV data
            if 'Symbol' in df.columns:
                if 'Weight' in df.columns:
                    for _, row in df.iterrows():
                        try:
                            if not pd.isna(row['Symbol']) and not pd.isna(row['Weight']):
                                portfolio_data[str(row['Symbol']).upper()] = float(row['Weight']) / 100
                                st.session_state.current_portfolio_data = portfolio_data
                        except (ValueError, TypeError):
                            continue
                elif 'Amount' in df.columns:
                    total_amount = df['Amount'].sum()
                    for _, row in df.iterrows():
                        try:
                            if not pd.isna(row['Symbol']) and not pd.isna(row['Amount']):
                                portfolio_data[str(row['Symbol']).upper()] = float(row['Amount']) / total_amount
                                st.session_state.current_portfolio_data = portfolio_data
                        except (ValueError, TypeError):
                            continue
            
            if portfolio_data:
                st.success(f"Successfully loaded {len(portfolio_data)} stocks from CSV")
            else:
                st.error("Could not parse portfolio data. Please check CSV format.")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

elif input_method == "Use Current Watchlist":
    st.markdown("### 📋 Import from Watchlist")
    
    if hasattr(st.session_state, 'watchlist') and st.session_state.watchlist:
        watchlist = st.session_state.watchlist
        st.write(f"Found {len(watchlist)} stocks in your watchlist:")
        
        # Display watchlist stocks
        for stock in watchlist:
            st.write(f"• {stock}")
        
        # Equal weight allocation by default
        if st.button("Import with Equal Weights"):
            equal_weight = 1.0 / len(watchlist)
            for stock in watchlist:
                portfolio_data[stock] = equal_weight
            st.session_state.current_portfolio_data = portfolio_data
            st.success(f"Imported {len(watchlist)} stocks with equal weights")
            st.rerun()
    else:
        st.warning("No stocks found in your watchlist. Please add stocks to your watchlist first.")

elif input_method == "Sample Portfolio":
    st.markdown("### 🎯 Sample Indian Portfolio")
    
    sample_portfolios = {
        "Conservative NIFTY 50": {
            "RELIANCE": 0.15, "TCS": 0.12, "HDFCBANK": 0.10, "ICICIBANK": 0.08,
            "HINDUNILVR": 0.07, "ITC": 0.06, "SBIN": 0.06, "BHARTIARTL": 0.05,
            "KOTAKBANK": 0.05, "LT": 0.04, "ASIANPAINT": 0.04, "MARUTI": 0.04,
            "AXISBANK": 0.03, "NESTLEIND": 0.03, "POWERGRID": 0.03, "NTPC": 0.03,
            "ULTRACEMCO": 0.02
        },
        "Growth Tech Portfolio": {
            "TCS": 0.20, "INFY": 0.15, "WIPRO": 0.10, "HCLTECH": 0.10,
            "TECHM": 0.08, "LTIM": 0.06, "MPHASIS": 0.05, "MINDTREE": 0.05,
            "PERSISTENT": 0.04, "COFORGE": 0.04, "BHARTIARTL": 0.08, "JIOFINANCE": 0.05
        },
        "Banking & Finance": {
            "HDFCBANK": 0.20, "ICICIBANK": 0.18, "KOTAKBANK": 0.12, "SBIN": 0.10,
            "AXISBANK": 0.10, "BAJFINANCE": 0.08, "INDUSINDBK": 0.07, "HDFCLIFE": 0.05,
            "ICICIGI": 0.05, "BAJAJFINSV": 0.05
        },
        "Pharma & Healthcare": {
            "SUNPHARMA": 0.20, "DRREDDY": 0.15, "CIPLA": 0.12, "DIVISLAB": 0.12,
            "BIOCON": 0.10, "AUROPHARMA": 0.08, "LUPIN": 0.08, "APOLLOHOSP": 0.08,
            "FORTIS": 0.04, "GLENMARK": 0.03
        }
    }
    
    selected_portfolio = st.selectbox(
        "Choose a sample portfolio:",
        list(sample_portfolios.keys()),
        help="Pre-configured portfolios for different investment strategies"
    )
    
    if st.button("Load Sample Portfolio"):
        portfolio_data = sample_portfolios[selected_portfolio]
        st.session_state.current_portfolio_data = portfolio_data
        st.success(f"Loaded {selected_portfolio} with {len(portfolio_data)} stocks")
        st.rerun()

# Display current portfolio
if portfolio_data:
    st.markdown("### 📈 Current Portfolio for Analysis")
    
    # Convert to DataFrame for display
    portfolio_df = pd.DataFrame([
        {"Stock": symbol, "Weight %": f"{weight*100:.1f}%"} 
        for symbol, weight in portfolio_data.items()
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(portfolio_df, use_container_width=True)
    
    with col2:
        total_weight = sum(portfolio_data.values())
        st.metric("Total Weight", f"{total_weight*100:.1f}%")
        st.metric("Number of Holdings", len(portfolio_data))
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning("Portfolio weights don't sum to 100%")
        
        # Clear portfolio button
        if st.button("Clear Portfolio", type="secondary"):
            st.session_state.current_portfolio_data = {}
            st.rerun()

# Risk Analysis Section
st.markdown("---")
st.subheader("🔍 Risk Analysis")

# Initialize variables with default values
show_detailed_charts = True
analysis_period = "1y"
include_stress_test = True

if portfolio_data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1y", "6mo", "3mo", "2y"],
            index=0,
            help="Historical data period for analysis"
        )
    
    with col2:
        include_stress_test = st.checkbox(
            "Include Stress Testing",
            value=True,
            help="Perform stress testing scenarios"
        )
    
    with col3:
        show_detailed_charts = st.checkbox(
            "Show Detailed Charts",
            value=True,
            help="Display comprehensive visualization"
        )
    
    # One-Click Analysis Button
    if st.button("🎯 Analyze Portfolio Risk", type="primary", use_container_width=True):
        with st.spinner("Performing comprehensive risk analysis..."):
            
            # Ensure stock symbols have .NS suffix for Indian stocks
            formatted_portfolio = {}
            for symbol, weight in portfolio_data.items():
                # Add .NS suffix if not already present
                if not symbol.endswith('.NS'):
                    formatted_symbol = symbol + '.NS'
                else:
                    formatted_symbol = symbol
                formatted_portfolio[formatted_symbol] = weight
            
            # Perform risk analysis
            risk_results = risk_analyzer.analyze_portfolio_risk(
                formatted_portfolio, 
                period=analysis_period
            )
            
            # Store results in session state
            st.session_state.risk_analysis_results = risk_results
            
            if risk_results and risk_results.get('portfolio_metrics'):
                st.success("✅ Risk analysis completed successfully!")
            else:
                st.error("❌ Risk analysis failed. Please check your stock symbols.")

# Display Results
if st.session_state.risk_analysis_results:
    results = st.session_state.risk_analysis_results
    
    # Risk Summary Dashboard
    st.markdown("---")
    st.subheader("📊 Risk Analysis Dashboard")
    
    # Key Metrics Row
    portfolio_metrics = results.get('portfolio_metrics', {})
    risk_metrics = results.get('risk_metrics', {})
    diversification = results.get('diversification', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        annual_return = portfolio_metrics.get('annual_return', 0) * 100
        st.metric(
            "Annual Return", 
            f"{annual_return:.1f}%",
            delta=f"{annual_return - 12:.1f}% vs Market" if annual_return != 0 else None
        )
    
    with col2:
        annual_vol = risk_metrics.get('annual_volatility', 0) * 100
        st.metric(
            "Volatility", 
            f"{annual_vol:.1f}%",
            delta=f"{annual_vol - 18:.1f}% vs Market" if annual_vol != 0 else None
        )
    
    with col3:
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        st.metric(
            "Sharpe Ratio", 
            f"{sharpe_ratio:.2f}",
            delta="Good" if sharpe_ratio > 1 else "Moderate" if sharpe_ratio > 0.5 else "Poor"
        )
    
    with col4:
        risk_grade = risk_metrics.get('risk_grade', 'Unknown')
        st.metric("Risk Grade", risk_grade)
    
    with col5:
        div_score = diversification.get('diversification_score', 0)
        st.metric(
            "Diversification", 
            f"{div_score:.0f}/100",
            delta=diversification.get('diversification_grade', 'Unknown')
        )
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance", "⚠️ Risk Metrics", "🎯 Diversification", 
        "🏭 Sector Analysis", "💡 Recommendations"
    ])
    
    with tab1:
        st.markdown("### Portfolio Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Performance Metrics")
            metrics_data = [
                ["Annual Return", f"{portfolio_metrics.get('annual_return', 0)*100:.2f}%"],
                ["Cumulative Return", f"{portfolio_metrics.get('cumulative_return', 0)*100:.2f}%"],
                ["Sharpe Ratio", f"{portfolio_metrics.get('sharpe_ratio', 0):.3f}"],
                ["Maximum Drawdown", f"{portfolio_metrics.get('max_drawdown', 0)*100:.2f}%"],
                ["Analysis Period", f"{portfolio_metrics.get('total_days', 0)} days"]
            ]
            
            metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
            st.table(metrics_df)
        
        with col2:
            st.markdown("#### 🎯 Risk-Adjusted Performance")
            
            # Performance grade
            sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            if sharpe > 1.5:
                perf_grade = "Excellent"
                grade_color = "green"
            elif sharpe > 1.0:
                perf_grade = "Good"
                grade_color = "blue"
            elif sharpe > 0.5:
                perf_grade = "Moderate"
                grade_color = "orange"
            else:
                perf_grade = "Poor"
                grade_color = "red"
            
            st.markdown(f"**Performance Grade:** :{grade_color}[{perf_grade}]")
            
            # Comparison with market
            portfolio_return = portfolio_metrics.get('annual_return', 0) * 100
            market_return = 12  # Approximate NIFTY return
            
            if portfolio_return > market_return:
                st.success(f"✅ Outperforming market by {portfolio_return - market_return:.1f}%")
            else:
                st.warning(f"⚠️ Underperforming market by {market_return - portfolio_return:.1f}%")
    
    with tab2:
        st.markdown("### Risk Analysis Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚠️ Risk Metrics")
            risk_data = [
                ["Daily Volatility", f"{risk_metrics.get('daily_volatility', 0)*100:.2f}%"],
                ["Annual Volatility", f"{risk_metrics.get('annual_volatility', 0)*100:.2f}%"],
                ["Downside Deviation", f"{risk_metrics.get('downside_deviation', 0)*100:.2f}%"],
                ["Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.3f}"],
                ["Beta (vs NIFTY)", f"{risk_metrics.get('beta', 1):.2f}"],
                ["Risk Grade", risk_metrics.get('risk_grade', 'Unknown')]
            ]
            
            risk_df = pd.DataFrame(risk_data, columns=["Metric", "Value"])
            st.table(risk_df)
        
        with col2:
            st.markdown("#### 📉 Value at Risk")
            var_analysis = results.get('var_analysis', {})
            
            var_data = [
                ["VaR 95% (Daily)", f"{var_analysis.get('var_95_daily', 0)*100:.2f}%"],
                ["VaR 99% (Daily)", f"{var_analysis.get('var_99_daily', 0)*100:.2f}%"],
                ["Expected Shortfall 95%", f"{var_analysis.get('expected_shortfall_95', 0)*100:.2f}%"],
                ["Expected Shortfall 99%", f"{var_analysis.get('expected_shortfall_99', 0)*100:.2f}%"]
            ]
            
            var_df = pd.DataFrame(var_data, columns=["Metric", "Value"])
            st.table(var_df)
            
            # Risk interpretation
            var_95 = abs(var_analysis.get('var_95_daily', 0) * 100)
            if var_95 > 5:
                st.error(f"⚠️ High risk: Potential daily loss of {var_95:.1f}% (95% confidence)")
            elif var_95 > 3:
                st.warning(f"⚠️ Moderate risk: Potential daily loss of {var_95:.1f}% (95% confidence)")
            else:
                st.success(f"✅ Low risk: Potential daily loss of {var_95:.1f}% (95% confidence)")
    
    with tab3:
        st.markdown("### Diversification Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Diversification Metrics")
            div_data = [
                ["Number of Holdings", diversification.get('num_holdings', 0)],
                ["Largest Holding", f"{diversification.get('largest_holding_weight', 0)*100:.1f}%"],
                ["Top 5 Concentration", f"{diversification.get('top_5_concentration', 0)*100:.1f}%"],
                ["HHI Index", f"{diversification.get('hhi_index', 0):.3f}"],
                ["Diversification Score", f"{diversification.get('diversification_score', 0):.0f}/100"],
                ["Grade", diversification.get('diversification_grade', 'Unknown')]
            ]
            
            div_df = pd.DataFrame(div_data, columns=["Metric", "Value"])
            st.table(div_df)
        
        with col2:
            st.markdown("#### 📊 Diversification Insights")
            
            # Diversification recommendations
            grade = diversification.get('diversification_grade', 'Unknown')
            if grade == "Excellent":
                st.success("✅ Excellent diversification - well-balanced portfolio")
            elif grade == "Good":
                st.info("✅ Good diversification - minor improvements possible")
            elif grade == "Moderate":
                st.warning("⚠️ Moderate diversification - consider adding more stocks")
            else:
                st.error("❌ Poor diversification - high concentration risk")
            
            # Concentration analysis
            largest = diversification.get('largest_holding_weight', 0) * 100
            if largest > 25:
                st.warning(f"⚠️ Concentration risk: Largest holding is {largest:.1f}%")
            
            num_holdings = diversification.get('num_holdings', 0)
            if num_holdings < 10:
                st.info(f"💡 Consider adding more stocks (current: {num_holdings})")
    
    with tab4:
        st.markdown("### Sector Concentration Analysis")
        
        sector_analysis = results.get('sector_analysis', {})
        sector_weights = sector_analysis.get('sector_weights', {})
        
        if sector_weights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏭 Sector Allocation")
                
                sector_data = []
                for sector, weight in sector_weights.items():
                    sector_data.append([sector, f"{weight*100:.1f}%"])
                
                sector_df = pd.DataFrame(sector_data, columns=["Sector", "Weight"])
                st.table(sector_df)
            
            with col2:
                st.markdown("#### 📊 Sector Insights")
                
                # Sector concentration metrics
                st.write(f"**Number of Sectors:** {sector_analysis.get('num_sectors', 0)}")
                st.write(f"**Largest Sector:** {sector_analysis.get('largest_sector_weight', 0)*100:.1f}%")
                st.write(f"**Concentration Grade:** {sector_analysis.get('concentration_grade', 'Unknown')}")
                
                # Create sector pie chart if charts are enabled
                if show_detailed_charts and len(sector_weights) > 0:
                    charts = risk_analyzer.create_risk_dashboard_charts(results)
                    
                    # Display sector allocation chart
                    for chart in charts:
                        if "Sector Allocation" in str(chart.layout.title.text):
                            st.plotly_chart(chart, use_container_width=True)
                            break
        else:
            st.warning("Unable to analyze sector concentration - stock sector mapping not available")
    
    with tab5:
        st.markdown("### AI-Powered Recommendations")
        
        recommendations = results.get('recommendations', [])
        
        if recommendations:
            st.markdown("#### 💡 Risk Management Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
            
            # Additional actionable insights
            st.markdown("---")
            st.markdown("#### 🎯 Action Items")
            
            # Generate specific action items based on analysis
            action_items = []
            
            # Diversification actions
            if diversification.get('num_holdings', 0) < 15:
                action_items.append("📈 Add 5-10 more stocks from different sectors to improve diversification")
            
            # Risk actions
            if risk_metrics.get('annual_volatility', 0) > 0.25:
                action_items.append("🛡️ Consider adding defensive stocks (utilities, consumer staples) to reduce volatility")
            
            # Concentration actions
            if diversification.get('largest_holding_weight', 0) > 0.20:
                action_items.append("⚖️ Reduce position size of largest holding to below 20%")
            
            # Performance actions
            if portfolio_metrics.get('sharpe_ratio', 0) < 0.5:
                action_items.append("🎯 Review stock selection - focus on quality companies with consistent returns")
            
            for item in action_items:
                st.markdown(f"• {item}")
        
        else:
            st.info("No specific recommendations available")
    
    # Advanced Charts Section
    if show_detailed_charts:
        st.markdown("---")
        st.subheader("📊 Advanced Risk Visualization")
        
        try:
            charts = risk_analyzer.create_risk_dashboard_charts(results)
            
            if charts:
                # Display charts in a 2x2 grid
                col1, col2 = st.columns(2)
                
                for i, chart in enumerate(charts):
                    with col1 if i % 2 == 0 else col2:
                        st.plotly_chart(chart, use_container_width=True, key=f"chart_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            else:
                st.info("Advanced charts not available for this portfolio")
                
        except Exception as e:
            st.warning(f"Error generating advanced charts: {e}")
    
    # Comprehensive Export Section
    st.markdown("---")
    st.subheader("📥 Comprehensive Export")
    
    # Store analysis results in session state for export
    export_data = {
        'Portfolio Analysis': {
            'Portfolio Metrics': portfolio_metrics,
            'Risk Metrics': risk_metrics,
            'Diversification': diversification,
            'Sector Analysis': results.get('sector_analysis', {}),
            'VaR Analysis': results.get('var_analysis', {}),
            'Recommendations': results.get('recommendations', [])
        }
    }
    
    # Store charts for export
    export_charts = {}
    if show_detailed_charts:
        try:
            charts = risk_analyzer.create_risk_dashboard_charts(results)
            if charts:
                for i, chart in enumerate(charts):
                    chart_title = str(chart.layout.title.text) if chart.layout.title and chart.layout.title.text else f"Portfolio_Chart_{i+1}"
                    # Clean up chart title
                    clean_title = chart_title.replace('<br>', ' ').replace('<b>', '').replace('</b>', '')
                    export_charts[clean_title] = chart
                    
                # Store charts in session state immediately
                st.session_state.export_charts = export_charts
                
        except Exception as e:
            st.warning(f"Charts available but export preparation had minor issues: {e}")
            pass
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced Comprehensive PDF Export
        if st.button("📄 Export Complete PDF Report", type="primary", help="Generate complete PDF with all data and charts as shown on screen"):
            try:
                # Try enhanced export first, fallback to simple export if needed
                try:
                    from utils.enhanced_pdf_export import EnhancedPDFExporter
                    export_type = "enhanced"
                except Exception:
                    from utils.simple_pdf_export import SimplePDFExporter as EnhancedPDFExporter
                    export_type = "simple"
                
                # Store current analysis data in session state for export with real values
                portfolio_analysis_data = {
                    'portfolio_metrics': {
                        'Annual Return': f"{portfolio_metrics.get('annual_return', 0)*100:.2f}%",
                        'Volatility': f"{portfolio_metrics.get('annual_volatility', 0)*100:.2f}%", 
                        'Sharpe Ratio': f"{portfolio_metrics.get('sharpe_ratio', 0):.3f}",
                        'Max Drawdown': f"{portfolio_metrics.get('max_drawdown', 0)*100:.2f}%",
                        'Beta': f"{portfolio_metrics.get('beta', 1):.2f}",
                        'Alpha': f"{portfolio_metrics.get('alpha', 0)*100:.2f}%"
                    },
                    'risk_metrics': {
                        'Risk Grade': risk_metrics.get('risk_grade', 'Unknown'),
                        'Daily Volatility': f"{risk_metrics.get('daily_volatility', 0)*100:.2f}%",
                        'VaR 95%': f"{results.get('var_analysis', {}).get('var_95_daily', 0)*100:.2f}%",
                        'VaR 99%': f"{results.get('var_analysis', {}).get('var_99_daily', 0)*100:.2f}%",
                        'Sortino Ratio': f"{risk_metrics.get('sortino_ratio', 0):.3f}",
                        'Downside Deviation': f"{risk_metrics.get('downside_deviation', 0)*100:.2f}%"
                    },
                    'diversification': {
                        'Diversification Score': f"{diversification.get('diversification_score', 0):.0f}/100",
                        'Number of Holdings': diversification.get('num_holdings', 0),
                        'Largest Holding': f"{diversification.get('largest_holding_weight', 0)*100:.1f}%",
                        'Top 5 Concentration': f"{diversification.get('top_5_concentration', 0)*100:.1f}%",
                        'HHI Index': f"{diversification.get('hhi_index', 0):.3f}",
                        'Diversification Grade': diversification.get('diversification_grade', 'Unknown')
                    },
                    'portfolio_holdings': portfolio_data
                }
                
                st.session_state.portfolio_analysis_results = portfolio_analysis_data
                st.session_state.risk_metrics = risk_metrics
                
                # Store charts in session state
                if export_charts:
                    st.session_state.export_charts = export_charts
                
                # Create enhanced exporter
                exporter = EnhancedPDFExporter()
                
                # Generate comprehensive PDF with all current data
                if export_type == "simple":
                    pdf_bytes = exporter.create_comprehensive_report(
                        portfolio_data=st.session_state.portfolio_analysis_results,
                        risk_data=risk_metrics,
                        charts=export_charts,
                        technical_data=None,
                        watchlist_data=None,
                        ml_signals=None
                    )
                else:
                    pdf_bytes = exporter.create_comprehensive_pdf_report(
                        portfolio_data=st.session_state.portfolio_analysis_results,
                        risk_data=risk_metrics,
                        charts=export_charts,
                        technical_data=None,
                        watchlist_data=None,
                        ml_signals=None
                    )
                
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download Complete PDF Report",
                        data=pdf_bytes,
                        file_name=f"portfolio_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help="Complete portfolio analysis report with all data and interactive charts",
                        type="primary"
                    )
                    st.success("✅ PDF report generated successfully!")
                else:
                    st.error("❌ Failed to generate PDF report")
                
            except Exception as e:
                st.error(f"Error generating enhanced PDF: {e}")
                st.info("Please ensure all analysis has been completed before exporting")
    
    with col2:
        # Comprehensive Excel Export
        if st.button("📊 Export Complete Excel Report"):
            try:
                from utils.comprehensive_export import ComprehensiveExporter
                
                exporter = ComprehensiveExporter()
                
                # Prepare data for Excel
                excel_data = {}
                
                # Portfolio holdings
                portfolio_summary = []
                for symbol, weight in portfolio_data.items():
                    portfolio_summary.append({
                        'Symbol': symbol,
                        'Weight (%)': weight if isinstance(weight, (int, float)) else 0,
                        'Sector': 'N/A'
                    })
                
                if portfolio_summary:
                    excel_data['Portfolio Holdings'] = pd.DataFrame(portfolio_summary)
                
                # Performance metrics
                performance_data = []
                for key, value in portfolio_metrics.items():
                    performance_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value if isinstance(value, (int, float)) else str(value)
                    })
                excel_data['Performance Metrics'] = pd.DataFrame(performance_data)
                
                # Risk metrics
                risk_data = []
                for key, value in risk_metrics.items():
                    risk_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value if isinstance(value, (int, float)) else str(value)
                    })
                excel_data['Risk Metrics'] = pd.DataFrame(risk_data)
                
                # Diversification metrics
                div_data = []
                for key, value in diversification.items():
                    div_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value if isinstance(value, (int, float)) else str(value)
                    })
                excel_data['Diversification'] = pd.DataFrame(div_data)
                
                # Sector analysis
                sector_analysis = results.get('sector_analysis', {})
                if sector_analysis.get('sector_weights'):
                    sector_data = []
                    for sector, weight in sector_analysis['sector_weights'].items():
                        sector_data.append({
                            'Sector': sector,
                            'Weight (%)': weight * 100
                        })
                    excel_data['Sector Analysis'] = pd.DataFrame(sector_data)
                
                # VaR analysis
                var_data = []
                var_analysis = results.get('var_analysis', {})
                for key, value in var_analysis.items():
                    var_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value if isinstance(value, (int, float)) else str(value)
                    })
                if var_data:
                    excel_data['VaR Analysis'] = pd.DataFrame(var_data)
                
                # Recommendations
                recommendations = results.get('recommendations', [])
                if recommendations:
                    rec_data = []
                    for i, rec in enumerate(recommendations, 1):
                        rec_data.append({
                            'Recommendation #': i,
                            'Description': rec
                        })
                    excel_data['Recommendations'] = pd.DataFrame(rec_data)
                
                # Generate Excel
                excel_bytes = exporter.create_excel_report(excel_data, export_charts)
                
                st.download_button(
                    label="📥 Download Complete Excel Report",
                    data=excel_bytes,
                    file_name=f"portfolio_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Comprehensive Excel report with multiple tabs and data analysis"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {e}")
                # Fallback to basic CSV
                summary_data = []
                for key, value in portfolio_metrics.items():
                    summary_data.append(["Portfolio Metrics", key, value])
                for key, value in risk_metrics.items():
                    summary_data.append(["Risk Metrics", key, value])
                for key, value in diversification.items():
                    summary_data.append(["Diversification", key, value])
                
                export_df = pd.DataFrame(summary_data, columns=["Category", "Metric", "Value"])
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download CSV (Fallback)",
                    data=csv_buffer.getvalue(),
                    file_name=f"portfolio_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        # Charts Export
        if st.button("📊 Export All Charts"):
            if export_charts:
                import zipfile
                
                try:
                    # Create a ZIP file with all charts as PNG images
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for chart_name, chart in export_charts.items():
                            # Configure chart for consistent color export
                            chart.update_layout(
                                font=dict(size=12, color='black'),
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            
                            # Convert chart to PNG with enhanced settings
                            img_bytes = chart.to_image(
                                format="png", 
                                width=1200, 
                                height=800, 
                                scale=2,
                                engine='kaleido'
                            )
                            
                            # Clean filename
                            clean_name = "".join(c for c in chart_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                            filename = f"{clean_name}.png"
                            
                            zip_file.writestr(filename, img_bytes)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download All Charts (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"portfolio_charts_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        help="Download all charts as PNG images in a ZIP file"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating charts ZIP: {e}")
            else:
                st.info("Enable 'Show Detailed Charts' to export charts")

else:
    # Help Section
    st.markdown("---")
    st.subheader("📖 How to Use Portfolio Risk Assessment")
    
    st.markdown("""
    ### 🎯 Quick Start Guide:
    
    1. **Choose Input Method:** Select how to input your portfolio (manual, CSV, watchlist, or sample)
    2. **Enter Holdings:** Add your stock symbols and weights/amounts
    3. **Configure Analysis:** Select time period and analysis options
    4. **One-Click Analysis:** Click the analyze button for comprehensive risk assessment
    5. **Review Results:** Examine metrics, charts, and AI recommendations
    6. **Take Action:** Implement suggested risk management strategies
    
    ### 📊 What You'll Get:
    
    - **Performance Metrics:** Returns, volatility, Sharpe ratio, maximum drawdown
    - **Risk Analysis:** VaR, beta, downside deviation, stress testing
    - **Diversification Assessment:** Holdings count, concentration, sector allocation
    - **AI Recommendations:** Actionable insights for risk optimization
    - **Visual Dashboard:** Interactive charts and risk visualization
    
    ### 💡 Tips for Better Analysis:
    
    - Use at least 10-15 stocks for meaningful diversification analysis
    - Include stocks from different sectors and market caps
    - Regular analysis (monthly/quarterly) helps track risk changes
    - Consider the recommendations for portfolio optimization
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This risk analysis is for educational purposes only. "
    "Always consult with financial advisors before making investment decisions."
)