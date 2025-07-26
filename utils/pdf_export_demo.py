"""
PDF Export Demo - Shows complete data capture functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.enhanced_pdf_export import EnhancedPDFExporter

def create_demo_portfolio_data():
    """Create sample portfolio data that shows how the PDF export captures real data"""
    
    demo_data = {
        'portfolio_metrics': {
            'Annual Return': '15.2%',
            'Volatility': '18.5%',
            'Sharpe Ratio': '0.82',
            'Max Drawdown': '-12.3%',
            'Beta': '1.15',
            'Alpha': '3.8%'
        },
        'risk_metrics': {
            'Risk Grade': 'Moderate Risk',
            'Daily Volatility': '1.2%',
            'VaR 95%': '-2.1%',
            'VaR 99%': '-3.5%',
            'Sortino Ratio': '1.15',
            'Downside Deviation': '12.8%'
        },
        'diversification': {
            'Diversification Score': '78/100',
            'Number of Holdings': 8,
            'Largest Holding': '18.5%',
            'Top 5 Concentration': '68.2%',
            'HHI Index': '0.156',
            'Diversification Grade': 'Good'
        },
        'portfolio_holdings': {
            'RELIANCE.NS': 0.185,
            'TCS.NS': 0.165,
            'INFY.NS': 0.142,
            'HDFC.NS': 0.138,
            'ICICIBANK.NS': 0.125,
            'ITC.NS': 0.098,
            'SBIN.NS': 0.082,
            'BHARTIARTL.NS': 0.065
        }
    }
    
    return demo_data

def demo_pdf_export_functionality():
    """Demonstrate the PDF export functionality with sample data"""
    
    st.subheader("📄 PDF Export Demo")
    st.info("This demo shows how the PDF export captures all your analysis data exactly as shown on screen")
    
    # Create demo data
    demo_data = create_demo_portfolio_data()
    
    # Display the data that will be captured
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Portfolio Metrics (captured in PDF):**")
        for key, value in demo_data['portfolio_metrics'].items():
            st.metric(key, value)
    
    with col2:
        st.markdown("**Risk Metrics (captured in PDF):**")
        for key, value in demo_data['risk_metrics'].items():
            st.metric(key, value)
    
    # Show diversification data
    st.markdown("**Diversification Analysis (captured in PDF):**")
    div_df = pd.DataFrame(list(demo_data['diversification'].items()), columns=['Metric', 'Value'])
    st.dataframe(div_df, use_container_width=True)
    
    # Show portfolio holdings
    st.markdown("**Portfolio Holdings (captured in PDF):**")
    holdings_df = pd.DataFrame([
        {'Symbol': symbol, 'Weight': f"{weight*100:.1f}%"} 
        for symbol, weight in demo_data['portfolio_holdings'].items()
    ])
    st.dataframe(holdings_df, use_container_width=True)
    
    # Export button
    if st.button("📥 Generate Demo PDF Report", type="primary", help="Generate PDF with the data shown above"):
        try:
            # Store demo data in session state
            st.session_state.portfolio_analysis_results = demo_data
            
            # Create PDF
            exporter = EnhancedPDFExporter()
            pdf_bytes = exporter.create_comprehensive_pdf_report(
                portfolio_data=demo_data,
                risk_data=None,
                charts=None,
                technical_data=None,
                watchlist_data=None,
                ml_signals=None
            )
            
            if pdf_bytes:
                st.download_button(
                    label="📥 Download Demo PDF Report",
                    data=pdf_bytes,
                    file_name=f"demo_portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    help="Demo PDF showing complete data capture functionality",
                    type="primary"
                )
                st.success("✅ Demo PDF generated successfully! This shows how real data is captured.")
                st.info("💡 Run actual portfolio analysis on the Portfolio Risk Assessment page to generate reports with your real data.")
            else:
                st.error("❌ Failed to generate demo PDF")
                
        except Exception as e:
            st.error(f"Error creating demo PDF: {e}")

if __name__ == "__main__":
    demo_pdf_export_functionality()