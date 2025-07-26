"""
Enhanced PDF Export System for Complete Stock Market Analysis
Creates comprehensive PDF reports with all data and charts as they appear on screen
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import os

# Configure Kaleido to use installed Chrome for chart export
try:
    pio.kaleido.scope.chromium.executable = "/nix/store/qa9cnw4v5xkxyip6mb9kxqfq1z4x2dx1-chromium-138.0.7204.100/bin/chromium-browser"
except Exception:
    pass

class EnhancedPDFExporter:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        self.metric_style = ParagraphStyle(
            'MetricStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20,
            fontName='Helvetica'
        )
        
        self.highlight_style = ParagraphStyle(
            'HighlightStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.red,
            fontName='Helvetica-Bold'
        )

    def create_comprehensive_pdf_report(self, portfolio_data=None, risk_data=None, charts=None, 
                                      technical_data=None, watchlist_data=None, ml_signals=None):
        """
        Create a comprehensive PDF report with all available data and charts
        
        Args:
            portfolio_data: Portfolio analysis data
            risk_data: Risk assessment data  
            charts: Dictionary of charts to include
            technical_data: Technical analysis data
            watchlist_data: Watchlist data
            ml_signals: ML prediction signals
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        story = []
        
        # Title Page
        self._add_title_page(story)
        
        # Executive Summary
        self._add_executive_summary(story, portfolio_data, risk_data, technical_data)
        
        # Portfolio Risk Analysis Section
        if portfolio_data or risk_data:
            self._add_portfolio_analysis_section(story, portfolio_data, risk_data, charts)
        
        # Technical Analysis Section
        if technical_data:
            self._add_technical_analysis_section(story, technical_data, charts)
        
        # Watchlist Analysis Section
        if watchlist_data:
            self._add_watchlist_section(story, watchlist_data, charts)
        
        # ML Predictions Section
        if ml_signals:
            self._add_ml_predictions_section(story, ml_signals, charts)
        
        # Charts Gallery
        if charts:
            self._add_charts_gallery(story, charts)
        
        # Footer
        self._add_footer_page(story)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _add_title_page(self, story):
        """Add title page to the report"""
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("📊 Indian Stock Market Intelligence Report", self.title_style))
        story.append(Spacer(1, 0.5*inch))
        
        subtitle = f"Comprehensive Analysis Report Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(subtitle, self.styles['Heading3']))
        story.append(Spacer(1, 1*inch))
        
        # Add report description
        desc_text = """
        This comprehensive report provides detailed analysis of your stock portfolio including:
        • Risk Assessment & Portfolio Performance Metrics
        • Technical Analysis with Buy/Sell Signals  
        • Machine Learning Predictions & Market Sentiment
        • Interactive Charts & Visual Analytics
        • Investment Recommendations & Market Insights
        """
        story.append(Paragraph(desc_text, self.styles['Normal']))
        story.append(PageBreak())
    
    def _add_executive_summary(self, story, portfolio_data, risk_data, technical_data):
        """Add executive summary section"""
        story.append(Paragraph("📋 Executive Summary", self.header_style))
        
        summary_points = []
        
        # Portfolio summary with actual data
        if portfolio_data and isinstance(portfolio_data, dict):
            if 'portfolio_metrics' in portfolio_data:
                metrics = portfolio_data['portfolio_metrics']
                annual_return = metrics.get('Annual Return', 'N/A')
                volatility = metrics.get('Volatility', 'N/A') 
                sharpe_ratio = metrics.get('Sharpe Ratio', 'N/A')
                
                summary_points.append(f"Portfolio Performance: Annual Return {annual_return}, Volatility {volatility}, Sharpe Ratio {sharpe_ratio}")
            
            # Risk and diversification summary
            if 'risk_metrics' in portfolio_data:
                risk_metrics = portfolio_data['risk_metrics']
                risk_grade = risk_metrics.get('Risk Grade', 'N/A')
                summary_points.append(f"Risk Assessment: {risk_grade} risk level")
            
            if 'diversification' in portfolio_data:
                div_metrics = portfolio_data['diversification']
                div_score = div_metrics.get('Diversification Score', 'N/A')
                div_grade = div_metrics.get('Diversification Grade', 'N/A')
                holdings_count = div_metrics.get('Number of Holdings', 'N/A')
                summary_points.append(f"Diversification: {div_grade} grade with {holdings_count} holdings (Score: {div_score})")
            
            # Portfolio holdings summary
            if 'portfolio_holdings' in portfolio_data:
                holdings = portfolio_data['portfolio_holdings']
                if isinstance(holdings, dict):
                    holdings_count = len(holdings)
                    top_holdings = list(holdings.keys())[:3]
                    summary_points.append(f"Portfolio consists of {holdings_count} stocks including {', '.join(top_holdings)}")
        
        # Technical summary
        if technical_data and isinstance(technical_data, dict):
            signal = technical_data.get('Purchase Signal', 'N/A')
            trend = technical_data.get('Trend Status', 'N/A')
            stock = technical_data.get('Stock Symbol', 'Selected Stock')
            summary_points.append(f"Technical Analysis for {stock}: {signal} signal with {trend} trend")
        
        # Default summary if no data available
        if not summary_points:
            summary_points.append("Comprehensive stock market analysis report generated from current session data.")
            summary_points.append("Visit specific analysis pages to generate detailed reports with complete metrics and charts.")
        
        # Add summary points
        for point in summary_points:
            story.append(Paragraph(f"• {point}", self.metric_style))
        
        story.append(Spacer(1, 20))
        story.append(PageBreak())
    
    def _add_portfolio_analysis_section(self, story, portfolio_data, risk_data, charts):
        """Add portfolio analysis section with metrics and charts"""
        story.append(Paragraph("📈 Portfolio Risk Analysis Dashboard", self.header_style))
        
        if portfolio_data and isinstance(portfolio_data, dict):
            # Portfolio Metrics Section
            if 'portfolio_metrics' in portfolio_data:
                story.append(Paragraph("Portfolio Performance Metrics", self.subheader_style))
                metrics_data = []
                for key, value in portfolio_data['portfolio_metrics'].items():
                    metrics_data.append([key, str(value)])
                
                if metrics_data:
                    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
                    metrics_table.setStyle(self._get_default_table_style())
                    story.append(metrics_table)
                    story.append(Spacer(1, 15))
            
            # Risk Metrics Section
            if 'risk_metrics' in portfolio_data:
                story.append(Paragraph("Risk Analysis Metrics", self.subheader_style))
                risk_data_table = []
                for key, value in portfolio_data['risk_metrics'].items():
                    risk_data_table.append([key, str(value)])
                
                if risk_data_table:
                    risk_table = Table(risk_data_table, colWidths=[3*inch, 2*inch])
                    risk_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(risk_table)
                    story.append(Spacer(1, 15))
            
            # Diversification Section
            if 'diversification' in portfolio_data:
                story.append(Paragraph("Diversification Analysis", self.subheader_style))
                div_data = []
                for key, value in portfolio_data['diversification'].items():
                    div_data.append([key, str(value)])
                
                if div_data:
                    div_table = Table(div_data, colWidths=[3*inch, 2*inch])
                    div_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(div_table)
                    story.append(Spacer(1, 15))
            
            # Portfolio Holdings Section
            if 'portfolio_holdings' in portfolio_data:
                story.append(Paragraph("Portfolio Holdings", self.subheader_style))
                holdings = portfolio_data['portfolio_holdings']
                if isinstance(holdings, dict):
                    holdings_data = [['Symbol', 'Weight']]
                    for symbol, weight in holdings.items():
                        weight_str = f"{weight*100:.2f}%" if isinstance(weight, (int, float)) else str(weight)
                        holdings_data.append([symbol, weight_str])
                    
                    holdings_table = Table(holdings_data, colWidths=[2*inch, 2*inch])
                    holdings_table.setStyle(self._get_default_table_style())
                    story.append(holdings_table)
                    story.append(Spacer(1, 15))
        
        # Add portfolio charts if available
        if charts:
            story.append(Paragraph("Portfolio Analysis Charts", self.subheader_style))
            for chart_name, chart in charts.items():
                story.append(Paragraph(f"Chart: {chart_name}", self.metric_style))
                chart_image = self._convert_chart_to_image(chart)
                if chart_image:
                    story.append(chart_image)
                    story.append(Spacer(1, 15))
        
        story.append(PageBreak())
    
    def _add_technical_analysis_section(self, story, technical_data, charts):
        """Add technical analysis section"""
        story.append(Paragraph("📊 Technical Analysis", self.header_style))
        
        if isinstance(technical_data, dict):
            # Technical indicators
            story.append(Paragraph("Technical Indicators", self.subheader_style))
            
            tech_metrics = []
            for key, value in technical_data.items():
                if key in ['RSI', 'MACD Signal', 'Support Level', 'Resistance Level', 'Volume Status', 'Trend']:
                    tech_metrics.append([key, str(value)])
            
            if tech_metrics:
                tech_table = Table(tech_metrics, colWidths=[3*inch, 2*inch])
                tech_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(tech_table)
                story.append(Spacer(1, 20))
        
        story.append(PageBreak())
    
    def _add_watchlist_section(self, story, watchlist_data, charts):
        """Add watchlist analysis section"""
        story.append(Paragraph("⭐ Watchlist Analysis", self.header_style))
        
        if isinstance(watchlist_data, pd.DataFrame) and not watchlist_data.empty:
            # Watchlist summary
            story.append(Paragraph(f"Total Stocks in Watchlist: {len(watchlist_data)}", self.metric_style))
            
            # Convert to table
            table_data = self._df_to_table_data(watchlist_data, max_rows=15)
            if table_data:
                watchlist_table = Table(table_data)
                watchlist_table.setStyle(self._get_default_table_style())
                story.append(watchlist_table)
                story.append(Spacer(1, 20))
        
        story.append(PageBreak())
    
    def _add_ml_predictions_section(self, story, ml_signals, charts):
        """Add ML predictions section"""
        story.append(Paragraph("🤖 Machine Learning Predictions", self.header_style))
        
        if isinstance(ml_signals, dict):
            # ML summary
            story.append(Paragraph("AI-Generated Market Signals", self.subheader_style))
            
            for signal_type, signal_data in ml_signals.items():
                story.append(Paragraph(f"{signal_type}: {signal_data}", self.metric_style))
        
        story.append(PageBreak())
    
    def _add_charts_gallery(self, story, charts):
        """Add comprehensive charts gallery"""
        story.append(Paragraph("📊 Charts Gallery", self.header_style))
        
        chart_count = 0
        for chart_name, chart in charts.items():
            if chart_count % 2 == 0 and chart_count > 0:
                story.append(PageBreak())
            
            story.append(Paragraph(chart_name, self.subheader_style))
            chart_image = self._convert_chart_to_image(chart)
            if chart_image:
                story.append(chart_image)
                story.append(Spacer(1, 20))
                chart_count += 1
    
    def _add_footer_page(self, story):
        """Add footer page with disclaimers"""
        story.append(PageBreak())
        story.append(Paragraph("📋 Important Disclaimers", self.header_style))
        
        disclaimer_text = """
        Investment Disclaimer: This report is for informational purposes only and does not constitute 
        financial advice. Past performance does not guarantee future results. Please consult with a 
        qualified financial advisor before making investment decisions.
        
        Data Sources: All data is sourced from publicly available market data providers. While we 
        strive for accuracy, we cannot guarantee the completeness or accuracy of all information.
        
        Risk Warning: Stock market investments carry inherent risks. The value of investments can 
        go down as well as up, and you may not get back the amount originally invested.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        footer_text = f"Report generated by Indian Stock Market Intelligence Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, self.styles['Italic']))
    
    def _convert_chart_to_image(self, chart, width=6*inch, height=4*inch):
        """Convert Plotly chart to image for PDF with enhanced color preservation"""
        try:
            if hasattr(chart, 'to_image'):
                # Try to convert Plotly figure to image with enhanced settings
                try:
                    # Configure chart for better PDF export with preserved colors
                    chart.update_layout(
                        font=dict(size=12, color='black'),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        width=800,
                        height=600
                    )
                    
                    # Generate high-quality image with color preservation
                    img_bytes = chart.to_image(
                        format='png', 
                        width=800, 
                        height=600, 
                        scale=3,  # Higher scale for better quality
                        engine='kaleido'  # Explicitly use kaleido engine
                    )
                    img_buffer = io.BytesIO(img_bytes)
                    img = Image(img_buffer, width=width, height=height)
                    return img
                except Exception as chrome_error:
                    # Chrome/Kaleido not available - create text representation
                    chart_title = "Chart"
                    if hasattr(chart, 'layout') and chart.layout.title:
                        chart_title = str(chart.layout.title.text) if chart.layout.title.text else "Chart"
                    
                    # Create a simple table showing chart info
                    chart_info = [["Chart Type", "Details"]]
                    chart_info.append(["Title", chart_title])
                    
                    if hasattr(chart, 'data') and chart.data:
                        chart_info.append(["Type", str(chart.data[0].type) if chart.data[0].type else "Plot"])
                        chart_info.append(["Series Count", str(len(chart.data))])
                    
                    chart_info.append(["Status", "Chart visualization available in web app"])
                    
                    chart_table = Table(chart_info, colWidths=[2*inch, 3*inch])
                    chart_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP')
                    ]))
                    return chart_table
            else:
                return None
        except Exception as e:
            print(f"Error converting chart to image: {e}")
            # Return simple text placeholder
            return Paragraph("[Chart visualization - view in web application]", self.metric_style)
    
    def _df_to_table_data(self, df, max_rows=15):
        """Convert DataFrame to table data for PDF"""
        if df.empty:
            return None
            
        # Limit rows to prevent PDF overflow
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        # Convert to list of lists
        table_data = [display_df.columns.tolist()]
        
        for _, row in display_df.iterrows():
            formatted_row = []
            for val in row:
                if pd.isna(val):
                    formatted_row.append('N/A')
                elif isinstance(val, (int, float)):
                    if abs(val) > 1000:
                        formatted_row.append(f'{val:,.0f}')
                    else:
                        formatted_row.append(f'{val:.2f}')
                else:
                    formatted_row.append(str(val)[:25])  # Truncate long strings
            table_data.append(formatted_row)
        
        return table_data
    
    def _get_default_table_style(self):
        """Get default table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ])

def create_one_click_pdf_export():
    """
    Create one-click PDF export that captures all current app data and charts
    """
    try:
        exporter = EnhancedPDFExporter()
        
        # Collect all available data from session state with fallbacks
        portfolio_data = st.session_state.get('portfolio_analysis_results', None)
        risk_data = st.session_state.get('risk_metrics', None)
        technical_data = st.session_state.get('technical_analysis', None)
        watchlist_data = st.session_state.get('watchlist_data', None)
        ml_signals = st.session_state.get('ml_signals', None)
        charts = st.session_state.get('export_charts', {})
        
        # If no specific data, try to get any available analysis data
        if not portfolio_data and not risk_data and not technical_data:
            # Check for any portfolio data
            if hasattr(st.session_state, 'portfolio_for_analysis'):
                portfolio_data = {'portfolio_holdings': st.session_state.portfolio_for_analysis}
            
            # Check for watchlist data to create a simple portfolio
            if st.session_state.get('watchlist') and len(st.session_state.watchlist) > 0:
                watchlist_stocks = st.session_state.watchlist
                watchlist_data = pd.DataFrame({
                    'Symbol': watchlist_stocks,
                    'Status': ['In Watchlist'] * len(watchlist_stocks)
                })
        
        # If still no data, create a summary report
        if not any([portfolio_data, risk_data, technical_data, watchlist_data, ml_signals]):
            # Create a session summary report
            session_summary = {
                'Session Info': {
                    'Report Type': 'Session Summary',
                    'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Watchlist Size': len(st.session_state.get('watchlist', [])),
                    'Available Pages': 'All analysis pages available for detailed reports'
                }
            }
            portfolio_data = session_summary
        
        # Generate comprehensive PDF
        pdf_bytes = exporter.create_comprehensive_pdf_report(
            portfolio_data=portfolio_data,
            risk_data=risk_data,
            charts=charts,
            technical_data=technical_data,
            watchlist_data=watchlist_data,
            ml_signals=ml_signals
        )
        
        return pdf_bytes
    
    except Exception as e:
        st.error(f"Error creating PDF export: {e}")
        return None

def add_one_click_pdf_button():
    """
    Add the one-click PDF export button to any page
    """
    st.markdown("---")
    st.subheader("📥 Complete Report Export")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📄 Generate Complete PDF Report", 
                    type="primary", 
                    help="Export all data and charts from the current session",
                    use_container_width=True):
            
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_bytes = create_one_click_pdf_export()
                
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download Complete Report",
                        data=pdf_bytes,
                        file_name=f"stock_market_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help="Complete stock market analysis report with all data and charts",
                        type="primary",
                        use_container_width=True
                    )
                    st.success("✅ PDF report generated successfully!")
                else:
                    st.error("❌ Failed to generate PDF report. Please try again.")