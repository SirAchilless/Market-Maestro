"""
Simplified PDF Export System - No Chrome Dependencies
Creates comprehensive PDF reports with all data as it appears on screen
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class SimplePDFExporter:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        self.metric_style = ParagraphStyle(
            'MetricStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20
        )
    
    def create_comprehensive_report(self, portfolio_data=None, risk_data=None, charts=None, technical_data=None, watchlist_data=None, ml_signals=None):
        """Create comprehensive PDF report with all available data"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Title Page
        self._add_title_page(story)
        
        # Executive Summary
        self._add_executive_summary(story, portfolio_data, risk_data, technical_data)
        
        # Portfolio Analysis Section
        if portfolio_data:
            self._add_portfolio_section(story, portfolio_data)
        
        # Technical Analysis Section
        if technical_data:
            self._add_technical_section(story, technical_data)
        
        # Watchlist Section
        if watchlist_data:
            self._add_watchlist_section(story, watchlist_data)
        
        # ML Signals Section
        if ml_signals:
            self._add_ml_section(story, ml_signals)
        
        # Charts Information Section
        if charts:
            self._add_charts_info_section(story, charts)
        
        # Footer and Disclaimers
        self._add_footer_section(story)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _add_title_page(self, story):
        """Add title page"""
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("📊 Indian Stock Market Intelligence Report", self.title_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Comprehensive Analysis Report Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['Normal']))
        story.append(Spacer(1, 1*inch))
        
        description = """
        This comprehensive report provides detailed analysis of your stock portfolio including:
        • Risk Assessment & Portfolio Performance Metrics
        • Technical Analysis with Buy/Sell Signals
        • Machine Learning Predictions & Market Sentiment
        • Interactive Charts & Visual Analytics
        • Investment Recommendations & Market Insights
        """
        story.append(Paragraph(description, self.styles['Normal']))
        story.append(PageBreak())
    
    def _add_executive_summary(self, story, portfolio_data, risk_data, technical_data):
        """Add executive summary"""
        story.append(Paragraph("📋 Executive Summary", self.header_style))
        
        summary_points = []
        
        if portfolio_data and isinstance(portfolio_data, dict):
            if 'portfolio_metrics' in portfolio_data:
                metrics = portfolio_data['portfolio_metrics']
                annual_return = metrics.get('Annual Return', 'N/A')
                volatility = metrics.get('Volatility', 'N/A')
                sharpe_ratio = metrics.get('Sharpe Ratio', 'N/A')
                summary_points.append(f"Portfolio Performance: Annual Return {annual_return}, Volatility {volatility}, Sharpe Ratio {sharpe_ratio}")
            
            if 'risk_metrics' in portfolio_data:
                risk_metrics = portfolio_data['risk_metrics']
                risk_grade = risk_metrics.get('Risk Grade', 'N/A')
                summary_points.append(f"Risk Assessment: {risk_grade} risk level")
            
            if 'diversification' in portfolio_data:
                div_metrics = portfolio_data['diversification']
                div_grade = div_metrics.get('Diversification Grade', 'N/A')
                holdings_count = div_metrics.get('Number of Holdings', 'N/A')
                summary_points.append(f"Diversification: {div_grade} grade with {holdings_count} holdings")
        
        if technical_data and isinstance(technical_data, dict):
            signal = technical_data.get('Purchase Signal', 'N/A')
            stock = technical_data.get('Stock Symbol', 'Selected Stock')
            summary_points.append(f"Technical Analysis for {stock}: {signal} signal")
        
        if not summary_points:
            summary_points.append("Comprehensive stock market analysis report generated from current session data.")
            summary_points.append("Visit specific analysis pages to generate detailed reports with complete metrics.")
        
        for point in summary_points:
            story.append(Paragraph(f"• {point}", self.metric_style))
        
        story.append(Spacer(1, 20))
        story.append(PageBreak())
    
    def _add_portfolio_section(self, story, portfolio_data):
        """Add portfolio analysis section"""
        story.append(Paragraph("📈 Portfolio Risk Analysis Dashboard", self.header_style))
        
        if 'portfolio_metrics' in portfolio_data:
            story.append(Paragraph("Portfolio Performance Metrics", self.subheader_style))
            metrics_data = [['Metric', 'Value']]
            for key, value in portfolio_data['portfolio_metrics'].items():
                metrics_data.append([key, str(value)])
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(self._get_table_style())
            story.append(metrics_table)
            story.append(Spacer(1, 15))
        
        if 'risk_metrics' in portfolio_data:
            story.append(Paragraph("Risk Analysis Metrics", self.subheader_style))
            risk_data = [['Risk Metric', 'Value']]
            for key, value in portfolio_data['risk_metrics'].items():
                risk_data.append([key, str(value)])
            
            risk_table = Table(risk_data, colWidths=[3*inch, 2*inch])
            risk_table.setStyle(self._get_table_style())
            story.append(risk_table)
            story.append(Spacer(1, 15))
        
        if 'diversification' in portfolio_data:
            story.append(Paragraph("Diversification Analysis", self.subheader_style))
            div_data = [['Diversification Metric', 'Value']]
            for key, value in portfolio_data['diversification'].items():
                div_data.append([key, str(value)])
            
            div_table = Table(div_data, colWidths=[3*inch, 2*inch])
            div_table.setStyle(self._get_table_style())
            story.append(div_table)
            story.append(Spacer(1, 15))
        
        if 'portfolio_holdings' in portfolio_data:
            story.append(Paragraph("Portfolio Holdings", self.subheader_style))
            holdings = portfolio_data['portfolio_holdings']
            if isinstance(holdings, dict):
                holdings_data = [['Stock Symbol', 'Weight']]
                for symbol, weight in holdings.items():
                    weight_str = f"{weight*100:.2f}%" if isinstance(weight, (int, float)) else str(weight)
                    holdings_data.append([symbol, weight_str])
                
                holdings_table = Table(holdings_data, colWidths=[2.5*inch, 2*inch])
                holdings_table.setStyle(self._get_table_style())
                story.append(holdings_table)
                story.append(Spacer(1, 15))
        
        story.append(PageBreak())
    
    def _add_technical_section(self, story, technical_data):
        """Add technical analysis section"""
        story.append(Paragraph("🔍 Technical Analysis", self.header_style))
        
        if isinstance(technical_data, dict):
            tech_data = [['Technical Indicator', 'Value']]
            for key, value in technical_data.items():
                tech_data.append([key, str(value)])
            
            tech_table = Table(tech_data, colWidths=[3*inch, 2*inch])
            tech_table.setStyle(self._get_table_style())
            story.append(tech_table)
        
        story.append(PageBreak())
    
    def _add_watchlist_section(self, story, watchlist_data):
        """Add watchlist section"""
        story.append(Paragraph("⭐ Watchlist Analysis", self.header_style))
        
        if isinstance(watchlist_data, pd.DataFrame) and not watchlist_data.empty:
            story.append(Paragraph(f"Total Stocks in Watchlist: {len(watchlist_data)}", self.metric_style))
            
            # Convert dataframe to table
            table_data = [watchlist_data.columns.tolist()]
            for _, row in watchlist_data.iterrows():
                table_data.append([str(val) for val in row])
            
            watchlist_table = Table(table_data)
            watchlist_table.setStyle(self._get_table_style())
            story.append(watchlist_table)
        
        story.append(PageBreak())
    
    def _add_ml_section(self, story, ml_signals):
        """Add ML predictions section"""
        story.append(Paragraph("🤖 Machine Learning Predictions", self.header_style))
        
        if isinstance(ml_signals, dict):
            ml_data = [['ML Signal Type', 'Prediction']]
            for signal_type, signal_data in ml_signals.items():
                ml_data.append([signal_type, str(signal_data)])
            
            ml_table = Table(ml_data, colWidths=[3*inch, 2*inch])
            ml_table.setStyle(self._get_table_style())
            story.append(ml_table)
        
        story.append(PageBreak())
    
    def _add_charts_info_section(self, story, charts):
        """Add charts information section"""
        story.append(Paragraph("📊 Charts & Visualizations", self.header_style))
        
        story.append(Paragraph("Chart Summary", self.subheader_style))
        chart_info = [['Chart Name', 'Type', 'Status']]
        
        for chart_name, chart in charts.items():
            chart_type = "Interactive Chart"
            if hasattr(chart, 'data') and chart.data:
                chart_type = str(chart.data[0].type) if chart.data[0].type else "Plot"
            
            chart_info.append([chart_name, chart_type, "Available in Web App"])
        
        charts_table = Table(chart_info, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        charts_table.setStyle(self._get_table_style())
        story.append(charts_table)
        
        story.append(Spacer(1, 15))
        story.append(Paragraph("Note: Interactive charts and visualizations are available in the web application for detailed analysis.", self.metric_style))
        
        story.append(PageBreak())
    
    def _add_footer_section(self, story):
        """Add footer with disclaimers"""
        story.append(Paragraph("📋 Important Disclaimers", self.header_style))
        
        disclaimer_text = """
        Investment Disclaimer: This report is for informational purposes only and does not constitute financial advice. 
        Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.
        
        Data Sources: All data is sourced from publicly available market data providers. While we strive for accuracy, 
        we cannot guarantee the completeness or accuracy of all information.
        
        Risk Warning: Stock market investments carry inherent risks. The value of investments can go down as well as up, 
        and you may not get back the amount originally invested.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        footer_text = f"Report generated by Indian Stock Market Intelligence Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, self.styles['Italic']))
    
    def _get_table_style(self):
        """Get default table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ])

def create_simple_pdf_export():
    """Create simple PDF export without Chrome dependencies"""
    try:
        exporter = SimplePDFExporter()
        
        # Collect data from session state
        portfolio_data = st.session_state.get('portfolio_analysis_results', None)
        risk_data = st.session_state.get('risk_metrics', None)
        technical_data = st.session_state.get('technical_analysis', None)
        watchlist_data = st.session_state.get('watchlist_data', None)
        ml_signals = st.session_state.get('ml_signals', None)
        charts = st.session_state.get('export_charts', {})
        
        # Generate PDF
        pdf_bytes = exporter.create_comprehensive_report(
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