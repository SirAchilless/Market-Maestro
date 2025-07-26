"""
Comprehensive Export System for Stock Market Analysis
Supports PDF and Excel export with charts and data from all application pages
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ComprehensiveExporter:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkgreen
        )

    def create_pdf_report(self, data_dict, charts_dict=None):
        """
        Create comprehensive PDF report with all data and charts
        
        Args:
            data_dict: Dictionary containing data from different pages
            charts_dict: Dictionary containing chart objects
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Title page
        story.append(Paragraph("Stock Market Analysis Report", self.title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.header_style))
        summary_text = self._generate_executive_summary(data_dict)
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add data sections
        for section_name, section_data in data_dict.items():
            story.append(Paragraph(f"{section_name} Analysis", self.header_style))
            
            if isinstance(section_data, pd.DataFrame):
                # Convert DataFrame to table
                table_data = self._df_to_table_data(section_data)
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            elif isinstance(section_data, dict):
                # Convert dict to formatted text
                for key, value in section_data.items():
                    story.append(Paragraph(f"<b>{key}:</b> {value}", self.styles['Normal']))
            
            story.append(Spacer(1, 15))
            
            # Add charts if available
            if charts_dict and section_name in charts_dict:
                chart_image = self._plotly_to_image(charts_dict[section_name])
                if chart_image:
                    story.append(chart_image)
                    story.append(Spacer(1, 10))
            
            story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def create_excel_report(self, data_dict, charts_dict=None):
        """
        Create comprehensive Excel report with multiple tabs and embedded charts
        
        Args:
            data_dict: Dictionary containing data from different pages
            charts_dict: Dictionary containing chart objects
        """
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = self._generate_summary_data(data_dict)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
                
                # Data sheets
                for sheet_name, data in data_dict.items():
                    # Clean sheet name for Excel compatibility
                    clean_sheet_name = sheet_name.replace('/', '_')[:31]  # Excel sheet name limit
                    
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                    elif isinstance(data, dict):
                        # Convert dict to DataFrame
                        dict_df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                        dict_df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                    elif isinstance(data, list):
                        # Convert list to DataFrame
                        list_df = pd.DataFrame(data)
                        list_df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                
                # Charts sheet (if charts available)
                if charts_dict:
                    charts_summary = []
                    for chart_name, chart_obj in charts_dict.items():
                        charts_summary.append({
                            'Chart Name': chart_name,
                            'Type': type(chart_obj).__name__,
                            'Description': f'Chart showing {chart_name} analysis'
                        })
                    
                    if charts_summary:
                        charts_df = pd.DataFrame(charts_summary)
                        charts_df.to_excel(writer, sheet_name='Charts Summary', index=False)
        except Exception as e:
            print(f"Error creating Excel report: {e}")
            return None
        
        output.seek(0)
        return output.getvalue()

    def _generate_executive_summary(self, data_dict):
        """Generate executive summary text from data"""
        summary_parts = []
        
        # Count total stocks analyzed
        total_stocks = 0
        for data in data_dict.values():
            if isinstance(data, pd.DataFrame):
                total_stocks += len(data)
        
        summary_parts.append(f"This comprehensive report analyzes {total_stocks} stocks across multiple dimensions including technical analysis, fundamental analysis, and risk assessment.")
        
        # Add key insights based on available data
        if 'Watchlist' in data_dict:
            watchlist_data = data_dict['Watchlist']
            if isinstance(watchlist_data, pd.DataFrame) and not watchlist_data.empty:
                buy_signals = len(watchlist_data[watchlist_data.get('Purchase Signal', '') == 'Buy'])
                summary_parts.append(f"Current analysis shows {buy_signals} stocks with active buy signals.")
        
        if 'Risk Analysis' in data_dict:
            risk_data = data_dict['Risk Analysis']
            if isinstance(risk_data, dict):
                risk_grade = risk_data.get('Risk Grade', 'N/A')
                summary_parts.append(f"Portfolio risk assessment indicates a {risk_grade} risk level.")
        
        return " ".join(summary_parts)

    def _generate_summary_data(self, data_dict):
        """Generate summary statistics from all data"""
        summary = {
            'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Total Data Sections': [len(data_dict)],
            'Analysis Type': ['Comprehensive Stock Market Analysis']
        }
        
        # Add section-specific summaries
        for section_name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                summary[f'{section_name} Records'] = [len(data)]
                if 'Purchase Signal' in data.columns:
                    buy_count = len(data[data['Purchase Signal'] == 'Buy'])
                    summary[f'{section_name} Buy Signals'] = [buy_count]
        
        return summary

    def _df_to_table_data(self, df, max_rows=20):
        """Convert DataFrame to table data for PDF"""
        # Limit rows to prevent PDF overflow
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        # Convert to list of lists
        table_data = [display_df.columns.tolist()]
        
        for _, row in display_df.iterrows():
            # Format values for display
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
                    formatted_row.append(str(val)[:20])  # Truncate long strings
            table_data.append(formatted_row)
        
        return table_data

    def _plotly_to_image(self, fig, width=500, height=300):
        """Convert Plotly figure to image for PDF with enhanced color preservation"""
        try:
            # Configure chart for better PDF export with preserved colors
            fig.update_layout(
                font=dict(size=12, color='black'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            # Convert Plotly figure to image bytes with enhanced settings
            img_bytes = pio.to_image(
                fig, 
                format='png', 
                width=width, 
                height=height, 
                scale=2,
                engine='kaleido'
            )
            
            # Create ReportLab Image
            img_buffer = io.BytesIO(img_bytes)
            img = Image(img_buffer, width=4*inch, height=2.4*inch)
            return img
        except Exception as e:
            print(f"Error converting chart to image: {e}")
            return None

def export_comprehensive_data(export_type='excel'):
    """
    Main function to export comprehensive data from all pages
    
    Args:
        export_type: 'excel' or 'pdf'
    """
    exporter = ComprehensiveExporter()
    data_dict = {}
    charts_dict = {}
    
    # Collect data from session state and various sources
    try:
        # Watchlist data
        if 'watchlist_data' in st.session_state and st.session_state.watchlist_data:
            watchlist_summary = []
            for symbol, data in st.session_state.watchlist_data.items():
                if not data.empty:
                    summary = {
                        'Symbol': symbol,
                        'Current Price': data['Close'].iloc[-1] if len(data) > 0 else 0,
                        'Volume': data['Volume'].iloc[-1] if len(data) > 0 else 0,
                        'High': data['High'].max() if len(data) > 0 else 0,
                        'Low': data['Low'].min() if len(data) > 0 else 0
                    }
                    watchlist_summary.append(summary)
            
            if watchlist_summary:
                data_dict['Watchlist Data'] = pd.DataFrame(watchlist_summary)
        
        # Portfolio data
        if 'portfolio_data' in st.session_state:
            data_dict['Portfolio'] = st.session_state.portfolio_data
        
        # Risk analysis data
        if 'risk_metrics' in st.session_state:
            data_dict['Risk Analysis'] = st.session_state.risk_metrics
        
        # ML signals data
        if hasattr(st, 'session_state') and 'ml_signals' in st.session_state:
            data_dict['ML Signals'] = st.session_state.ml_signals
        
    except Exception as e:
        st.error(f"Error collecting data: {e}")
    
    # Generate report
    if export_type == 'pdf':
        return exporter.create_pdf_report(data_dict, charts_dict)
    else:
        return exporter.create_excel_report(data_dict, charts_dict)

def create_download_link(file_data, filename, mime_type):
    """Create download link for exported data"""
    b64 = base64.b64encode(file_data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">📥 Download {filename}</a>'