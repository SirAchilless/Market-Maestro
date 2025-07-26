import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def create_candlestick_chart(data, symbol, indicators=None, height=600):
    """
    Create an interactive candlestick chart with technical indicators
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV
        symbol (str): Stock symbol for title
        indicators (dict): Technical indicators to overlay
        height (int): Chart height
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    try:
        if data.empty:
            st.warning("No data available for chart")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} - Price & Technical Indicators', 'MACD', 'Volume'),
            row_width=[0.2, 0.1, 0.1]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add technical indicators if available
        if indicators:
            # Moving Averages
            if 'MA20' in indicators and len(indicators['MA20']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['MA20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'MA44' in indicators and len(indicators['MA44']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['MA44'],
                        mode='lines',
                        name='MA44',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'MA200' in indicators and len(indicators['MA200']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['MA200'],
                        mode='lines',
                        name='MA200',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(k in indicators for k in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                if len(indicators['BB_Upper']) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.1)'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['BB_Middle'],
                            mode='lines',
                            name='BB Middle',
                            line=dict(color='purple', width=1)
                        ),
                        row=1, col=1
                    )
            
            # MACD
            if all(k in indicators for k in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                if len(indicators['MACD']) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['MACD_Signal'],
                            mode='lines',
                            name='MACD Signal',
                            line=dict(color='red', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    # MACD Histogram
                    histogram_colors = ['green' if x >= 0 else 'red' for x in indicators['MACD_Histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=indicators['MACD_Histogram'],
                            name='MACD Histogram',
                            marker_color=histogram_colors,
                            opacity=0.6
                        ),
                        row=2, col=1
                    )
        
        # Volume
        volume_colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
                        for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return go.Figure()

def create_rsi_chart(data, rsi_values, symbol):
    """Create RSI chart with overbought/oversold levels"""
    try:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi_values,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
        
        fig.update_layout(
            title=f'{symbol} - RSI (14)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 100])
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating RSI chart: {str(e)}")
        return go.Figure()

def create_volume_analysis_chart(data, volume_sma, symbol):
    """Create volume analysis chart"""
    try:
        fig = go.Figure()
        
        # Volume bars
        volume_colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
                        for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            )
        )
        
        # Volume SMA
        if len(volume_sma) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=volume_sma,
                    mode='lines',
                    name='Volume SMA (20)',
                    line=dict(color='orange', width=2)
                )
            )
        
        fig.update_layout(
            title=f'{symbol} - Volume Analysis',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=300,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")
        return go.Figure()

def create_comparison_chart(symbols_data, symbols, period='1y'):
    """Create a comparison chart for multiple stocks"""
    try:
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, symbol in enumerate(symbols):
            if symbol in symbols_data and not symbols_data[symbol].empty:
                data = symbols_data[symbol]
                # Normalize to percentage change from first value
                normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
        
        fig.update_layout(
            title='Stock Comparison - Percentage Returns',
            xaxis_title='Date',
            yaxis_title='Returns (%)',
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return go.Figure()

def create_heatmap(data_dict, metric='Price_Change_Pct'):
    """Create a heatmap for stock performance"""
    try:
        # Prepare data for heatmap
        symbols = list(data_dict.keys())
        values = [data_dict[symbol].get(metric, 0) for symbol in symbols]
        
        # Create a simple heatmap using bar chart with colors
        fig = go.Figure(data=go.Bar(
            x=symbols,
            y=values,
            marker=dict(
                color=values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Performance %")
            ),
            text=[f'{v:.2f}%' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Stock Performance Heatmap - {metric}',
            xaxis_title='Stocks',
            yaxis_title='Value',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return go.Figure()
