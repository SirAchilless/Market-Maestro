"""
Portfolio Risk Assessment Module for Indian Stock Market Screener
Provides comprehensive risk analysis, diversification metrics, and portfolio optimization suggestions
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class PortfolioRiskAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.065  # Current Indian 10-year bond yield (~6.5%)
        self.market_benchmark = "^NSEI"  # NIFTY 50 as benchmark
        
    def analyze_portfolio_risk(self, portfolio: Dict[str, float], period: str = "1y") -> Dict:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            portfolio: Dict with stock symbols as keys and weights/amounts as values
            period: Analysis period (1y, 6mo, 3mo)
            
        Returns:
            Dict containing comprehensive risk metrics
        """
        try:
            # Normalize portfolio weights
            total_value = sum(portfolio.values())
            weights = {symbol: value/total_value for symbol, value in portfolio.items()}
            
            # Fetch data for all stocks
            symbols = list(portfolio.keys())
            stock_data = self._fetch_portfolio_data(symbols, period)
            
            if stock_data.empty:
                return self._empty_risk_analysis()
            
            # Calculate returns
            returns = stock_data.pct_change().dropna()
            
            # Portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns, weights, period)
            
            # Diversification analysis
            diversification = self._analyze_diversification(symbols, weights)
            
            # Sector concentration
            sector_analysis = self._analyze_sector_concentration(symbols, weights)
            
            # Value at Risk
            var_analysis = self._calculate_var(returns, weights)
            
            # Stress testing
            stress_test = self._perform_stress_test(returns, weights)
            
            # Risk recommendations
            recommendations = self._generate_risk_recommendations(
                portfolio_metrics, risk_metrics, diversification, sector_analysis
            )
            
            return {
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'diversification': diversification,
                'sector_analysis': sector_analysis,
                'var_analysis': var_analysis,
                'stress_test': stress_test,
                'recommendations': recommendations,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'period': period
            }
            
        except Exception as e:
            st.error(f"Error in portfolio risk analysis: {e}")
            return self._empty_risk_analysis()
    
    def _fetch_portfolio_data(self, symbols: List[str], period: str) -> pd.DataFrame:
        """Fetch historical data for portfolio stocks"""
        try:
            # Add .NS suffix for Indian stocks if not present
            formatted_symbols = []
            for symbol in symbols:
                if not symbol.endswith('.NS') and not symbol.startswith('^'):
                    formatted_symbols.append(f"{symbol}.NS")
                else:
                    formatted_symbols.append(symbol)
            
            # Fetch data with auto_adjust=False to get standard columns
            data = yf.download(formatted_symbols, period=period, progress=False, auto_adjust=False)
            
            if len(formatted_symbols) == 1:
                # Single stock case - use 'Adj Close' if available, otherwise 'Close'
                if 'Adj Close' in data.columns:
                    return pd.DataFrame({symbols[0]: data['Adj Close']})
                else:
                    return pd.DataFrame({symbols[0]: data['Close']})
            else:
                # Multiple stocks case - try Adj Close first, fallback to Close
                try:
                    if 'Adj Close' in data.columns:
                        close_data = data['Adj Close']
                    else:
                        close_data = data['Close']
                    
                    # Handle multi-level columns
                    if isinstance(close_data.columns, pd.MultiIndex):
                        # Extract the symbol names from multi-level columns
                        close_data.columns = [col[1] if isinstance(col, tuple) else col for col in close_data.columns]
                    
                    # Rename columns back to original symbols (remove .NS suffix)
                    symbol_mapping = {}
                    for i, formatted_sym in enumerate(formatted_symbols):
                        if i < len(symbols):
                            symbol_mapping[formatted_sym] = symbols[i]
                    
                    close_data = close_data.rename(columns=symbol_mapping)
                    return close_data.dropna()
                    
                except Exception as e:
                    st.warning(f"Error processing multi-stock data: {e}")
                    return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Error fetching data for some stocks: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Calculate basic portfolio performance metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Annual metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Additional metrics
            cumulative_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'cumulative_return': cumulative_return,
                'max_drawdown': max_drawdown,
                'total_days': len(portfolio_returns)
            }
            
        except Exception as e:
            return {
                'annual_return': 0,
                'annual_volatility': 0,
                'sharpe_ratio': 0,
                'cumulative_return': 0,
                'max_drawdown': 0,
                'total_days': 0
            }
    
    def _calculate_risk_metrics(self, returns: pd.DataFrame, weights: Dict[str, float], period: str) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Volatility metrics
            daily_vol = portfolio_returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (portfolio_returns.mean() * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Beta calculation (vs NIFTY)
            try:
                nifty_data = yf.download(self.market_benchmark, period=period, progress=False)['Adj Close']
                nifty_returns = nifty_data.pct_change().dropna()
                
                # Align dates
                common_dates = portfolio_returns.index.intersection(nifty_returns.index)
                if len(common_dates) > 30:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    nifty_aligned = nifty_returns.loc[common_dates]
                    
                    covariance = np.cov(portfolio_aligned, nifty_aligned)[0, 1]
                    market_variance = np.var(nifty_aligned)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            # Risk grade
            risk_grade = self._calculate_risk_grade(annual_vol, beta, downside_deviation)
            
            return {
                'daily_volatility': daily_vol,
                'annual_volatility': annual_vol,
                'downside_deviation': downside_deviation,
                'sortino_ratio': sortino_ratio,
                'beta': beta,
                'risk_grade': risk_grade,
                'volatility_percentile': self._get_volatility_percentile(annual_vol)
            }
            
        except Exception as e:
            return {
                'daily_volatility': 0,
                'annual_volatility': 0,
                'downside_deviation': 0,
                'sortino_ratio': 0,
                'beta': 1.0,
                'risk_grade': 'Unknown',
                'volatility_percentile': 50
            }
    
    def _analyze_diversification(self, symbols: List[str], weights: Dict[str, float]) -> Dict:
        """Analyze portfolio diversification"""
        try:
            # Number of holdings
            num_holdings = len(symbols)
            
            # Concentration analysis
            weight_values = list(weights.values())
            largest_holding = max(weight_values)
            top_5_concentration = sum(sorted(weight_values, reverse=True)[:5])
            
            # Herfindahl-Hirschman Index (HHI)
            hhi = sum(w**2 for w in weight_values)
            
            # Diversification score (0-100)
            diversification_score = self._calculate_diversification_score(num_holdings, hhi, largest_holding)
            
            # Diversification grade
            if diversification_score >= 80:
                grade = "Excellent"
            elif diversification_score >= 60:
                grade = "Good"
            elif diversification_score >= 40:
                grade = "Moderate"
            else:
                grade = "Poor"
            
            return {
                'num_holdings': num_holdings,
                'largest_holding_weight': largest_holding,
                'top_5_concentration': top_5_concentration,
                'hhi_index': hhi,
                'diversification_score': diversification_score,
                'diversification_grade': grade
            }
            
        except Exception as e:
            return {
                'num_holdings': 0,
                'largest_holding_weight': 0,
                'top_5_concentration': 0,
                'hhi_index': 1,
                'diversification_score': 0,
                'diversification_grade': 'Unknown'
            }
    
    def _analyze_sector_concentration(self, symbols: List[str], weights: Dict[str, float]) -> Dict:
        """Analyze sector concentration risks"""
        try:
            from utils.stock_lists import SECTOR_STOCKS
            
            # Create symbol mapping for common variations
            symbol_mapping = {
                'INFY': 'INFOSYS',  # Handle INFY -> INFOSYS mapping
                'TCS': 'TCS',       # TCS is correct
                'ICICIBANK': 'ICICIBANK'  # ICICIBANK is correct
            }
            
            # Map stocks to sectors
            sector_weights = {}
            unclassified_weight = 0
            
            for symbol, weight in weights.items():
                # Clean symbol (remove .NS suffix if present)
                clean_symbol = symbol.replace('.NS', '')
                
                # Use mapping if available, otherwise use original
                mapped_symbol = symbol_mapping.get(clean_symbol, clean_symbol)
                
                found_sector = False
                for sector, stocks in SECTOR_STOCKS.items():
                    if mapped_symbol in stocks or clean_symbol in stocks:
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight
                        found_sector = True
                        break
                
                if not found_sector:
                    unclassified_weight += weight
            
            if unclassified_weight > 0:
                sector_weights['Other/Unclassified'] = unclassified_weight
            
            # Calculate concentration metrics
            if sector_weights:
                largest_sector_weight = max(sector_weights.values())
                num_sectors = len(sector_weights)
                sector_hhi = sum(w**2 for w in sector_weights.values())
            else:
                largest_sector_weight = 1.0
                num_sectors = 1
                sector_hhi = 1.0
            
            # Sector concentration grade
            if largest_sector_weight <= 0.3 and num_sectors >= 5:
                concentration_grade = "Well Diversified"
            elif largest_sector_weight <= 0.5 and num_sectors >= 3:
                concentration_grade = "Moderately Diversified"
            else:
                concentration_grade = "Concentrated"
            
            return {
                'sector_weights': sector_weights,
                'largest_sector_weight': largest_sector_weight,
                'num_sectors': num_sectors,
                'sector_hhi': sector_hhi,
                'concentration_grade': concentration_grade
            }
            
        except Exception as e:
            return {
                'sector_weights': {},
                'largest_sector_weight': 1.0,
                'num_sectors': 1,
                'sector_hhi': 1.0,
                'concentration_grade': 'Unknown'
            }
    
    def _calculate_var(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Calculate Value at Risk metrics"""
        try:
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # VaR at different confidence levels
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            return {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'var_95_annual': var_95 * np.sqrt(252),
                'var_99_annual': var_99 * np.sqrt(252),
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99
            }
            
        except Exception as e:
            return {
                'var_95_daily': 0,
                'var_99_daily': 0,
                'var_95_annual': 0,
                'var_99_annual': 0,
                'expected_shortfall_95': 0,
                'expected_shortfall_99': 0
            }
    
    def _perform_stress_test(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Perform stress testing scenarios"""
        try:
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Historical stress scenarios
            worst_day = portfolio_returns.min()
            worst_week = portfolio_returns.rolling(5).sum().min()
            worst_month = portfolio_returns.rolling(21).sum().min()
            
            # Market crash scenarios
            crash_10_percent = -0.10  # 10% market crash
            crash_20_percent = -0.20  # 20% market crash
            
            # Volatility spike scenario
            high_vol_days = portfolio_returns[abs(portfolio_returns) > portfolio_returns.std() * 2]
            volatility_spike_impact = high_vol_days.mean() if len(high_vol_days) > 0 else 0
            
            return {
                'worst_single_day': worst_day,
                'worst_week': worst_week,
                'worst_month': worst_month,
                'crash_10_percent_scenario': crash_10_percent,
                'crash_20_percent_scenario': crash_20_percent,
                'volatility_spike_impact': volatility_spike_impact,
                'stress_test_date': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            return {
                'worst_single_day': 0,
                'worst_week': 0,
                'worst_month': 0,
                'crash_10_percent_scenario': -0.10,
                'crash_20_percent_scenario': -0.20,
                'volatility_spike_impact': 0,
                'stress_test_date': datetime.now().strftime('%Y-%m-%d')
            }
    
    def _generate_risk_recommendations(self, portfolio_metrics: Dict, risk_metrics: Dict, 
                                     diversification: Dict, sector_analysis: Dict) -> List[str]:
        """Generate actionable risk management recommendations"""
        recommendations = []
        
        # Risk level recommendations
        if risk_metrics['risk_grade'] == 'High Risk':
            recommendations.append("⚠️ Your portfolio shows high risk levels. Consider reducing position sizes or adding defensive stocks.")
        
        # Diversification recommendations
        if diversification['diversification_grade'] == 'Poor':
            recommendations.append("📊 Add more holdings to improve diversification. Aim for at least 15-20 different stocks.")
        
        if diversification['largest_holding_weight'] > 0.25:
            recommendations.append("⚖️ Your largest holding exceeds 25% of portfolio. Consider reducing concentration risk.")
        
        # Sector concentration recommendations
        if sector_analysis['concentration_grade'] == 'Concentrated':
            recommendations.append("🏭 Your portfolio is concentrated in few sectors. Diversify across different industries.")
        
        # Volatility recommendations
        if risk_metrics['annual_volatility'] > 0.30:
            recommendations.append("📉 High portfolio volatility detected. Consider adding stable, large-cap stocks.")
        
        # Beta recommendations
        if risk_metrics['beta'] > 1.5:
            recommendations.append("📈 High beta portfolio - more volatile than market. Consider defensive stocks to reduce systematic risk.")
        elif risk_metrics['beta'] < 0.5:
            recommendations.append("🛡️ Low beta portfolio may underperform in bull markets. Consider adding growth stocks.")
        
        # Sharpe ratio recommendations
        if portfolio_metrics['sharpe_ratio'] < 0.5:
            recommendations.append("💡 Low risk-adjusted returns. Review stock selection and consider higher quality companies.")
        
        # Maximum drawdown recommendations
        if abs(portfolio_metrics['max_drawdown']) > 0.25:
            recommendations.append("📉 High maximum drawdown indicates significant losses. Implement stop-loss strategies.")
        
        # Default recommendation if portfolio is well-balanced
        if not recommendations:
            recommendations.append("✅ Your portfolio shows balanced risk characteristics. Continue monitoring and rebalancing quarterly.")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0
    
    def _calculate_risk_grade(self, volatility: float, beta: float, downside_dev: float) -> str:
        """Calculate overall risk grade"""
        risk_score = 0
        
        # Volatility component (40% weight)
        if volatility > 0.35:
            risk_score += 40
        elif volatility > 0.25:
            risk_score += 30
        elif volatility > 0.15:
            risk_score += 20
        else:
            risk_score += 10
        
        # Beta component (30% weight)
        if beta > 1.5:
            risk_score += 30
        elif beta > 1.2:
            risk_score += 20
        elif beta > 0.8:
            risk_score += 15
        else:
            risk_score += 10
        
        # Downside deviation component (30% weight)
        if downside_dev > 0.25:
            risk_score += 30
        elif downside_dev > 0.15:
            risk_score += 20
        elif downside_dev > 0.10:
            risk_score += 15
        else:
            risk_score += 10
        
        # Risk grade mapping
        if risk_score >= 80:
            return "Very High Risk"
        elif risk_score >= 60:
            return "High Risk"
        elif risk_score >= 40:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def _get_volatility_percentile(self, volatility: float) -> int:
        """Get volatility percentile compared to typical Indian equity portfolio"""
        # Typical Indian equity volatility ranges
        if volatility < 0.15:
            return 10  # Very low volatility
        elif volatility < 0.20:
            return 25
        elif volatility < 0.25:
            return 50  # Average
        elif volatility < 0.30:
            return 75
        else:
            return 90  # Very high volatility
    
    def _calculate_diversification_score(self, num_holdings: int, hhi: float, largest_holding: float) -> float:
        """Calculate diversification score (0-100)"""
        # Holdings component (40% weight)
        if num_holdings >= 25:
            holdings_score = 40
        elif num_holdings >= 15:
            holdings_score = 30
        elif num_holdings >= 10:
            holdings_score = 20
        else:
            holdings_score = max(0, num_holdings * 2)
        
        # HHI component (30% weight) - lower HHI is better
        hhi_score = max(0, 30 * (1 - min(hhi, 1)))
        
        # Concentration component (30% weight) - lower concentration is better
        concentration_score = max(0, 30 * (1 - min(largest_holding, 1)))
        
        return holdings_score + hhi_score + concentration_score
    
    def _empty_risk_analysis(self) -> Dict:
        """Return empty risk analysis structure"""
        return {
            'portfolio_metrics': {},
            'risk_metrics': {},
            'diversification': {},
            'sector_analysis': {},
            'var_analysis': {},
            'stress_test': {},
            'recommendations': ["Unable to analyze portfolio. Please check stock symbols and try again."],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': '1y'
        }
    
    def create_risk_dashboard_charts(self, risk_analysis: Dict) -> List[go.Figure]:
        """Create comprehensive risk dashboard charts"""
        charts = []
        
        try:
            # 1. Risk Metrics Radar Chart
            radar_chart = self._create_risk_radar_chart(risk_analysis)
            if radar_chart:
                charts.append(radar_chart)
            
            # 2. Sector Allocation Pie Chart
            sector_chart = self._create_sector_allocation_chart(risk_analysis)
            if sector_chart:
                charts.append(sector_chart)
            
            # 3. Risk-Return Scatter
            risk_return_chart = self._create_risk_return_chart(risk_analysis)
            if risk_return_chart:
                charts.append(risk_return_chart)
            
            # 4. VaR Visualization
            var_chart = self._create_var_chart(risk_analysis)
            if var_chart:
                charts.append(var_chart)
            
        except Exception as e:
            st.warning(f"Error creating charts: {e}")
        
        return charts
    
    def _create_risk_radar_chart(self, risk_analysis: Dict) -> Optional[go.Figure]:
        """Create risk metrics radar chart"""
        try:
            metrics = risk_analysis.get('risk_metrics', {})
            portfolio_metrics = risk_analysis.get('portfolio_metrics', {})
            diversification = risk_analysis.get('diversification', {})
            
            # Normalize metrics to 0-100 scale
            categories = ['Volatility', 'Beta Risk', 'Sharpe Ratio', 'Max Drawdown', 'Diversification']
            
            values = [
                max(0, 100 - metrics.get('volatility_percentile', 50)),  # Lower volatility is better
                max(0, 100 - min(abs(metrics.get('beta', 1) - 1) * 50, 100)),  # Beta closer to 1 is better
                min(portfolio_metrics.get('sharpe_ratio', 0) * 50, 100),  # Higher Sharpe is better
                max(0, 100 - abs(portfolio_metrics.get('max_drawdown', 0)) * 400),  # Lower drawdown is better
                diversification.get('diversification_score', 0)  # Higher diversification is better
            ]
            
            # Create unique figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Portfolio Risk Profile',
                line=dict(color='#007bff', width=3),
                fillcolor='rgba(0, 123, 255, 0.3)',
                marker=dict(color='#007bff', size=8),
                uid=f'radar_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickmode='linear',
                        tick0=0,
                        dtick=20
                    )),
                showlegend=True,
                title="Portfolio Risk Profile Radar",
                font=dict(size=12),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def _create_sector_allocation_chart(self, risk_analysis: Dict) -> Optional[go.Figure]:
        """Create sector allocation pie chart"""
        try:
            sector_data = risk_analysis.get('sector_analysis', {}).get('sector_weights', {})
            
            if not sector_data:
                return None
            
            labels = list(sector_data.keys())
            values = list(sector_data.values())
            
            # Define vibrant colors for sectors that will show properly in PDF
            sector_colors = [
                '#1f77b4',  # Blue
                '#ff7f0e',  # Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Olive
                '#17becf',  # Cyan
                '#aec7e8',  # Light Blue
                '#ffbb78',  # Light Orange
                '#98df8a',  # Light Green
                '#ff9896',  # Light Red
                '#c5b0d5'   # Light Purple
            ]
            
            # Ensure we have enough colors
            colors = sector_colors * (len(labels) // len(sector_colors) + 1)
            colors = colors[:len(labels)]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                uid=f'sector_pie_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )])
            
            fig.update_layout(
                title="Portfolio Sector Allocation",
                showlegend=True,
                font=dict(size=12),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def _create_risk_return_chart(self, risk_analysis: Dict) -> Optional[go.Figure]:
        """Create risk-return scatter plot"""
        try:
            portfolio_metrics = risk_analysis.get('portfolio_metrics', {})
            
            annual_return = portfolio_metrics.get('annual_return', 0) * 100
            annual_volatility = portfolio_metrics.get('annual_volatility', 0) * 100
            
            fig = go.Figure()
            
            # Portfolio point
            fig.add_trace(go.Scatter(
                x=[annual_volatility],
                y=[annual_return],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Your Portfolio',
                text=['Your Portfolio'],
                textposition='top center',
                uid=f'portfolio_point_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            ))
            
            # Market benchmark (approximate)
            fig.add_trace(go.Scatter(
                x=[18],  # Typical NIFTY volatility
                y=[12],  # Typical NIFTY return
                mode='markers',
                marker=dict(size=12, color='blue'),
                name='Market (NIFTY 50)',
                text=['NIFTY 50'],
                textposition='top center',
                uid=f'market_point_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            ))
            
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Annual Volatility (%)",
                yaxis_title="Annual Return (%)",
                showlegend=True,
                font=dict(size=12),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )
            
            return fig
            
        except Exception as e:
            return None
    
    def _create_var_chart(self, risk_analysis: Dict) -> Optional[go.Figure]:
        """Create VaR visualization chart"""
        try:
            var_data = risk_analysis.get('var_analysis', {})
            
            categories = ['VaR 95%', 'VaR 99%', 'Expected Shortfall 95%', 'Expected Shortfall 99%']
            values = [
                var_data.get('var_95_daily', 0) * 100,
                var_data.get('var_99_daily', 0) * 100,
                var_data.get('expected_shortfall_95', 0) * 100,
                var_data.get('expected_shortfall_99', 0) * 100
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['orange', 'red', 'darkorange', 'darkred']
                )
            ])
            
            fig.update_layout(
                title="Value at Risk (Daily %)",
                yaxis_title="Potential Loss (%)",
                showlegend=False,
                font=dict(size=12),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )
            
            return fig
            
        except Exception as e:
            return None