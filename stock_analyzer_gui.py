import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm

class EnhancedStockAnalyzer:
    def __init__(self):
        self.risk_metrics = {}
        self.technical_indicators = {}
        self.options_analysis = {}

    def analyze_stock(self, ticker, start_date, end_date):
        """
        Comprehensive stock analysis including technical indicators,
        risk management, and options analysis.
        """
        stock = yf.Ticker(ticker)

        # Get historical data
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            raise ValueError(f"No data available for {ticker}")

        # Get current price from last available close price
        current_price = hist['Close'].iloc[-1]

        self._calculate_technical_indicators(hist, ticker, current_price)
        self._calculate_risk_metrics(hist, ticker)
        self._analyze_options(stock, ticker)

        return self._generate_analysis_summary(ticker)

    def _calculate_technical_indicators(self, hist, ticker, current_price):
        """Calculate comprehensive technical indicators."""
        close = hist['Close']

        # Moving Averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # MACD
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        # Bollinger Bands
        bb_middle = sma_20
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # Average True Range (ATR)
        high = hist['High']
        low = hist['Low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1)
        tr['True Range'] = tr.max(axis=1)
        atr = tr['True Range'].rolling(14).mean()

        self.technical_indicators[ticker] = {
            'current_price': current_price,
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
            'macd': macd.iloc[-1],
            'macd_signal': signal.iloc[-1],
            'macd_hist': macd_hist.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'atr': atr.iloc[-1],
            'price_to_sma20': current_price / sma_20.iloc[-1] - 1 if sma_20.iloc[-1] != 0 else np.nan,
            'price_to_sma50': current_price / sma_50.iloc[-1] - 1 if sma_50.iloc[-1] != 0 else np.nan
        }

    def _calculate_risk_metrics(self, hist, ticker):
        """Calculate comprehensive risk metrics."""
        returns = hist['Close'].pct_change().dropna()

        # Value at Risk (VaR) calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Maximum Drawdown
        cumulative_return = (1 + returns).cumprod()
        peak = cumulative_return.expanding(min_periods=1).max()
        drawdown = (cumulative_return / peak) - 1
        max_drawdown = drawdown.min()

        # Volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Sharpe Ratio (assuming risk-free rate of 2%)
        excess_returns = returns - 0.02 / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_vol if daily_vol != 0 else np.nan

        self.risk_metrics[ticker] = {
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'position_size_suggest': self._calculate_position_size(annual_vol)
        }

    def _calculate_position_size(self, volatility, max_portfolio_risk=0.02):
        """Calculate suggested position size based on volatility."""
        return max_portfolio_risk / volatility if volatility != 0 else np.nan

    def _analyze_options(self, stock, ticker):
        """Enhanced options analysis."""
        try:
            options = stock.options
            if not options:
                self.options_analysis[ticker] = None
                return

            nearest_expiry = options[0]
            chain = stock.option_chain(nearest_expiry)

            # Get current price from last available close price
            current_price = stock.history(period='1d')['Close'].iloc[-1]

            risk_free_rate = 0.02  # Approximate risk-free rate

            calls = chain.calls
            puts = chain.puts

            # Calculate option Greeks
            T = (datetime.strptime(nearest_expiry, '%Y-%m-%d') - datetime.now()).days / 365
            calls['delta'] = calls.apply(lambda x: self._calculate_delta(
                current_price, x['strike'], risk_free_rate, x['impliedVolatility'],
                T, option_type='call'), axis=1)

            # Volatility skew analysis
            atm_strike = calls['strike'].iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            skew = pd.DataFrame()
            skew['strike_distance'] = calls['strike'] - atm_strike.iloc[0]
            skew['iv'] = calls['impliedVolatility']

            self.options_analysis[ticker] = {
                'put_call_ratio': len(puts) / len(calls) if len(calls) != 0 else np.nan,
                'atm_call_iv': calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0],
                'atm_put_iv': puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0],
                'iv_skew': skew['iv'].max() - skew['iv'].min(),
                'avg_call_delta': calls['delta'].mean(),
                'call_volume_put_volume_ratio': calls['volume'].sum() / puts['volume'].sum() if puts['volume'].sum() > 0 else float('inf')
            }

        except Exception as e:
            print(f"Error in options analysis for {ticker}: {str(e)}")
            self.options_analysis[ticker] = None

    def _calculate_delta(self, S, K, r, sigma, T, option_type='call'):
        """Calculate option delta using Black-Scholes formula."""
        try:
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            if option_type == 'call':
                return norm.cdf(d1)
            return norm.cdf(d1) - 1
        except Exception:
            return np.nan

    def _generate_analysis_summary(self, ticker):
        """Generate a comprehensive analysis summary suitable for LLM input."""
        technical = self.technical_indicators.get(ticker, {})
        risk = self.risk_metrics.get(ticker, {})
        options = self.options_analysis.get(ticker, {})

        summary = {
            'ticker': ticker,
            'technical_analysis': technical,
            'risk_metrics': risk,
            'options_data': options,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Create natural language prompt
        nlp_prompt = f"""
        Based on the following market data for {ticker}, please provide:
        1. A comprehensive market analysis
        2. Key risk factors to consider
        3. Potential trading opportunities
        4. Recommended position sizing and risk management strategy

        Technical Indicators:
        - Price relative to SMA20: {technical.get('price_to_sma20', np.nan):.2%}
        - Price relative to SMA50: {technical.get('price_to_sma50', np.nan):.2%}
        - MACD: {technical.get('macd', np.nan):.4f}
        - ATR: {technical.get('atr', np.nan):.4f}

        Risk Metrics:
        - Annual Volatility: {risk.get('annual_volatility', np.nan):.2%}
        - Value at Risk (95%): {risk.get('var_95', np.nan):.2%}
        - Maximum Drawdown: {risk.get('max_drawdown', np.nan):.2%}
        - Sharpe Ratio: {risk.get('sharpe_ratio', np.nan):.2f}

        Options Analysis:
        - Put/Call Ratio: {options.get('put_call_ratio', np.nan)}
        - ATM Call IV: {options.get('atm_call_iv', np.nan):.2%}
        - IV Skew: {options.get('iv_skew', np.nan):.2%}
        - Average Call Delta: {options.get('avg_call_delta', np.nan):.2f}

        Suggested Position Size: {risk.get('position_size_suggest', np.nan):.2%} of portfolio

        Please analyze these metrics and provide actionable insights for a high-risk trading strategy.
        """

        summary['nlp_prompt'] = nlp_prompt
        return summary

class StockAnalyzerGUI:
    def __init__(self):
        self.analyzer = EnhancedStockAnalyzer()

    def run(self):
        st.set_page_config(layout="wide")
        st.title("Advanced Stock & Options Analysis Dashboard")

        # Sidebar for inputs
        self._create_sidebar()

        # Main content area
        if st.session_state.get('selected_ticker') and st.session_state.get('run_analysis'):
            self._show_analysis()

    def _create_sidebar(self):
        st.sidebar.header("Analysis Parameters")

        # Stock selection
        ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
        if ticker:
            st.session_state.selected_ticker = ticker

        # Time period selection
        st.sidebar.subheader("Time Period")
        start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=180))
        end_date = st.sidebar.date_input("End Date", datetime.now())

        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date.")
        else:
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

        # Analysis options
        st.sidebar.subheader("Analysis Options")
        st.session_state.show_technical = st.sidebar.checkbox("Show Technical Analysis", value=True)
        st.session_state.show_options = st.sidebar.checkbox("Show Options Analysis", value=True)
        st.session_state.show_risk = st.sidebar.checkbox("Show Risk Metrics", value=True)

        # Risk management settings
        st.sidebar.subheader("Risk Management")
        st.session_state.max_portfolio_risk = st.sidebar.slider(
            "Max Portfolio Risk (%):",
            min_value=1,
            max_value=10,
            value=2
        ) / 100

        # Run analysis button
        if st.sidebar.button("Run Analysis"):
            st.session_state.run_analysis = True

    def _show_analysis(self):
        try:
            # Get stock data
            analysis = self.analyzer.analyze_stock(
                st.session_state.selected_ticker,
                st.session_state.start_date,
                st.session_state.end_date
            )

            # Create tabs for different analysis sections
            tabs = st.tabs(["Overview", "Technical Analysis", "Options Analysis", "Risk Management", "AI Analysis"])

            # Overview Tab
            with tabs[0]:
                self._show_overview(analysis)

            # Technical Analysis Tab
            with tabs[1]:
                if st.session_state.show_technical:
                    self._show_technical_analysis(analysis)

            # Options Analysis Tab
            with tabs[2]:
                if st.session_state.show_options:
                    self._show_options_analysis(analysis)

            # Risk Management Tab
            with tabs[3]:
                if st.session_state.show_risk:
                    self._show_risk_analysis(analysis)

            # AI Analysis Tab
            with tabs[4]:
                self._show_ai_analysis(analysis)
        except Exception as e:
            st.error(f"Error analyzing {st.session_state.selected_ticker}: {str(e)}")

    def _show_overview(self, analysis):
        st.header(f"Overview - {st.session_state.selected_ticker}")

        # Add refresh button in its own column
        col_refresh = st.columns([0.8, 0.2])
        with col_refresh[1]:
            refresh = st.button("🔄 Refresh Data")

        # Get real-time price
        try:
            stock = yf.Ticker(st.session_state.selected_ticker)

            # If refresh button is clicked or no price in session state, fetch new data
            if refresh or 'current_price' not in st.session_state:
                # Get the last available close price
                current_data = stock.history(period='1d')
                if not current_data.empty:
                    st.session_state.current_price = current_data['Close'].iloc[-1]
                    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    price_change = (st.session_state.current_price / analysis['technical_analysis']['current_price']) - 1
                else:
                    st.warning("Could not fetch the latest price data")
                    st.session_state.current_price = analysis['technical_analysis']['current_price']
                    price_change = 0

            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Price",
                          f"${st.session_state.current_price:.2f}",
                          f"{price_change:.2%}")
                st.caption(f"Last updated: {st.session_state.last_update}")

            with col2:
                st.metric("Volatility",
                          f"{analysis['risk_metrics']['annual_volatility']:.2%}")

            with col3:
                st.metric("Suggested Position Size",
                          f"{analysis['risk_metrics']['position_size_suggest']:.2%}")

            # Price chart
            self._create_price_chart(st.session_state.selected_ticker)

        except Exception as e:
            st.error(f"Error fetching current price: {str(e)}")

    def _show_technical_analysis(self, analysis):
        st.header("Technical Analysis")

        # Technical indicators in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Moving Averages")
            st.write(f"SMA20: {analysis['technical_analysis']['sma_20']:.2f}")
            st.write(f"SMA50: {analysis['technical_analysis']['sma_50']:.2f}")
            st.write(f"Price to SMA20: {analysis['technical_analysis']['price_to_sma20']:.2%}")
            st.write(f"Price to SMA50: {analysis['technical_analysis']['price_to_sma50']:.2%}")

        with col2:
            st.subheader("Momentum Indicators")
            st.write(f"MACD: {analysis['technical_analysis']['macd']:.4f}")
            st.write(f"MACD Signal: {analysis['technical_analysis']['macd_signal']:.4f}")
            st.write(f"MACD Histogram: {analysis['technical_analysis']['macd_hist']:.4f}")
            st.write(f"ATR: {analysis['technical_analysis']['atr']:.4f}")

        # Technical charts
        self._create_technical_charts(st.session_state.selected_ticker)

    def _show_options_analysis(self, analysis):
        st.header("Options Analysis")

        if analysis['options_data']:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Options Metrics")
                st.write(f"Put/Call Ratio: {analysis['options_data']['put_call_ratio']:.2f}")
                st.write(f"ATM Call IV: {analysis['options_data']['atm_call_iv']:.2%}")
                st.write(f"ATM Put IV: {analysis['options_data']['atm_put_iv']:.2%}")
                st.write(f"IV Skew: {analysis['options_data']['iv_skew']:.2%}")

            with col2:
                st.subheader("Greeks")
                st.write(f"Average Call Delta: {analysis['options_data']['avg_call_delta']:.2f}")
                cpv = analysis['options_data']['call_volume_put_volume_ratio']
                cpv_display = f"{cpv:.2f}" if not np.isinf(cpv) else "Infinity"
                st.write(f"Volume Ratio (Call/Put): {cpv_display}")
        else:
            st.warning("No options data available for this stock")

    def _show_risk_analysis(self, analysis):
        st.header("Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Metrics")
            st.write(f"Annual Volatility: {analysis['risk_metrics']['annual_volatility']:.2%}")
            st.write(f"Value at Risk (95%): {analysis['risk_metrics']['var_95']:.2%}")
            st.write(f"Value at Risk (99%): {analysis['risk_metrics']['var_99']:.2%}")
            st.write(f"Maximum Drawdown: {analysis['risk_metrics']['max_drawdown']:.2%}")

        with col2:
            st.subheader("Performance Metrics")
            st.write(f"Sharpe Ratio: {analysis['risk_metrics']['sharpe_ratio']:.2f}")
            st.write(f"Suggested Position Size: {analysis['risk_metrics']['position_size_suggest']:.2%}")

    def _show_ai_analysis(self, analysis):
        st.header("AI Analysis Prompt")
        st.text_area("Copy this prompt for AI analysis:",
                     analysis['nlp_prompt'],
                     height=400)

    def _create_price_chart(self, ticker):
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=st.session_state.start_date, end=st.session_state.end_date)

            if hist.empty:
                st.warning("No price data available")
                return

            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                 open=hist['Open'],
                                                 high=hist['High'],
                                                 low=hist['Low'],
                                                 close=hist['Close'])])

            fig.update_layout(
                title=f"{ticker} Price Chart",
                yaxis_title="Price",
                xaxis_title="Date",
                height=600,
                template="plotly_white",
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")

    def _create_technical_charts(self, ticker):
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=st.session_state.start_date, end=st.session_state.end_date)

            if hist.empty:
                st.warning("No data available for technical analysis")
                return

            # Create subplots
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Price and Moving Averages", "MACD"),
                                row_heights=[0.7, 0.3],
                                vertical_spacing=0.1)

            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Price"
            ), row=1, col=1)

            # Add moving averages
            sma20 = hist['Close'].rolling(window=20).mean()
            sma50 = hist['Close'].rolling(window=50).mean()

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=sma20,
                name="SMA20",
                line=dict(color='orange')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=sma50,
                name="SMA50",
                line=dict(color='blue')
            ), row=1, col=1)

            # Calculate and add MACD
            ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
            ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=macd,
                name="MACD",
                line=dict(color='blue')
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=signal,
                name="Signal",
                line=dict(color='orange')
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=hist.index,
                y=histogram,
                name="Histogram",
                marker_color=np.where(histogram >= 0, 'green', 'red')
            ), row=2, col=1)

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template="plotly_white",
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating technical charts: {str(e)}")

def main():
    app = StockAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main()
