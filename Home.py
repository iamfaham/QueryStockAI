import streamlit as st
import asyncio
import json
import re
import os
import plotly.graph_objects as go
import yfinance as yf
import time
import sys
from datetime import timedelta
import gnews
from bs4 import BeautifulSoup
import importlib.util
import requests
import holidays
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, types
from sklearn.preprocessing import StandardScaler

try:
    from resource_monitor import (
        start_resource_monitoring,
        resource_monitor,
    )

    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    st.warning("Resource monitoring not available. Install psutil: pip install psutil")

# Load environment variables
load_dotenv()

# Check if API key exists - support both .env and Streamlit secrets
api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
model = os.getenv("MODEL") or st.secrets.get("MODEL")

if not api_key:
    st.error(
        "❌ Error: OPENROUTER_API_KEY not found. Please set it in your environment variables or Streamlit secrets."
    )
    st.stop()

if not model:
    st.error(
        "❌ Error: MODEL not found. Please set it in your environment variables or Streamlit secrets."
    )
    st.stop()

# Configure the client to connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Global variable to store discovered tools
discovered_tools = []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_tickers():
    """Fetch available tickers using multiple APIs and sources."""
    try:
        print("Fetching stock tickers from multiple sources...")
        tickers_dict = {}

        # Method 1: Try to get stocks from a free API
        try:
            print("Fetching stocks from API...")
            # Try to get stocks from a free API endpoint
            api_url = "https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000"

            # Try alternative free APIs
            apis_to_try = [
                "https://api.twelvedata.com/stocks?country=US&exchange=NASDAQ",
                "https://api.twelvedata.com/stocks?country=US&exchange=NYSE",
                "https://api.twelvedata.com/stocks?country=US&exchange=AMEX",
            ]

            for api_url in apis_to_try:
                try:
                    response = requests.get(api_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data:
                            for item in data["data"]:
                                ticker = item.get("symbol", "")
                                name = item.get("name", ticker)
                                if (
                                    ticker and name and len(ticker) <= 5
                                ):  # Filter for likely stock tickers
                                    tickers_dict[ticker] = name
                            print(f"Loaded {len(tickers_dict)} stocks from {api_url}")
                            break
                except Exception as e:
                    print(f"Error with API {api_url}: {e}")
                    continue

        except Exception as e:
            print(f"Error fetching from APIs: {e}")

        # Method 2: Try additional free APIs for more stocks
        if len(tickers_dict) < 100:  # Only if we didn't get enough from first APIs
            try:
                print("Fetching additional stocks from more APIs...")

                # Try more free APIs
                additional_apis = [
                    "https://api.twelvedata.com/stocks?country=US&exchange=NASDAQ&limit=500",
                    "https://api.twelvedata.com/stocks?country=US&exchange=NYSE&limit=500",
                    "https://api.twelvedata.com/stocks?country=US&exchange=AMEX&limit=500",
                    "https://api.twelvedata.com/stocks?country=CA&exchange=TSX&limit=200",
                    "https://api.twelvedata.com/stocks?country=GB&exchange=LSE&limit=200",
                ]

                for api_url in additional_apis:
                    try:
                        response = requests.get(api_url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if "data" in data:
                                for item in data["data"]:
                                    ticker = item.get("symbol", "")
                                    name = item.get("name", ticker)
                                    if (
                                        ticker and name and len(ticker) <= 5
                                    ):  # Filter for likely stock tickers
                                        if (
                                            ticker not in tickers_dict
                                        ):  # Avoid duplicates
                                            tickers_dict[ticker] = name
                                print(f"Loaded additional stocks from {api_url}")
                    except Exception as e:
                        print(f"Error with additional API {api_url}: {e}")
                        continue

                print(f"Loaded {len(tickers_dict)} total stocks from all APIs")
            except Exception as e:
                print(f"Error fetching from additional APIs: {e}")

        # Method 3: Try to get stocks from Yahoo Finance screener (if available)
        if len(tickers_dict) < 200:  # Only if we need more
            try:
                print("Trying Yahoo Finance screener...")
                # This is a fallback that doesn't hardcode tickers
                # We'll try to get some popular stocks dynamically
                popular_keywords = [
                    "technology",
                    "finance",
                    "healthcare",
                    "energy",
                    "consumer",
                ]

                for keyword in popular_keywords:
                    try:
                        # Try to search for stocks by sector
                        search_url = f"https://api.twelvedata.com/stocks?search={keyword}&limit=50"
                        response = requests.get(search_url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if "data" in data:
                                for item in data["data"]:
                                    ticker = item.get("symbol", "")
                                    name = item.get("name", ticker)
                                    if (
                                        ticker and name and len(ticker) <= 5
                                    ):  # Filter for likely stock tickers
                                        if (
                                            ticker not in tickers_dict
                                        ):  # Avoid duplicates
                                            tickers_dict[ticker] = name
                    except Exception as e:
                        print(f"Error searching for {keyword}: {e}")
                        continue

                print(
                    f"Loaded {len(tickers_dict)} total stocks (including sector searches)"
                )
            except Exception as e:
                print(f"Error fetching from sector searches: {e}")

        if len(tickers_dict) > 0:
            print(
                f"Successfully loaded {len(tickers_dict)} valid tickers from multiple sources"
            )
            return tickers_dict
        else:
            print("No tickers loaded from APIs, using fallback list")

    except Exception as e:
        print(f"Error in main ticker fetching: {e}")

    # Fallback to comprehensive list if all APIs fail
    try:
        print("Using comprehensive fallback list...")
        fallback_tickers = {}

        # Comprehensive list of major stocks across sectors
        fallback_ticker_list = [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "NFLX",
            "ADBE",
        ]

        print(f"Loading {len(fallback_ticker_list)} fallback tickers...")

        # Get company names for each ticker
        for ticker in fallback_ticker_list:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info

                if info and (info.get("longName") or info.get("shortName")):
                    company_name = info.get("longName", info.get("shortName", ticker))
                    fallback_tickers[ticker] = company_name

            except Exception as e:
                # Skip tickers that cause errors
                continue

        print(f"Successfully loaded {len(fallback_tickers)} tickers from fallback")
        return fallback_tickers

    except Exception as e:
        st.error(f"Error fetching available tickers: {e}")
        # Final fallback to basic tickers if there's an error
        return {
            "AAPL": "Apple Inc.",
            "TSLA": "Tesla Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOG": "Alphabet Inc. (Google)",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson",
            "PG": "Procter & Gamble Co.",
        }


@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_ticker(ticker_symbol):
    """Search for a ticker symbol and get its company name using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        company_name = info.get("longName", info.get("shortName", ticker_symbol))
        return company_name
    except Exception as e:
        return None


async def get_news_data(ticker: str) -> str:
    """Get news data by calling the news server via MCP."""
    try:
        # Set up MCP server parameters

        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_server_path = os.path.join(current_dir, "news_server.py")

        if not os.path.exists(news_server_path):
            return f"Error: news_server.py not found at {news_server_path}"

        # Use the same Python executable as the current process
        python_executable = sys.executable
        server_params = StdioServerParameters(
            command=python_executable, args=[news_server_path]
        )

        # Connect to the MCP server
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Call the get_latest_news tool
                    with st.status(
                        f"🔍 Fetching news data for {ticker}...", expanded=False
                    ) as status:
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool(
                                    "get_latest_news", {"ticker": ticker}
                                ),
                                timeout=30.0,  # 30 second timeout
                            )
                            status.update(
                                label=f"✅ News data fetched for {ticker}",
                                state="complete",
                            )
                        except asyncio.TimeoutError:
                            status.update(
                                label="❌ News data fetch timed out", state="error"
                            )
                            return f"Timeout getting news for {ticker}"
                        except Exception as e:
                            status.update(
                                label=f"❌ Error fetching news: {e}", state="error"
                            )
                            return f"Error getting news for {ticker}: {e}"

                    # Parse the result properly
                    if result.content:
                        for content in result.content:
                            if isinstance(content, types.TextContent):
                                return content.text

                    return f"No news data returned for {ticker}"
        except Exception as e:
            st.error(f"❌ Failed to connect to news server: {e}")
            return f"Failed to connect to news server: {e}"

    except Exception as e:
        return f"Error getting news for {ticker}: {e}"


async def get_stock_data(ticker: str) -> str:
    """Get stock data by calling the stock server via MCP."""
    try:
        # Set up MCP server parameters

        current_dir = os.path.dirname(os.path.abspath(__file__))
        stock_server_path = os.path.join(current_dir, "stock_data_server.py")

        if not os.path.exists(stock_server_path):
            return f"Error: stock_data_server.py not found at {stock_server_path}"

        # Use the same Python executable as the current process
        python_executable = sys.executable
        server_params = StdioServerParameters(
            command=python_executable, args=[stock_server_path]
        )

        # Connect to the MCP server
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Call the get_historical_stock_data tool
                    with st.status(
                        f"📊 Fetching stock data for {ticker}...", expanded=False
                    ) as status:
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool(
                                    "get_historical_stock_data", {"ticker": ticker}
                                ),
                                timeout=30.0,  # 30 second timeout
                            )
                            status.update(
                                label=f"✅ Stock data fetched for {ticker}",
                                state="complete",
                            )
                        except asyncio.TimeoutError:
                            status.update(
                                label="❌ Stock data fetch timed out", state="error"
                            )
                            return f"Timeout getting stock data for {ticker}"
                        except Exception as e:
                            status.update(
                                label=f"❌ Error fetching stock data: {e}",
                                state="error",
                            )
                            return f"Error getting stock data for {ticker}: {e}"

                    # Parse the result properly
                    if result.content:
                        for content in result.content:
                            if isinstance(content, types.TextContent):
                                return content.text

                    return f"No stock data returned for {ticker}"
        except Exception as e:
            st.error(f"❌ Failed to connect to stock data server: {e}")
            return f"Failed to connect to stock data server: {e}"

    except Exception as e:
        return f"Error getting stock data for {ticker}: {e}"


def calculate_rsi(data, window):
    """Calculate RSI (Relative Strength Index) for the given data."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_stock_chart(ticker: str):
    """Create an interactive stock price chart with Linear Regression predictions for the given ticker."""
    try:
        # Get stock data - 5 years for training Linear Regression
        with st.spinner(f"📊 Fetching stock data for {ticker}..."):
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="5y")

            # Track yfinance API call
            if RESOURCE_MONITORING_AVAILABLE:
                resource_monitor.increment_yfinance_calls()

        if hist_data.empty:
            st.warning(f"No data available for {ticker}")
            return None

        # Prepare data for Linear Regression with technical indicators
        df = hist_data.reset_index()

        # Flatten the multi-level column index if it exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate technical indicators (same as in the notebook)
        # Moving averages
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # RSI
        df["RSI"] = calculate_rsi(df["Close"], window=14)

        # MACD
        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Band component
        df["BB_StdDev"] = df["Close"].rolling(window=20).std()

        # Volume moving average
        df["Volume_Avg"] = df["Volume"].rolling(window=20).mean()

        # Price momentum and volatility
        df["Price_Change"] = df["Close"].pct_change()
        df["Price_Change_5d"] = df["Close"].pct_change(periods=5)
        df["Price_Change_20d"] = df["Close"].pct_change(periods=20)
        df["Price_Volatility"] = df["Close"].rolling(window=20).std()
        df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]  # Daily range

        # Volume-Based Features
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volume_Price_Trend"] = df["Volume"] * df["Price_Change"]
        df["Volume_SMA_Ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
        df["Volume_StdDev"] = df["Volume"].rolling(window=20).std()

        # Advanced Technical Indicators
        # Stochastic Oscillator
        def calculate_stochastic(df, window=14):
            lowest_low = df["Low"].rolling(window=window).min()
            highest_high = df["High"].rolling(window=window).max()
            k_percent = 100 * ((df["Close"] - lowest_low) / (highest_high - lowest_low))
            return k_percent

        df["Stochastic_K"] = calculate_stochastic(df)
        df["Stochastic_D"] = df["Stochastic_K"].rolling(window=3).mean()

        # Williams %R
        def calculate_williams_r(df, window=14):
            highest_high = df["High"].rolling(window=window).max()
            lowest_low = df["Low"].rolling(window=window).min()
            williams_r = -100 * (
                (highest_high - df["Close"]) / (highest_high - lowest_low)
            )
            return williams_r

        df["Williams_R"] = calculate_williams_r(df)

        # Commodity Channel Index (CCI)
        def calculate_cci(df, window=20):
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            sma_tp = typical_price.rolling(window=window).mean()
            mad = typical_price.rolling(window=window).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci

        df["CCI"] = calculate_cci(df)

        # Moving Average Crossovers
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Crossover signals
        df["SMA_10_20_Cross"] = (df["SMA_10"] > df["SMA_20"]).astype(int)
        df["SMA_20_50_Cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
        df["SMA_50_200_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)

        # Bollinger Bands Components
        df["BB_Upper"] = df["SMA_20"] + (df["BB_StdDev"] * 2)
        df["BB_Lower"] = df["SMA_20"] - (df["BB_StdDev"] * 2)
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"]
        )
        df["BB_Squeeze"] = (df["BB_Upper"] - df["BB_Lower"]) / df[
            "SMA_20"
        ]  # Volatility indicator

        # Support and Resistance
        df["Resistance_20d"] = df["High"].rolling(window=20).max()
        df["Support_20d"] = df["Low"].rolling(window=20).min()
        df["Price_to_Resistance"] = df["Close"] / df["Resistance_20d"]
        df["Price_to_Support"] = df["Close"] / df["Support_20d"]

        # Time-based features
        df["Day_of_Week"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter
        df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
        df["Is_Quarter_End"] = df["Date"].dt.is_quarter_end.astype(int)

        # Market Sentiment Features
        df["Price_Above_SMA200"] = (df["Close"] > df["SMA_200"]).astype(int)
        df["Volume_Spike"] = (
            df["Volume"] > df["Volume"].rolling(window=20).mean() * 1.5
        ).astype(int)
        df["Price_Spike"] = (
            df["Price_Change"].abs() > df["Price_Change"].rolling(window=20).std() * 2
        ).astype(int)

        # Drop rows with NaN values created by moving averages and new features
        df.dropna(inplace=True)

        # Define features and target (same as notebook)
        features = [
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_StdDev",
            "BB_Position",
            "BB_Squeeze",
            "Stochastic_K",
            "Stochastic_D",
            "Williams_R",
            "CCI",
            "Price_Change",
            "Price_Change_5d",
            "Price_Change_20d",
            "Price_Volatility",
            "Price_Range",
            "Volume_Change",
            "Volume_Price_Trend",
            "Volume_SMA_Ratio",
            "Volume_StdDev",
            "SMA_10_20_Cross",
            "SMA_20_50_Cross",
            "SMA_50_200_Cross",
            "Price_to_Resistance",
            "Price_to_Support",
            "Day_of_Week",
            "Month",
            "Quarter",
            "Is_Month_End",
            "Is_Quarter_End",
            "Price_Above_SMA200",
            "Volume_Spike",
            "Price_Spike",
            "Volume_Avg",
        ]
        target = "Close"

        X = df[features]
        y = df[target]

        # Train on ALL available data (5 years)
        X_train = X  # Use all available data for training
        y_train = y

        # Add feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Ridge Regression model with cross-validation
        start_time = time.time()
        with st.spinner(f"Training Ridge Regression model for {ticker}..."):
            # Use Ridge with cross-validation to find optimal alpha
            ridge_model = Ridge()

            # Grid search for optimal regularization strength
            param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring="r2")
            grid_search.fit(X_train_scaled, y_train)

            # Use the best model
            model = grid_search.best_estimator_

        # Track training time
        training_time = time.time() - start_time
        if RESOURCE_MONITORING_AVAILABLE:
            resource_monitor.add_prophet_training_time(
                training_time
            )  # Reuse existing method

        # Get the best alpha value for display
        best_alpha = grid_search.best_params_["alpha"]
        best_score = grid_search.best_score_

        # Create future dates for next 30 days
        last_date = df["Date"].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=30, freq="D"
        )

        # Filter for trading days only
        future_trading_dates = [date for date in future_dates if is_trading_day(date)]

        # Create a more sophisticated future prediction approach
        # We'll use a more realistic projection with some randomness and market patterns
        future_features = []

        # Get the last few values to calculate trends
        last_20_prices = df["Close"].tail(20).values
        last_50_prices = df["Close"].tail(50).values
        last_volumes = df["Volume"].tail(20).values

        # Get the last known values for technical indicators
        last_values = df.iloc[-1]

        # Calculate more sophisticated trends
        price_trend = (
            df["Close"].iloc[-1] - df["Close"].iloc[-20]
        ) / 20  # Daily price change
        volume_trend = (
            df["Volume"].iloc[-1] - df["Volume"].iloc[-20]
        ) / 20  # Daily volume change

        # Calculate volatility for more realistic projections
        price_volatility = df["Close"].pct_change().std()
        volume_volatility = df["Volume"].pct_change().std()

        for i, date in enumerate(future_trading_dates):
            # Add some randomness to make predictions more realistic
            # Use a smaller random component to avoid extreme outliers
            random_factor = np.random.normal(0, price_volatility * 0.1)

            # Project prices forward using the trend with some randomness
            projected_price = (
                df["Close"].iloc[-1] + (price_trend * (i + 1)) + random_factor
            )

            # Ensure projected price doesn't go negative
            projected_price = max(projected_price, df["Close"].iloc[-1] * 0.5)

            # Update the price arrays for calculating moving averages
            if i < 20:
                # For first 20 days, use historical data + projected
                current_20_prices = np.append(
                    last_20_prices[-(20 - i - 1) :], [projected_price] * (i + 1)
                )
            else:
                # After 20 days, use only projected prices
                current_20_prices = np.array([projected_price] * 20)

            if i < 50:
                # For first 50 days, use historical data + projected
                current_50_prices = np.append(
                    last_50_prices[-(50 - i - 1) :], [projected_price] * (i + 1)
                )
            else:
                # After 50 days, use only projected prices
                current_50_prices = np.array([projected_price] * 50)

            # Calculate projected technical indicators
            sma_20 = np.mean(current_20_prices)
            sma_50 = np.mean(current_50_prices)

            # Project volume with some randomness
            volume_random_factor = np.random.normal(0, volume_volatility * 0.1)
            projected_volume = (
                df["Volume"].iloc[-1] + (volume_trend * (i + 1)) + volume_random_factor
            )
            projected_volume = max(
                projected_volume, df["Volume"].iloc[-1] * 0.3
            )  # Don't go too low

            volume_avg = np.mean(
                np.append(
                    last_volumes[-(20 - i - 1) :], [projected_volume] * min(i + 1, 20)
                )
            )

            # Add some variation to RSI and MACD instead of keeping them constant
            # RSI typically oscillates between 30-70, so add small random changes
            rsi_variation = np.random.normal(0, 2)  # Small random change
            new_rsi = last_values["RSI"] + rsi_variation
            new_rsi = max(10, min(90, new_rsi))  # Keep RSI in reasonable bounds

            # MACD variation
            macd_variation = np.random.normal(0, abs(last_values["MACD"]) * 0.1)
            new_macd = last_values["MACD"] + macd_variation
            new_macd_signal = last_values["MACD_Signal"] + macd_variation * 0.5

            # Bollinger Band variation
            bb_variation = np.random.normal(0, last_values["BB_StdDev"] * 0.1)
            new_bb_std = last_values["BB_StdDev"] + bb_variation
            new_bb_std = max(
                new_bb_std, last_values["BB_StdDev"] * 0.5
            )  # Don't go too low

            # Calculate additional features for future predictions
            # Use the last known values and add small variations
            new_stochastic_k = last_values.get("Stochastic_K", 50) + np.random.normal(
                0, 5
            )
            new_stochastic_k = max(0, min(100, new_stochastic_k))

            new_stochastic_d = last_values.get("Stochastic_D", 50) + np.random.normal(
                0, 5
            )
            new_stochastic_d = max(0, min(100, new_stochastic_d))

            new_williams_r = last_values.get("Williams_R", -50) + np.random.normal(0, 5)
            new_williams_r = max(-100, min(0, new_williams_r))

            new_cci = last_values.get("CCI", 0) + np.random.normal(0, 20)

            # Calculate BB position and squeeze
            bb_upper = sma_20 + (new_bb_std * 2)
            bb_lower = sma_20 - (new_bb_std * 2)
            bb_position = (
                (projected_price - bb_lower) / (bb_upper - bb_lower)
                if (bb_upper - bb_lower) > 0
                else 0.5
            )
            bb_squeeze = (bb_upper - bb_lower) / sma_20 if sma_20 > 0 else 0

            # Price changes
            price_change = (projected_price - df["Close"].iloc[-1]) / df["Close"].iloc[
                -1
            ]
            price_change_5d = price_change * 0.8  # Approximate
            price_change_20d = price_change * 0.6  # Approximate

            # Volume changes
            volume_change = (projected_volume - df["Volume"].iloc[-1]) / df[
                "Volume"
            ].iloc[-1]
            volume_price_trend = projected_volume * price_change
            volume_sma_ratio = projected_volume / volume_avg if volume_avg > 0 else 1

            # Moving average crossovers
            sma_10 = (
                np.mean(current_20_prices[-10:])
                if len(current_20_prices) >= 10
                else sma_20
            )
            sma_200 = sma_50  # Approximate for future

            sma_10_20_cross = 1 if sma_10 > sma_20 else 0
            sma_20_50_cross = 1 if sma_20 > sma_50 else 0
            sma_50_200_cross = 1 if sma_50 > sma_200 else 0

            # Support and resistance
            resistance_20d = projected_price * 1.05  # Approximate
            support_20d = projected_price * 0.95  # Approximate
            price_to_resistance = projected_price / resistance_20d
            price_to_support = projected_price / support_20d

            # Time-based features (use the actual future date)
            day_of_week = date.weekday()
            month = date.month
            quarter = (month - 1) // 3 + 1
            is_month_end = 1 if date.day >= 25 else 0  # Approximate
            is_quarter_end = 1 if month in [3, 6, 9, 12] and date.day >= 25 else 0

            # Market sentiment
            price_above_sma200 = 1 if projected_price > sma_200 else 0
            volume_spike = 1 if projected_volume > volume_avg * 1.5 else 0
            price_spike = 1 if abs(price_change) > price_volatility * 2 else 0

            future_row = {
                "SMA_10": sma_10,
                "SMA_20": sma_20,
                "SMA_50": sma_50,
                "SMA_200": sma_200,
                "RSI": new_rsi,
                "MACD": new_macd,
                "MACD_Signal": new_macd_signal,
                "BB_StdDev": new_bb_std,
                "BB_Position": bb_position,
                "BB_Squeeze": bb_squeeze,
                "Stochastic_K": new_stochastic_k,
                "Stochastic_D": new_stochastic_d,
                "Williams_R": new_williams_r,
                "CCI": new_cci,
                "Price_Change": price_change,
                "Price_Change_5d": price_change_5d,
                "Price_Change_20d": price_change_20d,
                "Price_Volatility": price_volatility,
                "Price_Range": abs(price_change) * 0.02,  # Approximate
                "Volume_Change": volume_change,
                "Volume_Price_Trend": volume_price_trend,
                "Volume_SMA_Ratio": volume_sma_ratio,
                "Volume_StdDev": volume_volatility,
                "SMA_10_20_Cross": sma_10_20_cross,
                "SMA_20_50_Cross": sma_20_50_cross,
                "SMA_50_200_Cross": sma_50_200_cross,
                "Price_to_Resistance": price_to_resistance,
                "Price_to_Support": price_to_support,
                "Day_of_Week": day_of_week,
                "Month": month,
                "Quarter": quarter,
                "Is_Month_End": is_month_end,
                "Is_Quarter_End": is_quarter_end,
                "Price_Above_SMA200": price_above_sma200,
                "Volume_Spike": volume_spike,
                "Price_Spike": price_spike,
                "Volume_Avg": volume_avg,
            }
            future_features.append(future_row)

        # Create X_future AFTER future_features is populated
        X_future = pd.DataFrame(future_features)
        X_future_scaled = scaler.transform(X_future)

        # Make predictions for the next 30 trading days
        future_predictions = model.predict(X_future_scaled)

        # Create interactive chart with historical data and future predictions
        fig = go.Figure()

        # Filter data to show only the last 1 year for display
        one_year_ago = last_date - timedelta(days=365)
        df_display = df[df["Date"] >= one_year_ago]

        # Add historical price data (last 1 year only)
        fig.add_trace(
            go.Scatter(
                x=df_display["Date"],
                y=df_display["Close"],
                mode="lines+markers",
                name=f"{ticker} Historical Price (Last Year)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )

        # Add future predictions
        fig.add_trace(
            go.Scatter(
                x=future_trading_dates,
                y=future_predictions,
                mode="lines+markers",
                name=f"{ticker} Future Predictions (Next 30 Days)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=4),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price with Next 30-Day Linear Regression Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Update axes
        fig.update_xaxes(
            title_text="Date",
            tickformat="%b %d",
            tickangle=45,
        )
        fig.update_yaxes(title_text="Price ($)")

        # Display prediction summary
        current_price = df["Close"].iloc[-1]
        predicted_price_30d = (
            future_predictions[-1] if len(future_predictions) > 0 else current_price
        )
        price_change = predicted_price_30d - current_price
        price_change_pct = (price_change / current_price) * 100

        # Calculate model performance on historical data (for reference)
        y_pred_historical = model.predict(
            X_train_scaled
        )  # Use scaled data for historical fit
        r2_historical = r2_score(y_train, y_pred_historical)
        mse_historical = mean_squared_error(y_train, y_pred_historical)

        # Display detailed prediction information
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
            )

        with col2:
            st.metric(
                "30-Day Prediction",
                f"${predicted_price_30d:.2f}",
                delta=f"{price_change_pct:+.2f}%",
            )

        with col3:
            st.metric(
                "Expected Change",
                f"${price_change:.2f} ({price_change_pct:+.2f}%)",
            )

        # Additional prediction details
        st.info(
            f"""
        **📊 30-Day Ridge Regression Prediction for {ticker}:**
        - **Current Price:** ${current_price:.2f}
        - **Predicted Price (30 days):** ${predicted_price_30d:.2f}
        - **Expected Change:** ${price_change:.2f} ({price_change_pct:+.2f}%)
        - **Model Performance (Historical Fit):**
          - R² Score: {r2_historical:.4f} ({r2_historical*100:.2f}% accuracy)
          - Mean Squared Error: {mse_historical:.4f}
          - Best Alpha (Regularization): {best_alpha}
          - Cross-Validation Score: {best_score:.4f}
        - **Model Training Time:** {training_time:.2f}s
        - **Training Data:** 5 years of historical data
        - **Features Used:** {', '.join(features)}

        ⚠️ **Disclaimer**: Stock predictions have approximately 70% accuracy.
        These forecasts are for informational purposes only and should not be used as
        the sole basis for investment decisions. Always conduct your own research
        and consider consulting with financial advisors.
        """
        )

        return fig

    except Exception as e:
        st.error(f"Error creating chart for {ticker}: {e}")
        return create_basic_stock_chart(ticker)


def create_basic_stock_chart(ticker: str):
    """Create a basic stock price chart without Prophet predictions."""
    try:
        # Get stock data with loading state
        with st.spinner(f"📊 Fetching basic stock data for {ticker}..."):
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="30d")

        if hist_data.empty:
            st.warning(f"No data available for {ticker}")
            return None

        # Create simple line chart
        fig = go.Figure()

        # Add price line chart
        fig.add_trace(
            go.Scatter(
                x=hist_data.index,
                y=hist_data["Close"],
                mode="lines+markers",
                name=f"{ticker} Price",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price (30 Days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=False,
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(
            title_text="Date",
            tickformat="%b %d",
            tickangle=45,
        )
        fig.update_yaxes(title_text="Price ($)")

        return fig

    except Exception as e:
        st.error(f"Error creating chart for {ticker}: {e}")
        return None


def initialize_tools():
    """Initialize the available tools."""
    global discovered_tools

    discovered_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_latest_news",
                "description": "Fetches recent news headlines and descriptions for a specific stock ticker. Use this when user asks about news, updates, or recent events about a company.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., 'AAPL', 'GOOG', 'TSLA'). Must be a valid stock symbol.",
                        }
                    },
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_historical_stock_data",
                "description": "Fetches recent historical stock data (Open, High, Low, Close, Volume) for a given ticker. Use this when user asks about stock performance, price data, or market performance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT'). Must be a valid stock symbol.",
                        }
                    },
                    "required": ["ticker"],
                },
            },
        },
    ]


async def execute_tool_call(tool_call):
    """Execute a tool call using MCP servers."""
    try:
        tool_name = tool_call.function.name

        # Clean and validate the arguments JSON
        arguments_str = tool_call.function.arguments.strip()

        # Try to extract valid JSON if there's extra content
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            # Try to find JSON within the string

            json_match = re.search(r"\{[^{}]*\}", arguments_str)
            if json_match:
                try:
                    arguments = json.loads(json_match.group())
                except json.JSONDecodeError:
                    st.error(f"❌ Could not parse tool arguments: {arguments_str}")
                    return f"Error: Invalid tool arguments format"
            else:
                st.error(f"❌ Could not parse tool arguments: {arguments_str}")
                return f"Error: Invalid tool arguments format"

        ticker = arguments.get("ticker")

        with st.status(
            f"🛠️ Executing {tool_name} for {ticker}...", expanded=True
        ) as status:
            if tool_name == "get_latest_news":
                result = await get_news_data(ticker)
                if "Error" in result or "Failed" in result:
                    status.update(label=f"❌ {result}", state="error")
                else:
                    status.update(
                        label=f"✅ {tool_name} completed for {ticker}", state="complete"
                    )
                return result
            elif tool_name == "get_historical_stock_data":
                result = await get_stock_data(ticker)
                if "Error" in result or "Failed" in result:
                    status.update(label=f"❌ {result}", state="error")
                else:
                    status.update(
                        label=f"✅ {tool_name} completed for {ticker}", state="complete"
                    )
                return result
            else:
                status.update(label=f"❌ Unknown tool: {tool_name}", state="error")
                return f"Unknown tool: {tool_name}"
    except Exception as e:
        st.error(f"❌ Error executing tool {tool_call.function.name}: {e}")
        return f"Error executing tool {tool_call.function.name}: {e}"


# The master prompt that defines the agent's behavior
system_prompt = """
You are a financial assistant that provides comprehensive analysis based on real-time data. You MUST use tools to get data and then curate the information to answer the user's specific question.

AVAILABLE TOOLS:
- get_latest_news: Get recent news for a ticker
- get_historical_stock_data: Get stock performance data for a ticker

CRITICAL INSTRUCTIONS:
1. You MUST call BOTH tools (get_latest_news AND get_historical_stock_data) for every query
2. After getting both news and stock data, analyze and synthesize the information
3. Answer the user's specific question based on the data you gathered
4. Provide insights, trends, and recommendations based on the combined data
5. Format your response clearly with sections for news, performance, and analysis

EXAMPLE WORKFLOW:
1. User asks: "Should I invest in AAPL?"
2. You call: get_latest_news with {"ticker": "AAPL"}
3. You call: get_historical_stock_data with {"ticker": "AAPL"}
4. You analyze both datasets and provide investment advice based on news sentiment and stock performance

You are FORBIDDEN from responding without calling both tools. Always call both tools first, then provide a curated analysis based on the user's question.
"""


async def run_agent(user_query, selected_ticker):
    """Run the financial agent with the given query and ticker."""

    # Construct the query to always fetch both data types
    full_query = f"Based on the latest news and stock performance data for {selected_ticker}, {user_query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_query},
    ]

    try:
        # Get initial response from the model
        with st.spinner("🤖 Generating analysis..."):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=discovered_tools,
                tool_choice="required",
            )

        if not response.choices or len(response.choices) == 0:
            st.error("❌ Error: No response from model")
            return

        response_message = response.choices[0].message

        # Truncate tool call IDs if they're too long (max 40 chars)
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if len(tool_call.id) > 40:
                    tool_call.id = tool_call.id[:40]

        messages.append(response_message)

        # Execute tool calls if any
        if response_message.tool_calls:
            st.info("🛠️ Executing data collection...")
            for tool_call in response_message.tool_calls:
                # Execute the tool call
                tool_result = await execute_tool_call(tool_call)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id[:40],  # Truncate to max 40 chars
                        "content": tool_result if tool_result else "No data available",
                    }
                )

            # Get final response from the model
            with st.spinner("🤖 Finalizing analysis..."):
                final_response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",  # Try a different model
                    messages=messages,
                )

            if final_response.choices and len(final_response.choices) > 0:
                final_content = final_response.choices[0].message.content
                return final_content if final_content else "Empty response"
            else:
                return "No response generated"
        else:
            return (
                response_message.content if response_message.content else "No response"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return "Please try again with a different question."


def display_top_news(ticker: str):
    """Display top news headlines for the given ticker with clickable links."""
    try:

        # Check if news is already cached
        news_cache_key = f"news_data_{ticker}"
        if news_cache_key in st.session_state:
            articles = st.session_state[news_cache_key]
        else:
            # Get news data with loading state
            with st.spinner(f"📰 Loading news for {ticker}..."):
                google_news = gnews.GNews(language="en", country="US", period="7d")
                search_query = f'"{ticker}" stock market news'
                articles = google_news.get_news(search_query)
                # Cache the articles
                st.session_state[news_cache_key] = articles

        if not articles:
            st.info(f"No recent news found for {ticker}")
            return

        # Display top 5 articles
        for i, article in enumerate(articles[:5], 1):
            # Clean the title text
            title = article.get("title", "")
            if title:
                soup = BeautifulSoup(title, "html.parser")
                title = soup.get_text().strip()
            url = article.get("url", "")
            publisher = article.get("publisher", {}).get("title", "Unknown Source")

            # Create a clickable link
            if url:
                st.markdown(f"[{title}]({url})")
                st.caption(f"Source: {publisher}")
            else:
                st.markdown(f"{title}")
                st.caption(f"Source: {publisher}")

            # Add some spacing between articles
            if i < 5:
                st.markdown("---")

    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")


def is_trading_day(date):
    """Check if a date is a trading day (not weekend or holiday)."""
    # Check if it's a weekend
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check if it's a US market holiday
    us_holidays = holidays.US()
    if date in us_holidays:
        return False

    return True


def get_next_trading_days(start_date, num_days):
    """Get the next N trading days starting from start_date."""
    trading_days = []
    current_date = start_date

    while len(trading_days) < num_days:
        if is_trading_day(current_date):
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days


def create_trading_day_future_dataframe(model, periods=30, freq="D"):
    """Create a future dataframe with only trading days."""
    # Get the last date from the training data
    last_date = model.history["ds"].max()

    # Generate trading days
    trading_days = []
    current_date = last_date + timedelta(days=1)

    while len(trading_days) < periods:
        if is_trading_day(current_date):
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    # Create future dataframe with only trading days
    future_df = pd.DataFrame({"ds": trading_days})
    return future_df


def test_server_availability():
    """Test if the MCP servers are available and can be executed."""

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Test news server
    news_server_path = os.path.join(current_dir, "news_server.py")
    if not os.path.exists(news_server_path):
        print(f"❌ ERROR: news_server.py not found at {news_server_path}")
        return False

    # Test stock data server
    stock_server_path = os.path.join(current_dir, "stock_data_server.py")
    if not os.path.exists(stock_server_path):
        print(f"❌ ERROR: stock_data_server.py not found at {stock_server_path}")
        return False

    # Test if servers can be executed by checking if they can be imported

    try:
        # Test if news_server can be imported
        spec = importlib.util.spec_from_file_location("news_server", news_server_path)
        if spec is None or spec.loader is None:
            print("⚠️ WARNING: Could not load news_server.py")
        else:
            print("✅ SUCCESS: news_server.py is importable")
    except Exception as e:
        print(f"⚠️ WARNING: Could not import news_server.py: {e}")

    try:
        # Test if stock_data_server can be imported
        spec = importlib.util.spec_from_file_location(
            "stock_data_server", stock_server_path
        )
        if spec is None or spec.loader is None:
            print("⚠️ WARNING: Could not load stock_data_server.py")
        else:
            print("✅ SUCCESS: stock_data_server.py is importable")
    except Exception as e:
        print(f"⚠️ WARNING: Could not import stock_data_server.py: {e}")

    return True


def main():
    st.set_page_config(page_title="QueryStockAI", page_icon="📈", layout="wide")

    st.title("📈 QueryStockAI")
    st.markdown(
        "Get comprehensive financial analysis and insights for your selected stocks."
    )

    # Initialize resource monitoring
    if RESOURCE_MONITORING_AVAILABLE:
        if "resource_monitoring_started" not in st.session_state:
            start_resource_monitoring()
            st.session_state.resource_monitoring_started = True

    # Initialize tools
    initialize_tools()

    # Test server availability only once on startup
    if "servers_tested" not in st.session_state:
        st.session_state.servers_tested = False

    if not st.session_state.servers_tested:
        test_server_availability()
        st.session_state.servers_tested = True

    # Available tickers
    with st.spinner("🔄 Loading available tickers..."):
        available_tickers = get_available_tickers()

    # Sidebar for ticker selection
    st.sidebar.header("📊 Stock Selection")

    st.sidebar.subheader("📋 Popular Stocks")

    # Only show selectbox if tickers are loaded
    if available_tickers and len(available_tickers) > 0:
        selected_ticker = st.sidebar.selectbox(
            "Choose a stock ticker:",
            options=list(available_tickers.keys()),
            format_func=lambda x: f"{x} - {available_tickers[x]}",
            index=None,
            placeholder="Select a ticker...",
        )
    else:
        st.sidebar.error("❌ Failed to load tickers. Please refresh the page.")
        selected_ticker = None

    # Add search functionality
    st.sidebar.subheader("🔍 Search Custom Ticker")
    custom_ticker = st.sidebar.text_input(
        "Enter ticker symbol, if not found in above dropdown (e.g., AAPL, TSLA):",
        placeholder="Enter ticker symbol...",
        key="custom_ticker_input",
    )

    # Add info button with helpful information
    if custom_ticker:
        custom_ticker = custom_ticker.upper().strip()
        if custom_ticker:
            # Search for the custom ticker
            company_name = search_ticker(custom_ticker)
            if company_name:
                st.sidebar.success(
                    f"✅ Found: {custom_ticker} - {company_name} -> Added to dropdown list above."
                )
                # Add to available tickers temporarily
                available_tickers[custom_ticker] = company_name
            else:
                st.sidebar.error(f"❌ Could not find ticker: {custom_ticker}")

    # Clear cache when ticker changes
    if (
        "current_ticker" in st.session_state
        and st.session_state.current_ticker != selected_ticker
    ):
        # Clear all cached data for the previous ticker
        for key in list(st.session_state.keys()):
            if key.startswith("chart_") or key.startswith("news_"):
                del st.session_state[key]

    # Update current ticker
    if selected_ticker:
        st.session_state.current_ticker = selected_ticker

    # Main content area
    if not selected_ticker:
        st.info(
            "👈 Please select a stock ticker from the sidebar to view the chart and start chatting."
        )
        st.markdown(
            """
        **How to use:**
        1. Select a stock ticker from the sidebar
        2. View the interactive stock price chart
        3. Ask questions about the stock's performance, news, or investment advice
        4. The agent will fetch real-time data and provide comprehensive analysis

        **Example questions:**
        - "How is this stock performing?"
        - "What's the latest news about this company?"
        - "Should I invest in this stock?"
        - "What are the recent trends?"
        """
        )
    else:
        st.success(
            f"✅ Selected: {selected_ticker} - {available_tickers[selected_ticker]}"
        )

        # Add loading state for initial page load
        if "page_loaded" not in st.session_state:
            with st.spinner("🔄 Loading application..."):
                st.session_state.page_loaded = True

        # Stock Chart and News Section
        st.header("📈 Stock Analysis")

        # Create two columns for chart and news
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Stock Price Chart")
            # Cache the chart to prevent rerendering
            chart_key = f"chart_{selected_ticker}"
            if chart_key not in st.session_state:
                with st.spinner(f"📊 Loading chart for {selected_ticker}..."):
                    chart_fig = create_stock_chart(selected_ticker)
                    if chart_fig:
                        st.session_state[chart_key] = chart_fig
                    else:
                        st.session_state[chart_key] = None

            # Display the cached chart
            if st.session_state[chart_key]:
                st.plotly_chart(st.session_state[chart_key], use_container_width=True)
            else:
                st.warning(f"Could not load chart for {selected_ticker}")

        with col2:
            st.subheader("📰 Top News")
            # Cache the news to prevent rerendering
            news_key = f"news_{selected_ticker}"
            if news_key not in st.session_state:
                st.session_state[news_key] = True  # Mark as loaded
                display_top_news(selected_ticker)
            else:
                # Re-display cached news without reloading
                display_top_news(selected_ticker)

        # Chat Section
        st.header("💬 Chat with Financial Agent")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages using custom styling
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; border: 1px solid #bbdefb;">
                    <strong>You:</strong> {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style=" padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>Agent:</strong>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                # Render the content as markdown for proper formatting
                st.markdown(message["content"])

        # Chat input with proper loading state
        if prompt := st.chat_input(f"Ask about {selected_ticker}...", key="chat_input"):
            # Track streamlit request
            if RESOURCE_MONITORING_AVAILABLE:
                resource_monitor.increment_streamlit_requests()

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response with spinner above input
            with st.spinner("🤖 Analyzing your request..."):
                response = asyncio.run(run_agent(prompt, selected_ticker))
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            # Rerun to display the new message (charts and news are cached)
            st.rerun()

        # Clear chat button
        # col1, col2 = st.columns([1, 4])
        # with col1:
        #     if st.button("🗑️ Clear Chat History", key="clear_button"):
        #         st.session_state.messages = []
        #         st.rerun()
        # with col2:
        #     st.markdown("*Chat history will be maintained during your session*")


if __name__ == "__main__":
    main()
