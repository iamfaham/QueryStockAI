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
from datetime import datetime, timedelta

try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet not installed. Please run: pip install prophet")
    Prophet = None
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, types

# Import resource monitoring
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


def create_stock_chart(ticker: str):
    """Create an interactive stock price chart with Prophet predictions for the given ticker."""
    try:
        # Check if Prophet is available
        if Prophet is None:
            st.error("Prophet is not installed. Please install it with: uv add prophet")
            return create_basic_stock_chart(ticker)

        # Get stock data - 1 year for training Prophet
        with st.spinner(f"📊 Fetching stock data for {ticker}..."):
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="1y")

            # Track yfinance API call
            if RESOURCE_MONITORING_AVAILABLE:
                resource_monitor.increment_yfinance_calls()

        if hist_data.empty:
            st.warning(f"No data available for {ticker}")
            return None

        # Prepare data for Prophet with outlier removal
        df = hist_data.reset_index()

        # Remove outliers using IQR method for better model training
        Q1 = df["Close"].quantile(0.25)
        Q3 = df["Close"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        df = df[(df["Close"] >= lower_bound) & (df["Close"] <= upper_bound)]

        # Remove timezone information from the Date column for Prophet compatibility
        df["ds"] = df["Date"].dt.tz_localize(
            None
        )  # Prophet requires timezone-naive dates
        df["y"] = df["Close"]  # Prophet requires 'y' column for values

        # Train Prophet model with optimized configuration
        start_time = time.time()
        with st.spinner(f"Training Prophet model for {ticker}..."):
            # Configure Prophet model with optimized parameters
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.01,  # Reduced for smoother trends
                seasonality_prior_scale=10.0,  # Increased seasonality strength
                seasonality_mode="multiplicative",
                interval_width=0.8,  # Tighter confidence intervals
                mcmc_samples=0,  # Disable MCMC for faster training
            )

            # Add custom seasonalities for better stock patterns
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

            model.fit(df[["ds", "y"]])

            # Make predictions for next 30 days
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # Get the forecast data for the next 30 days (future predictions only)
            # Find the last date in historical data
            last_historical_date = df["ds"].max()
            tomorrow = last_historical_date + timedelta(days=1)

            # Filter for only future predictions (starting from tomorrow)
            forecast_future = forecast[forecast["ds"] >= tomorrow].copy()

            # Filter out non-trading days
            forecast_future["is_trading_day"] = forecast_future["ds"].apply(
                is_trading_day
            )
            forecast_future = forecast_future[
                forecast_future["is_trading_day"] == True
            ].copy()

            # If we don't have enough trading days, get more predictions
            if len(forecast_future) < 20:  # Aim for at least 20 trading days
                # Calculate how many more days we need
                additional_days_needed = 30 - len(forecast_future)
                future_extended = model.make_future_dataframe(
                    periods=30 + additional_days_needed
                )
                forecast_extended = model.predict(future_extended)

                # Filter extended forecast for trading days
                forecast_extended_future = forecast_extended[
                    forecast_extended["ds"] >= tomorrow
                ].copy()
                forecast_extended_future["is_trading_day"] = forecast_extended_future[
                    "ds"
                ].apply(is_trading_day)
                forecast_future = forecast_extended_future[
                    forecast_extended_future["is_trading_day"] == True
                ].copy()

                # Take only the first 30 trading days
                forecast_future = forecast_future.head(30)

        # Track Prophet training time
        training_time = time.time() - start_time
        if RESOURCE_MONITORING_AVAILABLE:
            resource_monitor.add_prophet_training_time(training_time)

        # Create interactive chart with historical data and predictions
        fig = go.Figure()

        # Add historical price data (full year for context)
        # Ensure we only show actual historical data, not predictions
        # Convert timezone-aware dates to timezone-naive for comparison
        hist_data_filtered = hist_data[
            hist_data.index.tz_localize(None) <= last_historical_date
        ]
        fig.add_trace(
            go.Scatter(
                x=hist_data_filtered.index,
                y=hist_data_filtered["Close"],
                mode="lines+markers",
                name=f"{ticker} Historical Price (Last Year)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )

        # Add Prophet predictions for next 30 days (starting from tomorrow)
        fig.add_trace(
            go.Scatter(
                x=forecast_future["ds"],
                y=forecast_future["yhat"],
                mode="lines+markers",
                name=f"{ticker} Future Predictions (Next 30 Days)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=4),
            )
        )

        # Add confidence intervals for future predictions
        fig.add_trace(
            go.Scatter(
                x=forecast_future["ds"].tolist() + forecast_future["ds"].tolist()[::-1],
                y=forecast_future["yhat_upper"].tolist()
                + forecast_future["yhat_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.3)",
                line=dict(color="rgba(255, 127, 14, 0)"),
                name="Prediction Confidence Interval",
                showlegend=False,
            )
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price with Next 30-Day Predictions",
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
        current_price = hist_data["Close"].iloc[-1]
        predicted_price_30d = forecast_future["yhat"].iloc[-1]
        price_change = predicted_price_30d - current_price
        price_change_pct = (price_change / current_price) * 100

        # Calculate confidence interval
        confidence_lower = forecast_future["yhat_lower"].iloc[-1]
        confidence_upper = forecast_future["yhat_upper"].iloc[-1]
        confidence_range = confidence_upper - confidence_lower

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
        **📊 30-Day Prediction Details for {ticker}:**
        - **Current Price:** ${current_price:.2f}
        - **Predicted Price (30 days):** ${predicted_price_30d:.2f}
        - **Expected Change:** ${price_change:.2f} ({price_change_pct:+.2f}%)
        - **Confidence Range:** ${confidence_lower:.2f} - ${confidence_upper:.2f} (±${confidence_range/2:.2f})
        - **Model Training Time:** {training_time:.2f}s

        ⚠️ **Disclaimer**: Stock predictions have approximately 51% accuracy.
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

    # Add search functionality
    st.sidebar.subheader("🔍 Search Custom Ticker")
    custom_ticker = st.sidebar.text_input(
        "Enter ticker symbol (e.g., AAPL, TSLA):",
        placeholder="Enter ticker symbol...",
        key="custom_ticker_input",
    )

    # Add info button with helpful information
    with st.sidebar.expander("ℹ️ Can't find your ticker in the list?", expanded=False):
        st.markdown(
            """
        **Can't find your ticker in the list?** 
        
        Use this search box to check if a ticker is available:
        
        ✅ **How it works:**
        - Enter any ticker symbol (e.g., AAPL, TSLA, GOOGL)
        - If found, it will be automatically added to the dropdown
        - You can then select it from the "Popular Stocks" list below
        
        ✅ **Examples:**
        - `AAPL` → Apple Inc.
        - `TSLA` → Tesla Inc.
        - `MSFT` → Microsoft Corporation
        - `GOOGL` → Alphabet Inc.
        
        ✅ **Tips:**
        - Use uppercase letters for best results
        - Most major US and international stocks are supported
        - If not found, the ticker might not be available on Yahoo Finance
        """
        )

    if custom_ticker:
        custom_ticker = custom_ticker.upper().strip()
        if custom_ticker:
            # Search for the custom ticker
            company_name = search_ticker(custom_ticker)
            if company_name:
                st.sidebar.success(f"✅ Found: {custom_ticker} - {company_name}")
                # Add to available tickers temporarily
                available_tickers[custom_ticker] = company_name
            else:
                st.sidebar.error(f"❌ Could not find ticker: {custom_ticker}")

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
