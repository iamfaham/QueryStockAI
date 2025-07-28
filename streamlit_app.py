import streamlit as st
import asyncio
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, types

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


def extract_ticker_from_query(query: str) -> str:
    """Extract ticker symbol from user query."""
    query_upper = query.upper()

    # First try to find ticker in parentheses
    paren_match = re.search(r"\(([A-Z]{1,5})\)", query_upper)
    if paren_match:
        return paren_match.group(1)

    # Look for our predefined tickers in the query
    predefined_tickers = ["AAPL", "TSLA", "MSFT", "GOOG"]
    for ticker in predefined_tickers:
        if ticker in query_upper:
            return ticker

    # Try to find any 2-5 letter uppercase sequence that might be a ticker
    ticker_match = re.search(r"\b([A-Z]{2,5})\b", query_upper)
    if ticker_match:
        potential_ticker = ticker_match.group(1)
        # Avoid common words that might be mistaken for tickers
        if potential_ticker not in [
            "THE",
            "AND",
            "FOR",
            "HOW",
            "WHAT",
            "WHEN",
            "WHERE",
            "WHY",
        ]:
            return potential_ticker

    return None


def validate_ticker(ticker: str) -> bool:
    """Validate if ticker symbol is in correct format."""
    if not ticker:
        return False
    # Basic validation: 1-5 uppercase letters
    return bool(re.match(r"^[A-Z]{1,5}$", ticker))


async def get_news_data(ticker: str) -> str:
    """Get news data by calling the news server via MCP."""
    try:
        # Validate ticker
        if not validate_ticker(ticker):
            return f"Invalid ticker symbol: {ticker}. Please use a valid stock symbol (e.g., AAPL, TSLA)."

        # Set up MCP server parameters
        import os
        import sys

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
        # Validate ticker
        if not validate_ticker(ticker):
            return f"Invalid ticker symbol: {ticker}. Please use a valid stock symbol (e.g., AAPL, TSLA)."

        # Set up MCP server parameters
        import os
        import sys

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
    """Create an interactive stock price chart for the given ticker."""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="30d")

        if hist_data.empty:
            st.warning(f"No data available for {ticker}")
            return

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
        arguments = json.loads(tool_call.function.arguments)
        ticker = arguments.get("ticker")

        with st.status(
            f"🛠️ Executing {tool_name} for {ticker}...", expanded=False
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
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid tool arguments format: {e}")
        return f"Error: Invalid tool arguments format"
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
        with st.spinner("🤖 Analyzing your request..."):
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
            with st.spinner("🤖 Generating analysis..."):
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
        import gnews
        from bs4 import BeautifulSoup
        import re

        def preprocess_text(text):
            """A simple function to clean text by removing HTML and extra whitespace."""
            if not text:
                return ""
            soup = BeautifulSoup(text, "html.parser")
            clean_text = soup.get_text()
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text

        # Get news data
        google_news = gnews.GNews(language="en", country="US", period="7d")
        search_query = f'"{ticker}" stock market news'
        articles = google_news.get_news(search_query)

        if not articles:
            st.info(f"No recent news found for {ticker}")
            return

            # Display top 5 articles
        for i, article in enumerate(articles[:5], 1):
            title = preprocess_text(article.get("title", ""))
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


def test_server_availability():
    """Test if the MCP servers are available and can be executed."""
    import os
    import subprocess
    import time

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Test news server
    news_server_path = os.path.join(current_dir, "news_server.py")
    if not os.path.exists(news_server_path):
        st.error(f"❌ news_server.py not found at {news_server_path}")
        return False

    # Test stock data server
    stock_server_path = os.path.join(current_dir, "stock_data_server.py")
    if not os.path.exists(stock_server_path):
        st.error(f"❌ stock_data_server.py not found at {stock_server_path}")
        return False

    # Test if servers can be executed by checking if they can be imported
    import sys
    import importlib.util

    # Initialize session state for notifications
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "notification_times" not in st.session_state:
        st.session_state.notification_times = {}
    if "servers_importable_shown" not in st.session_state:
        st.session_state.servers_importable_shown = False

    current_time = time.time()

    # Clean up old notifications (older than 10 seconds)
    st.session_state.notifications = [
        msg
        for msg, timestamp in zip(
            st.session_state.notifications, st.session_state.notification_times.values()
        )
        if current_time - timestamp < 10
    ]
    st.session_state.notification_times = {
        k: v
        for k, v in st.session_state.notification_times.items()
        if current_time - v < 10
    }

    try:
        # Test if news_server can be imported
        spec = importlib.util.spec_from_file_location("news_server", news_server_path)
        if spec is None or spec.loader is None:
            st.warning("⚠️ Could not load news_server.py")
        else:
            # Add temporary success notification only once
            if not st.session_state.servers_importable_shown:
                st.success("✅ news_server.py is importable")
                st.session_state.servers_importable_shown = True
    except Exception as e:
        st.warning(f"⚠️ Could not import news_server.py: {e}")

    try:
        # Test if stock_data_server can be imported
        spec = importlib.util.spec_from_file_location(
            "stock_data_server", stock_server_path
        )
        if spec is None or spec.loader is None:
            st.warning("⚠️ Could not load stock_data_server.py")
        else:
            # Add temporary success notification only once
            if not st.session_state.servers_importable_shown:
                st.success("✅ stock_data_server.py is importable")
                st.session_state.servers_importable_shown = True
    except Exception as e:
        st.warning(f"⚠️ Could not import stock_data_server.py: {e}")

    return True


def main():
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")

    st.title("📈 Financial Agent")
    st.markdown(
        "Get comprehensive financial analysis and insights for your selected stocks."
    )

    # Initialize tools
    initialize_tools()

    # Test server availability only once on startup
    if "servers_tested" not in st.session_state:
        st.session_state.servers_tested = False

    if not st.session_state.servers_tested:
        test_server_availability()
        st.session_state.servers_tested = True

    # Available tickers
    available_tickers = {
        "AAPL": "Apple Inc.",
        "TSLA": "Tesla Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOG": "Alphabet Inc. (Google)",
    }

    # Sidebar for ticker selection
    st.sidebar.header("📊 Stock Selection")
    selected_ticker = st.sidebar.selectbox(
        "Choose a stock ticker:",
        options=list(available_tickers.keys()),
        format_func=lambda x: f"{x} - {available_tickers[x]}",
        index=None,
        placeholder="Select a ticker...",
    )

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

        # Stock Chart and News Section
        st.header("📈 Stock Analysis")

        # Create two columns for chart and news
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Stock Price Chart")
            # Create and display the stock chart
            with st.spinner(f"Loading chart for {selected_ticker}..."):
                chart_fig = create_stock_chart(selected_ticker)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    st.warning(f"Could not load chart for {selected_ticker}")

        with col2:
            st.subheader("📰 Top News")
            # Display top news for the selected ticker
            display_top_news(selected_ticker)

        # Chat Section in a container
        st.header("💬 Chat with Financial Agent")

        # Create a container for the chat interface
        with st.container():
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input(f"Ask about {selected_ticker}..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Analyzing..."):
                        response = asyncio.run(run_agent(prompt, selected_ticker))
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

            # Clear chat button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                st.markdown("*Chat history will be maintained during your session*")


if __name__ == "__main__":
    main()
