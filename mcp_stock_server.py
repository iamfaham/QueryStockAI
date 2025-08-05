from mcp.server.fastmcp import FastMCP
import yfinance as yf

# Initialize the MCP Server
import os

port = int(os.environ.get("PORT", "8001"))

mcp = FastMCP(
    name="Stock Market Data Server",
    description="Provides tools for fetching historical stock prices and market data with real-time updates via SSE.",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
)


@mcp.tool()
async def get_historical_stock_data(ticker: str) -> str:
    """
    Fetches recent historical stock data (Open, High, Low, Close, Volume) for a given stock ticker.
    The input must be a single, valid stock ticker symbol (e.g., 'TSLA').
    Returns data for the last 30 days.
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch data for the last 30 days
        hist_data = stock.history(period="30d")

        if hist_data.empty:
            return f"No stock data found for ticker {ticker}."

        return f"Recent stock data for {ticker}:\n{hist_data.to_string()}"
    except Exception as e:
        return f"An error occurred while fetching stock data: {e}"


@mcp.resource("docs://ohlc_definitions")
def get_ohlc_definitions() -> str:
    """Provides definitions for standard stock data columns (OHLCV)."""
    return """
    - Open: The price at which a stock first traded upon the opening of an exchange on a given day.
    - High: The highest price at which a stock traded during the course of a day.
    - Low: The lowest price at which a stock traded during the course of a day.
    - Close: The last price at which a stock traded during a regular trading day.
    - Volume: The total number of shares traded for a stock in a day.
    """


if __name__ == "__main__":
    # Use streamable HTTP transport with MCP endpoint
    print(f"Stock Data Server running on http://localhost:{port}")
    print("MCP endpoint available at:")
    print(f"- http://localhost:{port}/mcp")

    mcp.run(transport="streamable-http")
