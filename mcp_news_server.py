from mcp.server.fastmcp import FastMCP
import gnews
from bs4 import BeautifulSoup
import re
import os

port = int(os.environ.get("PORT", "8002"))

mcp = FastMCP(
    name="Financial News Server",
    description="Provides tools and resources for fetching and analyzing financial news with real-time updates via SSE.",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
)


def preprocess_text(text):
    """A simple function to clean text by removing HTML and extra whitespace."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


@mcp.tool()
async def get_latest_news(ticker: str) -> str:
    """
    Fetches recent news headlines and descriptions for a specific stock ticker.
    Use this tool when a user asks for news, updates, recent events about a company or about the company's stock.
    The input must be a valid stock ticker symbol (e.g., 'AAPL', 'GOOG').
    """
    try:
        google_news = gnews.GNews(language="en", country="US", period="7d")
        search_query = f'"{ticker}" stock market news'
        articles = google_news.get_news(search_query)

        if not articles:
            return f"No recent news found for ticker {ticker}."

        # Format the top 5 articles into a clean string for the LLM
        formatted_response = f"Recent News for {ticker}:\n"
        for article in articles[:5]:
            title = preprocess_text(article.get("title"))
            publisher = article.get("publisher", {}).get("title", "N/A")
            formatted_response += f"- {title} (Source: {publisher})\n"
        return formatted_response
    except Exception as e:
        return f"An error occurred while fetching news: {e}"


@mcp.resource("docs://news_sources_guide")
def get_news_sources_guide() -> str:
    """Provides guidance on interpreting financial news sources."""
    return """
    When analyzing financial news, consider the source's reputation.
    - Tier 1 (High Reliability): Reuters, Bloomberg, Wall Street Journal (WSJ). These are professional financial news outlets.
    - Tier 2 (Moderate Reliability): Major news networks (e.g., CNBC, Forbes), market analysis sites (e.g., Seeking Alpha). Often reliable but may have opinion pieces.
    - Tier 3 (Variable Reliability): General news aggregators, press releases. Verify information with other sources.
    """


@mcp.prompt()
def summarize_news_sentiment(news_headlines: str) -> str:
    """
    Analyzes a block of news headlines and provides a structured sentiment summary.
    The input should be a string containing multiple news headlines.
    """
    return f"""
    Analyze the following financial news headlines and provide a concise summary of the market sentiment.
    First, categorize the overall sentiment as 'Positive', 'Negative', or 'Neutral'.
    Then, provide a brief one-sentence justification for your choice, citing a key theme from the headlines.

    Headlines to analyze:
    ---
    {news_headlines}
    ---

    Your structured summary:
    """


if __name__ == "__main__":
    # Use streamable HTTP transport with MCP endpoint
    print(f"News Server running on http://localhost:{port}")
    print("MCP endpoint available at:")
    print(f"- http://localhost:{port}/mcp")

    mcp.run(transport="streamable-http")
