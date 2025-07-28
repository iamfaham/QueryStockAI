# Financial Agent

A comprehensive financial analysis tool that provides real-time stock data and news analysis through an AI-powered chat interface.

## Features

- **Real-time Stock Data**: Fetch historical stock prices and performance metrics
- **Latest News Analysis**: Get recent news headlines and sentiment analysis
- **AI-Powered Insights**: Receive comprehensive analysis and investment recommendations
- **Interactive Chat Interface**: Modern Streamlit-based web interface
- **Multiple Stock Support**: Analyze AAPL, TSLA, MSFT, GOOG, and more

## Setup

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Create a `.env` file** with your API keys:

   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   MODEL=openai/gpt-4o-mini
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. Open the web interface in your browser
2. Select a stock ticker from the dropdown in the sidebar
3. Start chatting with the financial agent about the selected stock
4. Ask questions like:
   - "How is this stock performing?"
   - "What's the latest news about this company?"
   - "Should I invest in this stock?"
   - "What are the recent trends?"

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with OpenAI/OpenRouter integration
- **Data Sources**:
  - Stock data via `yfinance`
  - News data via `gnews`
- **AI Model**: GPT-4o-mini via OpenRouter

## Files

- `streamlit_app.py`: Main Streamlit web application
- `agent_client.py`: Original terminal-based client
- `stock_data_server.py`: MCP server for stock data
- `news_server.py`: MCP server for news data
