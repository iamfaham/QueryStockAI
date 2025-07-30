# QueryStockAI

A comprehensive financial analysis tool that provides stock data, news analysis, and AI-powered insights through an interactive Streamlit web interface.

## Features

- **Stock Data**: Fetch historical stock prices and performance metrics using Yahoo Finance
- **Interactive Stock Charts**: Visualize stock performance with Plotly charts
- **Latest News Analysis**: Get recent news headlines for selected stocks
- **AI-Powered Chat Interface**: Chat with a financial agent powered by mistral via OpenRouter
- **MCP Server Integration**: Modular architecture with separate MCP servers for stock data and news
- **Prophet Forecasting**: Optional time series forecasting capabilities
- **System Resource Monitoring**: Real-time monitoring of CPU, memory, disk, and network usage
- **Stock Search & Discovery**: Search for custom tickers and browse popular stocks
- **Caching & Performance**: Intelligent caching for charts and news to improve performance

## Setup

1. **Install dependencies**:

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** with your API keys:

   ```
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   MODEL="mistralai/mistral-small-3.2-24b-instruct:free"      # or any model of your choice
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run Home.py
   ```

   or using uv:

   ```bash
   uv run streamlit run Home.py
   ```

## Usage

1. Open the web interface in your browser
2. Select a stock ticker from the dropdown in the sidebar or search for a custom ticker
3. View the interactive stock price chart and latest news
4. Start chatting with the financial agent about the selected stock
5. Ask questions like:
   - "How is this stock performing?"
   - "What's the latest news about this company?"
   - "Should I invest in this stock?"
   - "What are the recent trends?"

## Architecture

- **Frontend**: Streamlit web interface with interactive charts
- **Backend**: Python with OpenRouter integration
- **Data Sources**:
  - Stock data via `yfinance`
  - News data via `gnews`
- **AI Model**: mistral-small-3.2-24b-instruct via OpenRouter
- **MCP Servers**: Modular servers for stock data and news
- **Monitoring**: Real-time system resource monitoring

## Files

- `Home.py`: Main Streamlit web application
- `stock_data_server.py`: MCP server for stock data
- `news_server.py`: MCP server for news data
- `resource_monitor.py`: System resource monitoring
- `pages/System_Monitor.py`: System monitoring dashboard
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project configuration

## Dependencies

- **Streamlit**: Web interface framework
- **yfinance**: Stock data fetching
- **gnews**: News data fetching
- **plotly**: Interactive charts
- **prophet**: Time series forecasting (optional)
- **psutil**: System monitoring
- **openai**: AI model integration
- **fastmcp**: MCP server framework

## System Requirements

- Python 3.10 or higher
- OpenRouter API key
- Internet connection for real-time data
- Optional: psutil for system monitoring features
