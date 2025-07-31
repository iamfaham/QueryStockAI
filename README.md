# QueryStockAI

A comprehensive financial analysis tool that provides stock data, news analysis, and AI-powered insights through an interactive Streamlit web interface. Features advanced machine learning-based stock price predictions using Ridge Regression with comprehensive technical indicators.

## Features

- **Stock Data**: Fetch historical stock prices and performance metrics using Yahoo Finance
- **Interactive Stock Charts**: Visualize stock performance with Plotly charts showing 1 year of data
- **Advanced ML Predictions**: Ridge Regression model with 5 years of training data and 30-day forecasts
- **Comprehensive Technical Indicators**: 35+ technical indicators including RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, and more
- **Latest News Analysis**: Get recent news headlines for selected stocks
- **AI-Powered Chat Interface**: Chat with a financial agent powered by mistral via OpenRouter
- **MCP Server Integration**: Modular architecture with separate MCP servers for stock data and news
- **System Resource Monitoring**: Real-time monitoring of CPU, memory, disk, and network usage
- **Stock Search & Discovery**: Search for custom tickers and browse popular stocks
- **Caching & Performance**: Intelligent caching for charts and news to improve performance
- **Feature Scaling**: StandardScaler for optimal model performance
- **Cross-Validation**: GridSearchCV for hyperparameter tuning

## Machine Learning Model

### Ridge Regression with Enhanced Features

- **Training Data**: 5 years of historical stock data
- **Display Data**: Last 1 year shown in charts
- **Prediction Period**: 30 trading days
- **Features**: 35+ technical indicators including:
  - Moving Averages (SMA 10, 20, 50, 200)
  - Momentum Indicators (RSI, MACD, Stochastic, Williams %R, CCI)
  - Volatility Indicators (Bollinger Bands, Price Volatility)
  - Volume Analysis (Volume Change, Volume-Price Trend)
  - Support/Resistance Levels
  - Time-Based Features (Day of Week, Month, Quarter)
  - Market Sentiment Indicators

### Model Performance

- **Regularization**: Ridge Regression with L2 regularization
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Feature Scaling**: StandardScaler for optimal performance
- **Accuracy**: Typically 80-95% R² score on historical data
- **Training Time**: ~2-5 seconds per stock

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
3. View the interactive stock price chart showing:
   - Last 1 year of historical data
   - 30-day Ridge Regression predictions
   - Model performance metrics
4. Start chatting with the financial agent about the selected stock
5. Ask questions like:
   - "How is this stock performing?"
   - "What's the latest news about this company?"
   - "Should I invest in this stock?"
   - "What are the recent trends?"

## Architecture

- **Frontend**: Streamlit web interface with interactive charts
- **Backend**: Python with OpenRouter integration
- **ML Pipeline**: Ridge Regression with scikit-learn
- **Data Sources**:
  - Stock data via `yfinance`
  - News data via `gnews`
- **AI Model**: mistral-small-3.2-24b-instruct via OpenRouter
- **MCP Servers**: Modular servers for stock data and news
- **Monitoring**: Real-time system resource monitoring

## Files

- `Home.py`: Main Streamlit web application with ML predictions
- `stock_data_server.py`: MCP server for stock data
- `news_server.py`: MCP server for news data
- `resource_monitor.py`: System resource monitoring
- `pages/System_Monitor.py`: System monitoring dashboard
- `stock_data_linear_regression.ipynb`: Jupyter notebook with original ML approach
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project configuration

## Dependencies

- **Streamlit**: Web interface framework
- **yfinance**: Stock data fetching
- **gnews**: News data fetching
- **plotly**: Interactive charts
- **scikit-learn**: Machine learning (Ridge Regression, StandardScaler, GridSearchCV)
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **psutil**: System monitoring
- **openai**: AI model integration
- **fastmcp**: MCP server framework

## Technical Indicators Used

### Price-Based Features

- Simple Moving Averages (10, 20, 50, 200-day)
- Price Change (1, 5, 20-day)
- Price Volatility and Range
- Support/Resistance Levels

### Momentum Indicators

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator (K% and D%)
- Williams %R
- Commodity Channel Index (CCI)

### Volatility Indicators

- Bollinger Bands (Standard Deviation, Position, Squeeze)
- Price Volatility
- Price Range

### Volume Analysis

- Volume Change and Trends
- Volume-Price Relationship
- Volume Moving Averages
- Volume Spikes

### Market Sentiment

- Moving Average Crossovers
- Price vs Long-term Averages
- Time-based Patterns

## System Requirements

- Python 3.10 or higher
- OpenRouter API key
- Internet connection for real-time data
- Optional: psutil for system monitoring features

## Disclaimer

Stock predictions have approximately 80% accuracy. These forecasts are for informational purposes only and should not be used as the sole basis for investment decisions. Always conduct your own research and consider consulting with financial advisors.
