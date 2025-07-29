import os
import asyncio
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, types

# Load API key from.env file
load_dotenv()

# Check if API key exists
api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("MODEL")
if not api_key:
    print("❌ Error: OPENROUTER_API_KEY not found in .env file")
    exit(1)
if not model:
    print("❌ Error: MODEL not found in .env file")
    exit(1)

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
        server_params = StdioServerParameters(command="python", args=["news_server.py"])

        # Connect to the MCP server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()

                # Call the get_latest_news tool
                print(f"🔍 Calling MCP tool: get_latest_news with ticker: {ticker}")
                try:
                    result = await asyncio.wait_for(
                        session.call_tool("get_latest_news", {"ticker": ticker}),
                        timeout=30.0,  # 30 second timeout
                    )
                    print(f"🔍 MCP result type: {type(result)}")
                    print(f"🔍 MCP result content: {result.content}")
                except asyncio.TimeoutError:
                    print("❌ MCP call timed out")
                    return f"Timeout getting news for {ticker}"
                except Exception as e:
                    print(f"❌ MCP call error: {e}")
                    return f"Error getting news for {ticker}: {e}"

                # Parse the result properly
                if result.content:
                    for content in result.content:
                        if isinstance(content, types.TextContent):
                            return content.text

                return f"No news data returned for {ticker}"

    except Exception as e:
        return f"Error getting news for {ticker}: {e}"


async def get_stock_data(ticker: str) -> str:
    """Get stock data by calling the stock server via MCP."""
    try:
        # Validate ticker
        if not validate_ticker(ticker):
            return f"Invalid ticker symbol: {ticker}. Please use a valid stock symbol (e.g., AAPL, TSLA)."

        # Set up MCP server parameters
        server_params = StdioServerParameters(
            command="python", args=["stock_data_server.py"]
        )

        # Connect to the MCP server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()

                # Call the get_historical_stock_data tool
                print(
                    f"🔍 Calling MCP tool: get_historical_stock_data with ticker: {ticker}"
                )
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(
                            "get_historical_stock_data", {"ticker": ticker}
                        ),
                        timeout=30.0,  # 30 second timeout
                    )
                    print(f"🔍 MCP result type: {type(result)}")
                    print(f"🔍 MCP result content: {result.content}")
                except asyncio.TimeoutError:
                    print("❌ MCP call timed out")
                    return f"Timeout getting stock data for {ticker}"
                except Exception as e:
                    print(f"❌ MCP call error: {e}")
                    return f"Error getting stock data for {ticker}: {e}"

                # Parse the result properly
                if result.content:
                    for content in result.content:
                        if isinstance(content, types.TextContent):
                            return content.text

                return f"No stock data returned for {ticker}"

    except Exception as e:
        return f"Error getting stock data for {ticker}: {e}"


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

    print(f"✅ Initialized {len(discovered_tools)} tools")


async def execute_tool_call(tool_call):
    """Execute a tool call using MCP servers."""
    try:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        ticker = arguments.get("ticker")

        if tool_name == "get_latest_news":
            return await get_news_data(ticker)
        elif tool_name == "get_historical_stock_data":
            return await get_stock_data(ticker)
        else:
            return f"Unknown tool: {tool_name}"
    except json.JSONDecodeError:
        return f"Error: Invalid tool arguments format"
    except Exception as e:
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


async def run_agent(user_query):
    print(f"\n🔍 User Query: {user_query}")

    # Extract ticker from query for validation
    extracted_ticker = extract_ticker_from_query(user_query)
    if extracted_ticker:
        print(f"📈 Detected ticker: {extracted_ticker}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        # Get initial response from the model
        print("🤖 Getting response from model...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=discovered_tools,
            tool_choice="required",
        )

        if not response.choices or len(response.choices) == 0:
            print("❌ Error: No response from model")
            return

        response_message = response.choices[0].message
        print(f"📝 Response type: {type(response_message)}")
        print(
            f"🔧 Has tool calls: {hasattr(response_message, 'tool_calls') and response_message.tool_calls}"
        )

        # Truncate tool call IDs if they're too long
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if len(tool_call.id) > 40:
                    tool_call.id = tool_call.id[:40]

        messages.append(response_message)

        # Execute tool calls if any
        if response_message.tool_calls:
            print("\n🛠️  Executing Tool Calls ---")
            for tool_call in response_message.tool_calls:
                print(f"📞 Calling: {tool_call.function.name}")
                print(f"📋 Arguments: {tool_call.function.arguments}")

                # Execute the tool call
                tool_result = await execute_tool_call(tool_call)
                print(
                    f"📊 Result: {tool_result[:200] if tool_result else 'No result'}..."
                )  # Show first 200 chars

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id[:40],  # Truncate to max 40 chars
                        "content": tool_result if tool_result else "No data available",
                    }
                )

            # Get final response from the model (using same model for consistency)
            print("🤖 Getting final response from model...")
            print(f"📝 Messages count: {len(messages)}")
            final_response = client.chat.completions.create(
                model="openai/gpt-4o-mini",  # Try a different model
                messages=messages,
            )

            print("\n🤖 Final Agent Response ---")
            if final_response.choices and len(final_response.choices) > 0:
                final_content = final_response.choices[0].message.content
                print(
                    f"Final response length: {len(final_content) if final_content else 0}"
                )
                print(final_content if final_content else "Empty response")
            else:
                print("No response generated")
            print("----------------------------")
        else:
            print("\n🤖 Agent Response ---")
            print(response_message.content)
            print("----------------------")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please try again with a different question.")


async def main():
    """Main function to run the connected agent interactively."""
    try:
        # Initialize tools
        initialize_tools()

        # Predefined tickers
        available_tickers = {"1": "AAPL", "2": "TSLA", "3": "MSFT", "4": "GOOG"}

        print("=== QueryStockAI ===")
        print("Select a stock ticker to analyze:")
        print("1. AAPL (Apple)")
        print("2. TSLA (Tesla)")
        print("3. MSFT (Microsoft)")
        print("4. GOOG (Google)")
        print("Type 'quit' or 'exit' to stop the program.")
        print("=" * 50)

        while True:
            # Show ticker menu
            print("\n📊 Available Stocks:")
            for key, ticker in available_tickers.items():
                print(f"  {key}. {ticker}")
            print("  q. Quit")

            # Get user selection
            selection = (
                input("\n💬 Select a stock (1-4) or type 'quit': ").strip().lower()
            )

            # Check if user wants to exit
            if selection in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            # Check if selection is valid
            if selection not in available_tickers:
                print("❌ Invalid selection. Please choose 1-4 or 'quit'.")
                continue

            # Get the selected ticker
            selected_ticker = available_tickers[selection]
            print(f"\n📈 Selected: {selected_ticker}")

            # Always fetch both news and stock data by default
            print(f"\n📊 Fetching comprehensive data for {selected_ticker}...")

            # Get user's specific question
            user_question = input(
                f"\n💬 What would you like to know about {selected_ticker}? (e.g., 'How is it performing?', 'What's the latest news?', 'Should I invest?'): "
            ).strip()

            if not user_question:
                user_question = (
                    f"How is {selected_ticker} performing and what's the latest news?"
                )

            # Construct the query to always fetch both data types
            user_query = f"Based on the latest news and stock performance data for {selected_ticker}, {user_question}"

            # Run the agent with the user's query
            await run_agent(user_query)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
