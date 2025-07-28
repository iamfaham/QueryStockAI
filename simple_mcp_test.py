#!/usr/bin/env python3
"""
Simple MCP test based on official documentation
"""

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


async def test_simple_mcp():
    """Test MCP connection using the official approach"""
    print("Testing MCP connection...")

    try:
        # Use the official StdioServerParameters approach
        server_params = StdioServerParameters(command="python", args=["news_server.py"])

        # Connect using the official method
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Session initialized")

                # List available tools
                tools_response = await session.list_tools()
                print(
                    f"✅ Available tools: {[tool.name for tool in tools_response.tools]}"
                )

                # Call a tool
                result = await session.call_tool("get_latest_news", {"ticker": "TSLA"})
                print(f"✅ Tool result: {result.content[0].text[:100]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_mcp())
