#!/usr/bin/env python3
"""
Script to update MCP server URLs for Hugging Face Spaces deployment.
This script modifies the MCP client configuration in Home.py to use the correct ports.
"""

import os
import re


def get_port():
    """Get the main port from environment (Hugging Face Spaces uses 7860)."""
    port = os.environ.get("PORT", "7860")
    return int(port)


def update_home_py():
    """Update Home.py to use correct MCP server URLs for Hugging Face Spaces deployment."""
    port = get_port()

    # Calculate the ports for MCP servers
    stock_port = port + 1
    news_port = port + 2

    # Read the current Home.py file
    with open("Home.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Update the MCP client configuration
    # Find the MultiServerMCPClient configuration
    pattern = r'client = MultiServerMCPClient\(\s*\{\s*"news_server":\s*\{\s*"url":\s*"[^"]*",\s*"transport":\s*"[^"]*",?\s*\},\s*"stock_server":\s*\{\s*"url":\s*"[^"]*",\s*"transport":\s*"[^"]*",?\s*\},\s*\}\s*\)'

    new_config = f"""client = MultiServerMCPClient(
                {{
                    "news_server": {{
                        "url": "http://localhost:{news_port}/mcp",
                        "transport": "streamable_http",
                    }},
                    "stock_server": {{
                        "url": "http://localhost:{stock_port}/mcp",
                        "transport": "streamable_http",
                    }},
                }}
            )"""

    # Replace the configuration
    updated_content = re.sub(pattern, new_config, content, flags=re.DOTALL)

    # Write the updated content back
    with open("Home.py", "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"✅ Updated Home.py with MCP server URLs:")
    print(f"   - Stock server: http://localhost:{stock_port}/mcp")
    print(f"   - News server: http://localhost:{news_port}/mcp")


if __name__ == "__main__":
    update_home_py()
