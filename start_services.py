#!/usr/bin/env python3
"""
Startup script to run all services on Railway deployment.
This script launches:
1. Streamlit app on $PORT (main port)
2. MCP Stock Server on $PORT + 1
3. MCP News Server on $PORT + 2
"""

import os
import sys
import time
import signal
import subprocess
import multiprocessing
from pathlib import Path


def get_port():
    """Get the main port from Railway environment."""
    port = os.environ.get("PORT", "8501")
    return int(port)


def run_streamlit():
    """Run Streamlit app on main port."""
    port = get_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "Home.py",
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"🚀 Starting Streamlit on port {port}")
    subprocess.run(cmd, check=True)


def run_stock_server():
    """Run MCP Stock Server on port + 1."""
    port = get_port() + 1
    cmd = [sys.executable, "mcp_stock_server.py"]
    print(f"📊 Starting MCP Stock Server on port {port}")
    # Set the port environment variable for the stock server
    env = os.environ.copy()
    env["PORT"] = str(port)
    subprocess.run(cmd, env=env, check=True)


def run_news_server():
    """Run MCP News Server on port + 2."""
    port = get_port() + 2
    cmd = [sys.executable, "mcp_news_server.py"]
    print(f"📰 Starting MCP News Server on port {port}")
    # Set the port environment variable for the news server
    env = os.environ.copy()
    env["PORT"] = str(port)
    subprocess.run(cmd, env=env, check=True)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Received shutdown signal. Stopping all services...")
    sys.exit(0)


def main():
    """Main function to start all services."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 Starting Financial Agent Services...")
    print(f"📋 Configuration:")
    print(f"   - Main Port: {get_port()}")
    print(f"   - Stock Server Port: {get_port() + 1}")
    print(f"   - News Server Port: {get_port() + 2}")

    # Check if required files exist
    required_files = ["Home.py", "mcp_stock_server.py", "mcp_news_server.py"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Error: {file} not found!")
            sys.exit(1)

    # Update MCP server URLs in Home.py for Railway deployment
    try:
        print("🔄 Updating MCP server URLs...")
        subprocess.run([sys.executable, "update_mcp_urls.py"], check=True)
        print("✅ MCP server URLs updated successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not update MCP URLs: {e}")
        print("Continuing with default configuration...")

    # Start all services in separate processes
    processes = []

    try:
        # Start MCP servers first (they need to be ready before Streamlit)
        print("🔄 Starting MCP servers...")

        # Start stock server
        stock_process = multiprocessing.Process(
            target=run_stock_server, name="stock-server"
        )
        stock_process.start()
        processes.append(stock_process)

        # Start news server
        news_process = multiprocessing.Process(
            target=run_news_server, name="news-server"
        )
        news_process.start()
        processes.append(news_process)

        # Wait a bit for MCP servers to start
        print("⏳ Waiting for MCP servers to initialize...")
        time.sleep(5)

        # Start Streamlit app
        print("🔄 Starting Streamlit app...")
        streamlit_process = multiprocessing.Process(
            target=run_streamlit, name="streamlit"
        )
        streamlit_process.start()
        processes.append(streamlit_process)

        print("✅ All services started successfully!")
        print(f"🌐 Streamlit app available at: http://localhost:{get_port()}")
        print(f"📊 Stock server available at: http://localhost:{get_port() + 1}")
        print(f"📰 News server available at: http://localhost:{get_port() + 2}")

        # Optional: Run deployment test
        if os.environ.get("RUN_DEPLOYMENT_TEST", "false").lower() == "true":
            print("\n🧪 Running deployment test...")
            try:
                subprocess.run([sys.executable, "test_deployment.py"], check=True)
                print("✅ Deployment test passed!")
            except subprocess.CalledProcessError:
                print("⚠️ Deployment test failed, but services may still be working...")

        # Wait for all processes
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    finally:
        # Terminate all processes
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        print("✅ All services stopped.")


if __name__ == "__main__":
    main()
