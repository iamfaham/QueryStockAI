#!/usr/bin/env python3
"""
Test script to verify Railway deployment is working correctly.
This script checks if all services are running on the expected ports.
"""

import os
import time
import requests
import subprocess
import sys


def get_port():
    """Get the main port from Railway environment."""
    port = os.environ.get("PORT", "8501")
    return int(port)


def test_service(url, service_name, timeout=10):
    """Test if a service is responding."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {service_name} is running at {url}")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code} at {url}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} is not responding at {url}: {e}")
        return False


def test_mcp_endpoint(url, service_name):
    """Test MCP endpoint specifically."""
    try:
        response = requests.get(f"{url}/mcp", timeout=10)
        if response.status_code == 200:
            print(f"✅ {service_name} MCP endpoint is accessible at {url}/mcp")
            return True
        else:
            print(
                f"❌ {service_name} MCP endpoint returned status {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} MCP endpoint is not accessible: {e}")
        return False


def main():
    """Main test function."""
    port = get_port()

    print("🧪 Testing Railway Deployment...")
    print(f"📋 Configuration:")
    print(f"   - Main Port: {port}")
    print(f"   - Stock Server Port: {port + 1}")
    print(f"   - News Server Port: {port + 2}")
    print()

    # Test services
    services = [
        (f"http://localhost:{port}", "Streamlit App"),
        (f"http://localhost:{port + 1}", "Stock Server"),
        (f"http://localhost:{port + 2}", "News Server"),
    ]

    results = []

    for url, name in services:
        result = test_service(url, name)
        results.append((name, result))

    print()
    print("🔍 Testing MCP Endpoints...")

    # Test MCP endpoints
    mcp_results = []
    mcp_services = [
        (f"http://localhost:{port + 1}", "Stock Server"),
        (f"http://localhost:{port + 2}", "News Server"),
    ]

    for url, name in mcp_services:
        result = test_mcp_endpoint(url, name)
        mcp_results.append((name, result))

    print()
    print("📊 Test Results Summary:")

    all_passed = True
    for name, result in results + mcp_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All services are running correctly!")
        print("🚀 Your Railway deployment is ready to use.")
    else:
        print("⚠️ Some services failed. Check the logs above for details.")
        print("🔧 Try restarting the deployment or check Railway logs.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    # Wait a bit for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(5)

    exit_code = main()
    sys.exit(exit_code)
