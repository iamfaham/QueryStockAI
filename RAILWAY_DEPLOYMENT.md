# Railway Deployment Guide

This guide explains how to deploy the Financial Agent on Railway with all services running on one server.

## Architecture

The deployment runs three services on a single Railway server:

1. **Streamlit App** (Main Port): The main web interface
2. **MCP Stock Server** (Port + 1): Provides stock data tools
3. **MCP News Server** (Port + 2): Provides news data tools

## Port Configuration

- **Main Port**: `$PORT` (set by Railway)
- **Stock Server**: `$PORT + 1`
- **News Server**: `$PORT + 2`

For example, if Railway assigns port 3000:

- Streamlit: `http://localhost:3000`
- Stock Server: `http://localhost:3001`
- News Server: `http://localhost:3002`

## Files Overview

### Core Application Files

- `Home.py` - Main Streamlit application (unchanged)
- `mcp_stock_server.py` - Stock data MCP server
- `mcp_news_server.py` - News data MCP server

### Deployment Files

- `start_services.py` - Main startup script that launches all services
- `update_mcp_urls.py` - Updates MCP server URLs in Home.py
- `railway.toml` - Railway configuration
- `Dockerfile` - Container configuration

## Deployment Process

1. **Railway Build**: Railway builds the Docker container
2. **Startup**: `start_services.py` is executed
3. **URL Update**: MCP server URLs are updated for the correct ports
4. **Service Launch**: All three services start in parallel
5. **Health Check**: Railway monitors the main Streamlit port

## Environment Variables

Set these in Railway dashboard:

```bash
GROQ_API_KEY=your_groq_api_key_here
MODEL=mistralai/mistral-small-3.2-24b-instruct:free
```

## Local Development

To test locally:

```bash
# Set environment variables
export PORT=8501
export GROQ_API_KEY=your_key_here
export MODEL=mistralai/mistral-small-3.2-24b-instruct:free

# Run the startup script
python start_services.py
```

## Troubleshooting

### Services Not Starting

1. Check Railway logs for error messages
2. Verify all required files are present
3. Ensure environment variables are set correctly

### MCP Connection Issues

1. Check if MCP servers are running on correct ports
2. Verify URLs in Home.py are updated correctly
3. Check network connectivity between services

### Memory Issues

- Railway provides limited memory
- Consider reducing model complexity if needed
- Monitor memory usage in Railway dashboard

## Monitoring

- **Main App**: Access via Railway URL
- **Logs**: Check Railway dashboard for service logs
- **Health**: Railway monitors `/_stcore/health` endpoint

## Security Notes

- All services run on localhost (internal communication)
- Only the main Streamlit port is exposed externally
- MCP servers are not directly accessible from outside

## Customization

To modify the deployment:

1. **Add new services**: Update `start_services.py`
2. **Change ports**: Modify port calculation logic
3. **Update configuration**: Edit `railway.toml` or `Dockerfile`

## Support

For deployment issues:

1. Check Railway logs first
2. Verify environment variables
3. Test locally before deploying
4. Review service startup sequence
