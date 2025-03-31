# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_LOG_LEVEL=debug \
    GEMINI_API_KEY=your_api_key_here

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app/data

# Expose the port Streamlit runs on
EXPOSE 8501

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Switch to non-root user
USER appuser

# Run the application with debug logging
CMD ["streamlit", "run", "research_assistant.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=debug", "--browser.serverAddress=0.0.0.0", "--browser.serverPort=8501"] 