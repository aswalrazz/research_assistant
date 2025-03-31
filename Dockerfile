# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs ons
EXPOSE 8503

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["streamlit", "run", "research_assistant.py", "--server.port=8503", "--server.address=0.0.0.0"] 