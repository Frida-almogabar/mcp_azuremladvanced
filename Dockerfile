# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt ./
COPY pyproject.toml poetry.lock* ./

# Install dependencies - prefer requirements.txt for simpler deployment
RUN pip install --no-cache-dir -r requirements.txt

# Alternative: Use Poetry if you prefer
# RUN pip install poetry && \
#     poetry config virtualenvs.create false && \
#     poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "mcp_server.py", "--host", "0.0.0.0", "--port", "8080"]
