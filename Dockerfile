FROM python:3.11-slim

WORKDIR /app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose standard port
EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

# Start modern web application
CMD ["python", "run_app.py"]
