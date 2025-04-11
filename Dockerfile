# Use Python 3.11 base image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_MODELS=/app/.ollama/models
ENV OLLAMA_HOST=localhost

# Set working directory
WORKDIR /app

# Install system and Ollama dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    ca-certificates \
    gnupg \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3.2 model during build (ensure it's cached)
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
# Copy application code
COPY . .

# Expose only FastAPI backend
EXPOSE 8000

# Start both Ollama and Uvicorn
CMD bash -c "ollama serve & sleep 5 && uvicorn main:app --host 0.0.0.0 --port 8000"
