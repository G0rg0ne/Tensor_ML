# Use NVIDIA PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY train_bert.py .

# Create directory for results
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1

