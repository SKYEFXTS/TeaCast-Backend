#!/usr/bin/env bash
# Build script for Render deployment

echo "Build started"

# Create necessary directories
mkdir -p logs
mkdir -p Data/PreProcessedData

# Install dependencies
pip install -r requirements.txt

echo "Build completed"