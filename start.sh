#!/bin/bash

# Create necessary directories
mkdir -p downloads temp .flask_session .spotify_cache

# Install dependencies
pip install -r requirements.txt

# Start the application
gunicorn app:app --bind 0.0.0.0:${PORT:-5005} --workers 2 --threads 4 --timeout 120
