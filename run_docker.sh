#!/bin/bash

# Script to run ElDensiago in Docker container

echo "Building ElDensiago Docker image..."
docker build -t eldensiago .

echo "Running ElDensiago container..."
echo "Current directory will be mounted to /app in the container"
echo "External data directory will be mounted to /data in the container"
echo ""
echo "To run a prediction, use:"
echo "  python -c \"from guess import MlDensity; predictor = MlDensity(); predictor.predict('example/batio3.xyz')\""
echo ""
echo "To start Jupyter notebook:"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""

# Run the container with volume mounting
docker run -it --rm \
    -v "$(pwd):/app" \
    -v "$(pwd)/data:/data" \
    -p 8888:8888 \
    eldensiago 