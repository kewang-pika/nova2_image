#!/bin/bash

# Nova2 Image Creator - Launcher Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default port
PORT=${1:-8501}

echo "========================================"
echo "Nova2 Image Creator"
echo "========================================"
echo "Starting Streamlit app on port $PORT..."
echo ""

# Run streamlit
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
