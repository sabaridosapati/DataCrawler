#!/bin/bash
# gpu_services/llm_service/setup_local.sh
# Setup script for running Gemma 3-4B LLM service locally on Lenovo laptop

set -e

echo "=== Setting up Gemma 3-4B LLM Service locally ==="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "CUDA Status:"
nvidia-smi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Try to install flash attention (optional, may fail on some systems)
echo "Attempting to install flash attention (optional)..."
pip install flash-attn --no-build-isolation || echo "Flash attention installation failed, continuing without it"

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p cache

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Gemma 3-4B LLM Service Configuration
MODEL_NAME=google/gemma-3-4b-it
DEVICE=cuda
HOST=0.0.0.0
PORT=8001
GPU_MEMORY_FRACTION=0.95
HF_TOKEN=
EOF
    echo "Created .env file. Please edit it if needed."
fi

echo "=== Setup complete! ==="
echo ""
echo "To start the service:"
echo "1. source venv/bin/activate"
echo "2. python main.py"
echo ""
echo "Or run the start script:"
echo "./start_local.sh"