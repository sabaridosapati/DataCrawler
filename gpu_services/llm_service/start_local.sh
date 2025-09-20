#!/bin/bash
# gpu_services/llm_service/start_local.sh
# Start script for Gemma 3-4B LLM service

set -e

echo "=== Starting Gemma 3-4B LLM Service ==="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup_local.sh
fi

# Activate virtual environment
source venv/bin/activate

# Check GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

# Create log file with timestamp
LOG_FILE="logs/llm_service_$(date +%Y%m%d_%H%M%S).log"

echo "Starting service with logs in: $LOG_FILE"
echo "Service will be available at: http://localhost:8001"
echo "Health check: http://localhost:8001/health"
echo "API docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start the service with logging
python main.py 2>&1 | tee "$LOG_FILE"