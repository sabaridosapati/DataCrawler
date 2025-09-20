#!/bin/bash
# start_lenovo_services.sh
# Startup script for Lenovo Laptop - Local LLM Service Only

set -e

echo "=== Starting Lenovo Laptop Services ==="
echo "Local: Gemma 3-4B LLM Service"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check if service is running
check_service() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts=60  # Longer timeout for LLM service
    local attempt=1
    
    print_info "Checking $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$health_url" > /dev/null 2>&1; then
            print_success "$service_name is running ✓"
            return 0
        fi
        echo -n "."
        sleep 3
        ((attempt++))
    done
    print_error "$service_name failed to start"
    return 1
}

# Check prerequisites
print_info "Checking prerequisites..."

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    print_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

print_info "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Check if we're in the right directory
if [ ! -d "gpu_services/llm_service" ]; then
    print_error "LLM service directory not found. Please run from the root project directory."
    exit 1
fi

# Step 1: Setup LLM Service (Local)
print_info "Step 1: Setting up Gemma 3-4B LLM Service (Local)..."

cd gpu_services/llm_service

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment for LLM Service..."
    
    # Check if setup script exists
    if [ -f "setup_local.sh" ]; then
        chmod +x setup_local.sh
        ./setup_local.sh
        print_success "LLM Service environment created ✓"
    else
        print_info "Running manual setup..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        
        print_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        print_info "Installing other requirements..."
        pip install -r requirements.txt
        
        print_success "LLM Service environment created manually ✓"
    fi
else
    print_info "Virtual environment already exists"
fi

# Activate environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f "../../.env.lenovo.template" ]; then
        cp ../../.env.lenovo.template .env
        print_success "Created .env file from template"
    else
        print_warning "No .env template found, using defaults"
        cat > .env << 'EOF'
MODEL_NAME=google/gemma-3-4b-it
DEVICE=cuda
TORCH_DTYPE=bfloat16
HOST=0.0.0.0
PORT=8001
GPU_MEMORY_FRACTION=0.95
EOF
    fi
fi

# Step 2: Start LLM Service
print_info "Step 2: Starting Gemma 3-4B LLM Service..."
print_warning "This may take several minutes for first-time model download..."

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

# Create log directory
mkdir -p ../../logs

# Start the service
if [ -f "start_local.sh" ]; then
    chmod +x start_local.sh
    print_info "Using start_local.sh script..."
    # Run in background and capture PID
    nohup ./start_local.sh > ../../logs/llm_service.log 2>&1 &
    LLM_PID=$!
else
    print_info "Starting LLM service directly..."
    nohup python main.py > ../../logs/llm_service.log 2>&1 &
    LLM_PID=$!
fi

echo $LLM_PID > ../../.llm.pid
print_info "LLM Service started (PID: $LLM_PID)"

cd ../..

# Step 3: Wait for LLM Service to be ready
print_info "Step 3: Waiting for LLM Service to load model and be ready..."
print_warning "This can take 2-5 minutes depending on your RTX 4070..."

if check_service "LLM Service" "http://localhost:8001/health"; then
    print_success "LLM Service is ready and loaded ✅"
    
    # Display model info
    print_info "Checking model information..."
    MODEL_INFO=$(curl -s http://localhost:8001/models 2>/dev/null || echo "Could not fetch model info")
    if [ "$MODEL_INFO" != "Could not fetch model info" ]; then
        echo "Model Info: $MODEL_INFO"
    fi
else
    print_error "LLM Service failed to start properly"
    print_info "Check the logs for details:"
    tail -n 20 logs/llm_service.log
    exit 1
fi

echo ""
print_success "=== Lenovo Laptop Services Started Successfully ==="
echo ""
echo "Service Status:"
echo "✅ Gemma 3-4B LLM Service (Local): http://localhost:8001"
echo ""
echo "Access Points:"
echo "- LLM API: http://localhost:8001/v1/chat/completions"
echo "- Health Check: http://localhost:8001/health"
echo "- API Documentation: http://localhost:8001/docs"
echo "- Model Info: http://localhost:8001/models"
echo ""
echo "Quick Test:"
echo 'curl -X POST "http://localhost:8001/v1/chat/completions" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'"'"
echo ""
echo "Logs:"
echo "- LLM Service: logs/llm_service.log"
echo "- Real-time logs: tail -f logs/llm_service.log"
echo ""
print_info "To stop the service, run: ./stop_lenovo_services.sh"
print_warning "GPU Memory usage: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MB"