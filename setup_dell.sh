#!/bin/bash
# start_dell_services.sh
# Startup script for Dell Laptop - Mixed Deployment

set -e

echo "=== Starting Dell Laptop Services (Mixed Deployment) ==="
echo "Docker: Milvus v2.6.2 Vector Database"
echo "Local: Embedding Service + Knowledge Graph Service"
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
    local max_attempts=30
    local attempt=1
    
    print_info "Checking $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$health_url" > /dev/null 2>&1; then
            print_success "$service_name is running ✓"
            return 0
        fi
        echo -n "."
        sleep 2
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

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

# Step 1: Start Milvus v2.6.2 (Docker)
print_info "Step 1: Starting Milvus v2.6.2 (Docker)..."

if [ ! -d "milvus-v2.6.2" ]; then
    print_error "Milvus v2.6.2 setup not found. Please run: ./setup_milvus_v2.6.2.sh first"
    exit 1
fi

cd milvus-v2.6.2
docker compose up -d

if [ $? -eq 0 ]; then
    print_success "Milvus v2.6.2 Docker services started ✓"
else
    print_error "Failed to start Milvus v2.6.2"
    exit 1
fi

cd ..

# Wait for Milvus to be ready
print_info "Waiting for Milvus v2.6.2 to initialize..."
sleep 45

if ! check_service "Milvus v2.6.2" "http://localhost:9091/healthz"; then
    print_error "Milvus v2.6.2 failed to start. Check Docker logs:"
    cd milvus-v2.6.2
    docker compose logs milvus-standalone
    cd ..
    exit 1
fi

# Step 2: Setup and Start Embedding Service (Local)
print_info "Step 2: Setting up Embedding Service (Local)..."

cd gpu_services/embedding_service

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment for Embedding Service..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    print_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_info "Installing other requirements..."
    pip install -r requirements.txt
    
    print_success "Embedding Service environment created ✓"
else
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    cp ../../.env.dell.template .env
    print_warning "Created .env file. Please edit with your Neo4j Desktop password."
    print_error "REQUIRED: Update NEO4J_PASSWORD in .env"
    exit 1
fi

# Start embedding service in background
print_info "Starting Embedding Service (Local)..."
python main.py > ../../logs/embedding.log 2>&1 &
EMBEDDING_PID=$!
echo $EMBEDDING_PID > ../../.embedding.pid
print_info "Embedding Service started (PID: $EMBEDDING_PID)"

cd ../..

# Wait for embedding service to be ready
sleep 30
if ! check_service "Embedding Service" "http://localhost:8002/health"; then
    print_error "Embedding Service failed to start. Check logs/embedding.log"
    exit 1
fi

# Step 3: Setup and Start Knowledge Graph Service (Local)
print_info "Step 3: Setting up Knowledge Graph Service (Local)..."

cd gpu_services/knowledge_graph_service

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment for Knowledge Graph Service..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Knowledge Graph Service environment created ✓"
else
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    cp ../../.env.dell.template .env
    print_warning "Created .env file. Please edit with your Neo4j Desktop password."
    print_error "REQUIRED: Update NEO4J_PASSWORD in .env"
    exit 1
fi

# Start knowledge graph service in background
print_info "Starting Knowledge Graph Service (Local)..."
python main.py > ../../logs/knowledge_graph.log 2>&1 &
KNOWLEDGE_GRAPH_PID=$!
echo $KNOWLEDGE_GRAPH_PID > ../../.knowledge_graph.pid
print_info "Knowledge Graph Service started (PID: $KNOWLEDGE_GRAPH_PID)"

cd ../..

# Wait for knowledge graph service to be ready
sleep 15
if ! check_service "Knowledge Graph Service" "http://localhost:8003/health"; then
    print_error "Knowledge Graph Service failed to start. Check logs/knowledge_graph.log"
    exit 1
fi

echo ""
print_success "=== Dell Laptop Services Started Successfully ==="
echo ""
echo "Service Status:"
echo "✅ Milvus v2.6.2 (Docker): http://localhost:19530"
echo "✅ Milvus WebUI: http://127.0.0.1:9091/webui/"
echo "✅ MinIO Console: http://localhost:9090 (minioadmin/minioadmin)"
echo "✅ Embedding Service (Local): http://localhost:8002"
echo "✅ Knowledge Graph Service (Local): http://localhost:8003"
echo ""
echo "Health Checks:"
echo "curl http://localhost:8002/health"
echo "curl http://localhost:8003/health"
echo "curl http://localhost:9091/healthz"
echo ""
echo "Logs:"
echo "- Embedding Service: logs/embedding.log"
echo "- Knowledge Graph Service: logs/knowledge_graph.log"
echo "- Milvus Docker: cd milvus-v2.6.2 && docker compose logs -f"
echo ""
print_info "To stop services, run: ./stop_dell_services.sh"
print_warning "Ensure Lenovo LLM service (8001) and Mac Neo4j (7687) are accessible!"