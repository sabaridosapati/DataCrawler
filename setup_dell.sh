#!/bin/bash
# setup_dell_laptop.sh
# Automated setup script for Dell Laptop (Embedding & Vector Services)

set -e

echo "=== Dell Laptop Setup (Embedding & Vector Services) ==="
echo "This script will set up embedding service, knowledge graph service, and Milvus vector database"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
print_status "Checking prerequisites..."

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    print_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

print_status "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]; then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

print_status "Python version: $PYTHON_VERSION ✓"

# Check Docker (for Milvus dependencies)
if ! command -v docker &> /dev/null; then
    print_error "Docker is required for Milvus. Please install Docker first."
    exit 1
fi

print_status "Docker found ✓"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p data/milvus
mkdir -p data/etcd
mkdir -p data/minio
mkdir -p logs
mkdir -p gpu_services/embedding_service/logs
mkdir -p gpu_services/knowledge_graph_service/logs
mkdir -p cache/huggingface

# Setup environment files
print_status "Setting up environment files..."

if [ ! -f "gpu_services/embedding_service/.env" ]; then
    cp .env.dell.template gpu_services/embedding_service/.env
    print_status "Created embedding service .env from template"
fi

if [ ! -f "gpu_services/knowledge_graph_service/.env" ]; then
    cp .env.dell.template gpu_services/knowledge_graph_service/.env
    print_status "Created knowledge graph service .env from template"
fi

# Setup embedding service
print_status "Setting up Embedding Service..."
cd gpu_services/embedding_service

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Created virtual environment for embedding service"
fi

source venv/bin/activate
pip install --upgrade pip

print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print_status "Installing other embedding service requirements..."
pip install -r requirements.txt

cd ../..

# Setup knowledge graph service
print_status "Setting up Knowledge Graph Service..."
cd gpu_services/knowledge_graph_service

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Created virtual environment for knowledge graph service"
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd ../..

# Setup Milvus v2.6.2 (exact official version)
print_status "Setting up Milvus v2.6.2 vector database..."
if [ ! -f "setup_milvus_v2.6.2.sh" ]; then
    print_error "Milvus setup script not found. Please ensure setup_milvus_v2.6.2.sh is available."
    exit 1
fi

chmod +x setup_milvus_v2.6.2.sh
print_status "Running Milvus v2.6.2 setup script..."
# Note: The Milvus script will be run separately as it needs Docker

print_status "Milvus v2.6.2 setup script is ready to run."
print_warning "After completing this setup, run: ./setup_milvus_v2.6.2.sh"

# Create service startup script
cat > start_services.sh << 'EOF'
#!/bin/bash
echo "Starting Dell Laptop Services..."

# Function to run service in background
run_service() {
    local name=$1
    local command=$2
    local logfile=$3
    local workdir=$4
    
    echo "Starting $name..."
    if [ -n "$workdir" ]; then
        cd "$workdir"
    fi
    $command > "$logfile" 2>&1 &
    local pid=$!
    echo "$name PID: $pid (log: $logfile)"
    echo $pid > ".$name.pid"
    if [ -n "$workdir" ]; then
        cd - > /dev/null
    fi
}

# Start Milvus v2.6.2 first
echo "Starting Milvus v2.6.2 vector database..."
if [ ! -d "milvus-v2.6.2" ]; then
    print_error "Milvus v2.6.2 not found. Please run ./setup_milvus_v2.6.2.sh first"
    exit 1
fi

cd milvus-v2.6.2
docker compose up -d
cd ..

echo "Waiting for Milvus v2.6.2 to be ready..."
sleep 45

# Start embedding service
cd gpu_services/embedding_service
source venv/bin/activate
run_service "embedding" "python main.py" "../../logs/embedding.log" "."
cd ../..

# Wait for embedding service to initialize
sleep 15

# Start knowledge graph service
cd gpu_services/knowledge_graph_service
source venv/bin/activate
run_service "knowledge_graph" "python main.py" "../../logs/knowledge_graph.log" "."
cd ../..

echo ""
echo "All Dell laptop services started!"
echo ""
echo "Service endpoints:"
echo "- Embedding Service: http://localhost:8002"
echo "- Knowledge Graph Service: http://localhost:8003"
echo "- Milvus: localhost:19530"
echo "- MinIO Console: http://localhost:9001"
echo ""
echo "Logs:"
echo "- Embedding: logs/embedding.log"
echo "- Knowledge Graph: logs/knowledge_graph.log"
echo ""
echo "Health checks:"
echo "curl http://localhost:8002/health"
echo "curl http://localhost:8003/health"
echo ""
echo "To stop services: ./stop_services.sh"
EOF

chmod +x start_services.sh

# Create stop script
cat > stop_services.sh << 'EOF'
#!/bin/bash
echo "Stopping Dell Laptop Services..."

# Function to stop service by PID
stop_service() {
    local name=$1
    local pidfile=".$name.pid"
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 $pid 2>/dev/null; then
            echo "Stopping $name (PID: $pid)..."
            kill $pid
            rm "$pidfile"
        else
            echo "$name is not running"
            rm "$pidfile"
        fi
    else
        echo "No PID file for $name"
    fi
}

stop_service "knowledge_graph"
stop_service "embedding"

# Stop Milvus v2.6.2 containers
echo "Stopping Milvus v2.6.2 containers..."
if [ -d "milvus-v2.6.2" ]; then
    cd milvus-v2.6.2
    docker compose down
    cd ..
else
    # Fallback to stop by container names
    docker stop milvus-standalone milvus-minio milvus-etcd 2>/dev/null || true
fi

echo "All services stopped."
EOF

chmod +x stop_services.sh

print_status "Setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Edit gpu_services/embedding_service/.env with your configuration"
print_status "2. Edit gpu_services/knowledge_graph_service/.env with your configuration" 
print_status "3. Update IP addresses to match your network setup"
print_status "4. Run: ./start_services.sh"
print_status ""
print_status "For Docker deployment instead:"
print_status "docker-compose -f docker-compose.dell.yml up -d"
print_status ""
print_warning "Make sure the remote LLM service (Lenovo) is running first!"
print_warning "Also ensure Neo4j (Mac Mini) is accessible from this machine!"