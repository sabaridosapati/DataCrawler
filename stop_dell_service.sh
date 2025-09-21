#!/bin/bash
# setup_dell_laptop.sh
# Complete setup script for Dell Laptop (RTX 4050, 16GB RAM)
# Services: Embedding Service, Knowledge Graph Service, Milvus v2.6.2 Standalone

set -e

echo "=== Setting up Dell Laptop for Distributed Document Library ==="
echo "Services: Embedding Service (EmbeddingGemma), Knowledge Graph Service, Milvus v2.6.2"
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

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only."
    exit 1
fi

print_info "System: $(lsb_release -d -s 2>/dev/null || echo 'Linux')"

# Step 1: Check Prerequisites
print_info "Step 1: Checking prerequisites..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    print_info "Please install Python 3.10+ first"
    exit 1
fi

print_success "Python $PYTHON_VERSION found ✓"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    print_success "NVIDIA GPU detected ✓"
else
    print_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_success "CUDA $CUDA_VERSION found ✓"
else
    print_warning "CUDA nvcc not found in PATH. Please ensure CUDA toolkit is installed."
fi

# Step 2: Set up Milvus v2.6.2 Standalone
print_info "Step 2: Setting up Milvus v2.6.2 Standalone..."

if [ ! -f "setup_milvus_v2.6.2_standalone.sh" ]; then
    print_error "Milvus setup script not found. Please ensure setup_milvus_v2.6.2_standalone.sh is in the current directory."
    exit 1
fi

chmod +x setup_milvus_v2.6.2_standalone.sh
./setup_milvus_v2.6.2_standalone.sh

print_success "Milvus v2.6.2 standalone setup complete ✓"

# Step 3: Create project directory structure
print_info "Step 3: Creating project directory structure..."

mkdir -p gpu_services/embedding_service
mkdir -p gpu_services/knowledge_graph_service
mkdir -p logs
mkdir -p data
mkdir -p cache/huggingface

print_success "Directory structure created ✓"

# Step 4: Set up Embedding Service
print_info "Step 4: Setting up Enhanced Embedding Service..."

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
    print_info "Virtual environment already exists for Embedding Service"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file for Embedding Service..."
    cat > .env << 'EOF'
# Enhanced Embedding Service Configuration for Dell Laptop (RTX 4050)

# Model configuration
MODEL_NAME=google/embeddinggemma-300m
EMBEDDING_DIMENSION=768

# Hardware configuration optimized for RTX 4050 (6GB VRAM)
DEVICE=cuda
TORCH_DTYPE=bfloat16
BATCH_SIZE=16
MAX_SEQ_LENGTH=512
GPU_MEMORY_FRACTION=0.85

# Service configuration
HOST=0.0.0.0
PORT=8002

# Processing optimization
NORMALIZE_EMBEDDINGS=true
SHOW_PROGRESS=true
TRUST_REMOTE_CODE=true

# Text cleaning configuration
MIN_TEXT_LENGTH=10
MAX_TEXT_LENGTH=8192
CLEAN_DOCLING_ARTIFACTS=true

# Hugging Face configuration
HF_TOKEN=
TRANSFORMERS_CACHE=./cache/huggingface

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
EOF
    
    print_success "Created .env file for Embedding Service ✓"
fi

cd ../..

# Step 5: Set up Knowledge Graph Service
print_info "Step 5: Setting up Knowledge Graph Service..."

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
    print_info "Virtual environment already exists for Knowledge Graph Service"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file for Knowledge Graph Service..."
    cat > .env << 'EOF'
# Knowledge Graph Service Configuration for Dell Laptop

# Neo4j connection (Mac Mini - Update IP as needed)
NEO4J_URI=bolt://192.168.100.41:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Remote LLM Service (Lenovo Laptop - Update IP as needed) 
LOCAL_LLM_URL=http://192.168.100.43:8001/v1

# Local Embedding Service (Dell Laptop)
EMBEDDING_SERVICE_URL=http://localhost:8002/embed-documents

# Service configuration
HOST=0.0.0.0
PORT=8003

# Graph building parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_ENTITIES_PER_CHUNK=20

# LLM parameters for entity extraction
EXTRACTION_TEMPERATURE=0.0
MAX_EXTRACTION_TOKENS=2048
EOF
    
    print_warning "Created .env file for Knowledge Graph Service"
    print_error "IMPORTANT: Please update NEO4J_PASSWORD in gpu_services/knowledge_graph_service/.env"
fi

cd ../..

# Step 6: Install PyMilvus for vector database operations
print_info "Step 6: Installing PyMilvus..."

pip install --user "pymilvus==2.6.2"

print_success "PyMilvus installed ✓"

# Step 7: Create startup scripts
print_info "Step 7: Creating startup scripts..."

# Create Milvus startup script
cat > start_milvus.sh << 'EOF'
#!/bin/bash
# Start Milvus v2.6.2 standalone

echo "Starting Milvus v2.6.2 standalone..."

cd milvus-v2.6.2-standalone
./start_milvus.sh &

echo "Waiting for Milvus to initialize..."
sleep 30

# Test connection
python3 test_milvus_connection.py

cd ..

echo "Milvus v2.6.2 is ready!"
EOF

chmod +x start_milvus.sh

# Create services startup script
cat > start_services.sh << 'EOF'
#!/bin/bash
# Start all services for Dell laptop

echo "Starting Dell Laptop Services..."

# Step 1: Start Milvus v2.6.2
echo "Step 1: Starting Milvus v2.6.2..."
./start_milvus.sh

# Step 2: Start Embedding Service
echo "Step 2: Starting Enhanced Embedding Service..."
cd gpu_services/embedding_service
source venv/bin/activate
python main.py > ../../logs/embedding.log 2>&1 &
EMBEDDING_PID=$!
echo $EMBEDDING_PID > ../../.embedding.pid
echo "Enhanced Embedding Service started (PID: $EMBEDDING_PID)"
cd ../..

# Wait for embedding service to load model
echo "Waiting for embedding model to load..."
sleep 45

# Step 3: Start Knowledge Graph Service
echo "Step 3: Starting Knowledge Graph Service..."
cd gpu_services/knowledge_graph_service
source venv/bin/activate
python main.py > ../../logs/knowledge_graph.log 2>&1 &
KNOWLEDGE_GRAPH_PID=$!
echo $KNOWLEDGE_GRAPH_PID > ../../.knowledge_graph.pid
echo "Knowledge Graph Service started (PID: $KNOWLEDGE_GRAPH_PID)"
cd ../..

echo ""
echo "Dell Laptop services started successfully!"
echo ""
echo "Services running:"
echo "  ✅ Milvus v2.6.2: localhost:19530"
echo "  ✅ Enhanced Embedding Service: http://localhost:8002"
echo "  ✅ Knowledge Graph Service: http://localhost:8003"
echo ""
echo "Service Health Checks:"
echo "  curl http://localhost:8002/health"
echo "  curl http://localhost:8003/health"
echo ""
echo "Milvus Web UI: http://localhost:9001 (minioadmin/minioadmin)"
EOF

chmod +x start_services.sh

# Create stop services script
cat > stop_services.sh << 'EOF'
#!/bin/bash
# Stop all services for Dell laptop

echo "Stopping Dell Laptop Services..."

# Stop Knowledge Graph Service
if [ -f ".knowledge_graph.pid" ]; then
    KNOWLEDGE_GRAPH_PID=$(cat .knowledge_graph.pid)
    if kill -0 $KNOWLEDGE_GRAPH_PID 2>/dev/null; then
        kill $KNOWLEDGE_GRAPH_PID
        echo "Knowledge Graph Service stopped (PID: $KNOWLEDGE_GRAPH_PID)"
    fi
    rm .knowledge_graph.pid
fi

# Stop Embedding Service
if [ -f ".embedding.pid" ]; then
    EMBEDDING_PID=$(cat .embedding.pid)
    if kill -0 $EMBEDDING_PID 2>/dev/null; then
        kill $EMBEDDING_PID
        echo "Enhanced Embedding Service stopped (PID: $EMBEDDING_PID)"
    fi
    rm .embedding.pid
fi

# Stop Milvus
echo "Stopping Milvus v2.6.2..."
cd milvus-v2.6.2-standalone
./stop_milvus.sh
cd ..

echo "Dell Laptop services stopped"
EOF

chmod +x stop_services.sh

# Step 8: Create health check script
print_info "Step 8: Creating health check script..."

cat > health_check.sh << 'EOF'
#!/bin/bash
# Health check for Dell Laptop services

echo "=== Dell Laptop Services Health Check ==="

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo ""

# Check Milvus
if netstat -tuln | grep -q ":19530 "; then
    echo "✅ Milvus v2.6.2 is running on port 19530"
else
    echo "❌ Milvus is not running"
fi

# Check Enhanced Embedding Service
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ Enhanced Embedding Service is running on port 8002"
    # Get model info
    curl -s http://localhost:8002/model-info | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'   Model: {data[\"model_name\"]} ({data[\"embedding_dimension\"]}d)')"
else
    echo "❌ Enhanced Embedding Service is not running"
fi

# Check Knowledge Graph Service
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "✅ Knowledge Graph Service is running on port 8003"
else
    echo "❌ Knowledge Graph Service is not running"
fi

echo ""
echo "Service URLs:"
echo "  - Enhanced Embedding: http://localhost:8002"
echo "  - Knowledge Graph: http://localhost:8003" 
echo "  - Milvus: localhost:19530"
echo "  - MinIO UI: http://localhost:9001"
echo ""
echo "API Documentation:"
echo "  - http://localhost:8002/docs"
echo "  - http://localhost:8003/docs"
EOF

chmod +x health_check.sh

echo ""
print_success "=== Dell Laptop Setup Complete! ==="
echo ""
echo "🖥️  Milvus v2.6.2 standalone installed"
echo "🧠 Enhanced Embedding Service (EmbeddingGemma-300M) ready"
echo "🕸️  Knowledge Graph Service ready"
echo "🔧 All startup scripts created"
echo ""
print_warning "IMPORTANT NEXT STEPS:"
echo ""
echo "1. Update configuration files:"
echo "   - gpu_services/embedding_service/.env: Configure GPU settings if needed"
echo "   - gpu_services/knowledge_graph_service/.env: Set NEO4J_PASSWORD and network IPs"
echo ""
echo "2. Start the services:"
echo "   ./start_services.sh"
echo ""
echo "3. Verify everything is working:"
echo "   ./health_check.sh"
echo ""
echo "4. Test embedding service:"
echo '   curl -X POST "http://localhost:8002/embed-documents" -H "Content-Type: application/json" -d '"'"'{"texts": ["test document"]}'"'"
echo ""
print_info "The Dell Laptop is now ready to handle embedding and vector operations!"