#!/bin/bash
# stop_dell_services.sh
# Stop script for Dell Laptop - Mixed Deployment

echo "=== Stopping Dell Laptop Services ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# Stop Knowledge Graph Service (Local)
print_info "Stopping Knowledge Graph Service (Local)..."
if [ -f ".knowledge_graph.pid" ]; then
    KNOWLEDGE_GRAPH_PID=$(cat .knowledge_graph.pid)
    if kill -0 $KNOWLEDGE_GRAPH_PID 2>/dev/null; then
        kill $KNOWLEDGE_GRAPH_PID
        print_success "Knowledge Graph Service stopped (PID: $KNOWLEDGE_GRAPH_PID)"
    else
        print_warning "Knowledge Graph Service was not running"
    fi
    rm .knowledge_graph.pid
else
    print_warning "No Knowledge Graph PID file found"
fi

# Stop Embedding Service (Local)
print_info "Stopping Embedding Service (Local)..."
if [ -f ".embedding.pid" ]; then
    EMBEDDING_PID=$(cat .embedding.pid)
    if kill -0 $EMBEDDING_PID 2>/dev/null; then
        kill $EMBEDDING_PID
        print_success "Embedding Service stopped (PID: $EMBEDDING_PID)"
    else
        print_warning "Embedding Service was not running"
    fi
    rm .embedding.pid
else
    print_warning "No Embedding PID file found"
fi

# Stop Milvus v2.6.2 (Docker)
print_info "Stopping Milvus v2.6.2 (Docker)..."
if [ -d "milvus-v2.6.2" ]; then
    cd milvus-v2.6.2
    docker compose down
    if [ $? -eq 0 ]; then
        print_success "Milvus v2.6.2 Docker services stopped ✓"
    else
        print_warning "Some Milvus services may not have stopped cleanly"
    fi
    cd ..
else
    print_warning "Milvus v2.6.2 directory not found"
    # Fallback to stop by container names
    docker stop milvus-standalone milvus-minio milvus-etcd 2>/dev/null || true
fi

echo ""
print_success "Dell Laptop services stopped"
echo ""
echo "What was stopped:"
echo "- Knowledge Graph Service (Local)"
echo "- Embedding Service (Local)"  
echo "- Milvus v2.6.2 Vector Database (Docker)"
echo "- MinIO Object Storage (Docker)"
echo "- etcd Metadata Store (Docker)"