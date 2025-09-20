#!/bin/bash
# stop_mac_services.sh
# Stop script for Mac Mini - Mixed Deployment

echo "=== Stopping Mac Mini Services ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# Stop Docling Service (Local)
print_info "Stopping Docling Service (Local)..."
if [ -f ".docling.pid" ]; then
    DOCLING_PID=$(cat .docling.pid)
    if kill -0 $DOCLING_PID 2>/dev/null; then
        kill $DOCLING_PID
        print_success "Docling Service stopped (PID: $DOCLING_PID)"
    else
        print_warning "Docling Service was not running"
    fi
    rm .docling.pid
else
    print_warning "No Docling PID file found"
    # Try to kill by process name
    pkill -f "python main.py" || true
fi

# Stop Docker Services (MongoDB + Orchestrator)
print_info "Stopping Docker Services..."
docker-compose -f docker-compose.mac.yml down

if [ $? -eq 0 ]; then
    print_success "Docker services stopped ✓"
else
    print_warning "Some Docker services may not have stopped cleanly"
fi

# Note about Neo4j Desktop
print_info "Neo4j Desktop is still running (not stopped automatically)"
print_info "Stop it manually from Neo4j Desktop if needed"

echo ""
print_success "Mac Mini services stopped"
echo ""
echo "What's still running:"
echo "- Neo4j Desktop (stop manually if needed)"
echo ""
echo "What was stopped:"
echo "- Docling Service (Local)"
echo "- Orchestrator API (Docker)"
echo "- MongoDB (Docker)"