#!/bin/bash
# start_mac_services.sh
# Startup script for Mac Mini - Mixed Deployment

set -e

echo "=== Starting Mac Mini Services (Mixed Deployment) ==="
echo "Docker: Orchestrator API + MongoDB"
echo "Local: Docling Service + Neo4j Desktop"
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

# Step 1: Check Neo4j Desktop
print_info "Step 1: Checking Neo4j Desktop..."
if curl -s -u neo4j:password http://localhost:7474/db/data/ > /dev/null 2>&1; then
    print_success "Neo4j Desktop is running ✓"
else
    print_warning "Neo4j Desktop may not be running"
    print_info "Please start Neo4j Desktop manually and ensure it's running on port 7687"
    read -p "Press Enter when Neo4j Desktop is running..."
fi

# Step 2: Setup Docling Service (Local)
print_info "Step 2: Setting up Docling Service (Local)..."

if [ ! -d "gpu_services/docling_service/venv" ]; then
    print_info "Creating virtual environment for Docling..."
    cd gpu_services/docling_service
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ../..
    print_success "Docling environment created ✓"
fi

# Start Docling Service in background
print_info "Starting Docling Service (Local)..."
cd gpu_services/docling_service
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    cp ../../.env.mac-mini.template .env
    print_warning "Created .env file. Please edit if needed."
fi

# Start docling service in background
python main.py > ../../logs/docling.log 2>&1 &
DOCLING_PID=$!
echo $DOCLING_PID > ../../.docling.pid
print_info "Docling Service started (PID: $DOCLING_PID)"

cd ../..

# Wait for Docling to be ready
sleep 10
if ! check_service "Docling Service" "http://localhost:8004/health"; then
    print_error "Docling Service failed to start. Check logs/docling.log"
    exit 1
fi

# Step 3: Start Docker Services (MongoDB + Orchestrator)
print_info "Step 3: Starting Docker Services (MongoDB + Orchestrator)..."

# Check if .env exists for orchestrator
if [ ! -f "orchestrator_api/.env" ]; then
    cp .env.mac-mini.template orchestrator_api/.env
    print_warning "Created orchestrator_api/.env. Please edit with your settings."
    print_error "REQUIRED: Update NEO4J_PASSWORD in orchestrator_api/.env"
    exit 1
fi

# Start Docker services
docker-compose -f docker-compose.mac.yml up -d

if [ $? -eq 0 ]; then
    print_success "Docker services started ✓"
else
    print_error "Failed to start Docker services"
    exit 1
fi

# Step 4: Wait for all services to be ready
print_info "Step 4: Waiting for all services to be ready..."

sleep 20

# Check MongoDB
if check_service "MongoDB" "http://localhost:27017"; then
    print_success "MongoDB is ready ✓"
else
    print_warning "MongoDB may not be ready"
fi

# Check Orchestrator API
if check_service "Orchestrator API" "http://localhost:8000/health"; then
    print_success "Orchestrator API is ready ✓"
else
    print_error "Orchestrator API failed to start"
    docker-compose -f docker-compose.mac.yml logs orchestrator_api
    exit 1
fi

echo ""
print_success "=== Mac Mini Services Started Successfully ==="
echo ""
echo "Service Status:"
echo "✅ Neo4j Desktop: bolt://localhost:7687"
echo "✅ MongoDB: mongodb://localhost:27017"  
echo "✅ Docling Service (Local): http://localhost:8004"
echo "✅ Orchestrator API (Docker): http://localhost:8000"
echo ""
echo "Access Points:"
echo "- Main API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Docling Health: http://localhost:8004/health"
echo ""
echo "Logs:"
echo "- Docling (Local): logs/docling.log"
echo "- Docker Services: docker-compose -f docker-compose.mac.yml logs -f"
echo ""
print_info "To stop services, run: ./stop_mac_services.sh"