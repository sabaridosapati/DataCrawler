#!/bin/bash
# setup_milvus_v2.6.2.sh
# Exact Milvus v2.6.2 setup script for Dell Laptop as per official documentation

set -e

echo "=== Setting up Milvus v2.6.2 (Exact Official Version) ==="
echo "This script downloads and sets up the exact Milvus v2.6.2 configuration"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Docker installation
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    print_error "Docker Compose V2 is not available. Please install Docker Desktop with Compose V2."
    exit 1
fi

print_status "Docker and Docker Compose V2 found ✓"

# Create Milvus directory
MILVUS_DIR="milvus-v2.6.2"
if [ ! -d "$MILVUS_DIR" ]; then
    mkdir -p "$MILVUS_DIR"
    print_status "Created Milvus directory: $MILVUS_DIR"
fi

cd "$MILVUS_DIR"

# Download the exact v2.6.2 configuration file as per documentation
print_status "Downloading Milvus v2.6.2 standalone Docker Compose configuration..."
if [ -f "docker-compose.yml" ]; then
    print_warning "docker-compose.yml already exists, backing up..."
    mv docker-compose.yml "docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Use wget to download the exact configuration from Milvus releases
wget https://github.com/milvus-io/milvus/releases/download/v2.6.2/milvus-standalone-docker-compose.yml -O docker-compose.yml

if [ $? -eq 0 ]; then
    print_status "Successfully downloaded Milvus v2.6.2 Docker Compose configuration ✓"
else
    print_error "Failed to download Milvus configuration. Check your internet connection."
    exit 1
fi

# Create volumes directory as per Milvus documentation
mkdir -p volumes/etcd
mkdir -p volumes/minio
mkdir -p volumes/milvus

print_status "Created volume directories for Milvus data"

# Display the downloaded configuration
print_status "Milvus v2.6.2 Configuration:"
echo "=========================="
cat docker-compose.yml
echo "=========================="
echo ""

# Start Milvus v2.6.2
print_status "Starting Milvus v2.6.2 services..."
echo "This may take a few minutes for first-time setup..."

docker compose up -d

if [ $? -eq 0 ]; then
    print_status "Milvus v2.6.2 started successfully ✓"
    echo ""
    print_status "Waiting for services to initialize..."
    sleep 30
    
    print_status "Checking service status..."
    docker compose ps
    
    echo ""
    print_status "Milvus v2.6.2 Setup Complete!"
    echo ""
    echo "Service Access Points:"
    echo "- Milvus gRPC: localhost:19530"
    echo "- Milvus WebUI: http://127.0.0.1:9091/webui/"
    echo "- MinIO Console: http://localhost:9090 (minioadmin/minioadmin)"
    echo ""
    echo "Data is stored in: $(pwd)/volumes/"
    echo ""
    print_status "You can now test the connection with Python:"
    echo "from pymilvus import connections"
    echo "connections.connect(host='localhost', port='19530')"
    echo ""
    print_warning "To stop Milvus: docker compose down"
    print_warning "To remove all data: docker compose down && sudo rm -rf volumes"
    
else
    print_error "Failed to start Milvus v2.6.2. Check the logs:"
    docker compose logs
    exit 1
fi

cd ..

print_status "Milvus v2.6.2 is now running and ready for use!"