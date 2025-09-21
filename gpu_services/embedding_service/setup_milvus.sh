#!/bin/bash
# setup_milvus_v2.6.2_standalone.sh
# Milvus v2.6.2 standalone installation script for Dell Laptop (Linux)

set -e

echo "=== Setting up Milvus v2.6.2 Standalone (Non-Docker) for Dell Laptop ==="
echo "This script installs Milvus v2.6.2 standalone version with GPU support on Linux"
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

# Check for required tools
print_info "Checking prerequisites..."

# Check if wget is available
if ! command -v wget &> /dev/null; then
    print_error "wget is required but not installed. Please install wget first."
    exit 1
fi

# Check if GPU is available (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "NVIDIA drivers not found. Milvus will run on CPU mode."
fi

# Create Milvus directory
MILVUS_DIR="milvus-v2.6.2-standalone"
MILVUS_HOME="$HOME/milvus"

print_info "Creating Milvus installation directory: $MILVUS_DIR"
mkdir -p "$MILVUS_DIR"
cd "$MILVUS_DIR"

# Download Milvus v2.6.2 standalone binary
print_info "Downloading Milvus v2.6.2 standalone binary..."
MILVUS_URL="https://github.com/milvus-io/milvus/releases/download/v2.6.2/milvus-standalone-2.6.2-linux-amd64.tar.gz"

if [ ! -f "milvus-standalone-2.6.2-linux-amd64.tar.gz" ]; then
    wget "$MILVUS_URL" -O milvus-standalone-2.6.2-linux-amd64.tar.gz
    if [ $? -eq 0 ]; then
        print_success "Milvus binary downloaded successfully"
    else
        print_error "Failed to download Milvus binary"
        exit 1
    fi
else
    print_info "Milvus binary already exists, skipping download"
fi

# Extract Milvus binary
print_info "Extracting Milvus binary..."
tar -xzf milvus-standalone-2.6.2-linux-amd64.tar.gz

# Make Milvus executable
chmod +x milvus

# Create Milvus configuration directory and files
print_info "Creating Milvus configuration..."
mkdir -p configs
mkdir -p logs
mkdir -p data/db
mkdir -p data/wal

# Create Milvus configuration file
cat > configs/milvus.yaml << 'EOF'
# Milvus v2.6.2 Standalone Configuration
# Optimized for Dell laptop with RTX 4050

# System configuration
etcd:
  endpoints: 
    - localhost:2379
  rootPath: by-dev
  metaSubPath: meta
  kvSubPath: kv

minio:
  address: localhost
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  useSSL: false
  bucketName: a-bucket

# Storage configuration
localStorage:
  path: ./data/db

# WAL configuration  
wal:
  path: ./data/wal

# Log configuration
log:
  level: info
  file:
    rootPath: ./logs
    maxSize: 300
    maxAge: 10
    maxBackups: 20

# GPU configuration (if available)
gpu:
  initMemPool: 1024
  maxMemPool: 2048

# Server configuration
grpc:
  serverMaxRecvSize: 268435456  # 256 MB
  serverMaxSendSize: 268435456  # 256 MB
  clientMaxRecvSize: 268435456  # 256 MB
  clientMaxSendSize: 268435456  # 256 MB

# Performance tuning for RTX 4050
queryNode:
  gracefulTime: 1000
  gracefulStopTimeout: 30
  
indexNode:
  gracefulTime: 1000
  gracefulStopTimeout: 30

# Data node configuration  
dataNode:
  gracefulTime: 1000
  gracefulStopTimeout: 30
  
# Root coordinator
rootCoord:
  minSegmentSizeToEnableIndex: 1024
  
# Query coordinator
queryCoord:
  autoHandoff: true
  autoBalance: true

# Index coordinator
indexCoord:
  gc:
    interval: 600
EOF

# Create startup script
print_info "Creating Milvus startup script..."
cat > start_milvus.sh << 'EOF'
#!/bin/bash
# Start Milvus v2.6.2 standalone

echo "Starting Milvus v2.6.2 standalone..."

# Check if etcd is running
if ! pgrep -f "etcd" > /dev/null; then
    echo "Starting etcd..."
    etcd --data-dir ./data/etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379 &
    sleep 5
fi

# Check if MinIO is running  
if ! pgrep -f "minio" > /dev/null; then
    echo "Starting MinIO..."
    mkdir -p ./data/minio
    minio server ./data/minio --address ":9000" --console-address ":9001" &
    sleep 5
fi

# Start Milvus
echo "Starting Milvus server..."
export MILVUS_CONFIG_PATH="./configs/milvus.yaml"
./milvus run standalone

EOF

chmod +x start_milvus.sh

# Create stop script
print_info "Creating Milvus stop script..."
cat > stop_milvus.sh << 'EOF'
#!/bin/bash
# Stop Milvus v2.6.2 standalone

echo "Stopping Milvus v2.6.2 standalone..."

# Stop Milvus
pkill -f "milvus"

# Stop MinIO
pkill -f "minio"

# Stop etcd
pkill -f "etcd"

echo "Milvus stopped."
EOF

chmod +x stop_milvus.sh

# Install required dependencies
print_info "Installing required dependencies..."

# Check if etcd is installed
if ! command -v etcd &> /dev/null; then
    print_info "Installing etcd..."
    ETCD_VER=v3.5.5
    GITHUB_URL=https://github.com/etcd-io/etcd/releases/download
    DOWNLOAD_URL=${GITHUB_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz
    
    wget ${DOWNLOAD_URL} -O etcd-${ETCD_VER}-linux-amd64.tar.gz
    tar -xzf etcd-${ETCD_VER}-linux-amd64.tar.gz
    sudo mv etcd-${ETCD_VER}-linux-amd64/etcd* /usr/local/bin/
    rm -rf etcd-${ETCD_VER}-linux-amd64*
    
    print_success "etcd installed"
else
    print_info "etcd already installed"
fi

# Check if MinIO is installed
if ! command -v minio &> /dev/null; then
    print_info "Installing MinIO..."
    wget https://dl.min.io/server/minio/release/linux-amd64/minio
    chmod +x minio
    sudo mv minio /usr/local/bin/
    print_success "MinIO installed"
else
    print_info "MinIO already installed"
fi

# Create Python client test script
print_info "Creating Python client test script..."
cat > test_milvus_connection.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Milvus v2.6.2 standalone installation and connection.
"""

import sys
try:
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
except ImportError:
    print("ERROR: pymilvus not installed. Please install with: pip install pymilvus==2.6.2")
    sys.exit(1)

def test_milvus_connection():
    """Test Milvus connection and basic operations"""
    try:
        # Connect to Milvus
        print("Connecting to Milvus...")
        connections.connect(
            alias="default",
            host='localhost',
            port='19530'
        )
        
        # Check if connection is successful
        if connections.has_connection("default"):
            print("✅ Successfully connected to Milvus v2.6.2!")
        
        # Get server version
        print(f"📋 Milvus server version: {utility.get_server_version()}")
        
        # List existing collections
        collections = utility.list_collections()
        print(f"📂 Existing collections: {collections}")
        
        print("\n🎉 Milvus v2.6.2 standalone is working correctly!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Milvus is running: ./start_milvus.sh")
        print("2. Check if port 19530 is available")
        print("3. Check Milvus logs in ./logs/ directory")
    finally:
        if connections.has_connection("default"):
            connections.disconnect("default")

if __name__ == "__main__":
    test_milvus_connection()
EOF

chmod +x test_milvus_connection.py

cd ..

# Final setup summary
echo ""
print_success "=== Milvus v2.6.2 Standalone Setup Complete! ==="
echo ""
echo "📁 Installation directory: $(pwd)/$MILVUS_DIR"
echo ""
echo "🚀 To start Milvus:"
echo "   cd $MILVUS_DIR && ./start_milvus.sh"
echo ""
echo "🛑 To stop Milvus:"
echo "   cd $MILVUS_DIR && ./stop_milvus.sh"
echo ""
echo "🧪 To test the installation:"
echo "   cd $MILVUS_DIR && python3 test_milvus_connection.py"
echo ""
echo "📊 Service endpoints:"
echo "   - Milvus gRPC: localhost:19530"
echo "   - MinIO Web UI: http://localhost:9001 (minioadmin/minioadmin)"
echo "   - etcd: localhost:2379"
echo ""
echo "📝 Configuration file: $MILVUS_DIR/configs/milvus.yaml"
echo "📋 Logs directory: $MILVUS_DIR/logs/"
echo ""
print_warning "Important: Make sure to install pymilvus==2.6.2 in your Python environment:"
echo "pip install pymilvus==2.6.2"
echo ""
print_info "Milvus v2.6.2 standalone is now ready for your document library system!"