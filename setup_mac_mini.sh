#!/bin/bash
# setup_mac_mini.sh
# Complete setup script for Mac Mini (16GB RAM) - Mixed Deployment
# Services: Orchestrator API, Docling Service (granite-docling-258M-mlx), MongoDB, Neo4j Desktop

set -e

echo "=== Setting up Mac Mini for Distributed Document Library ==="
echo "Services: Orchestrator API, Granite Docling Service, MongoDB, Neo4j Desktop"
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

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS (Mac Mini) only."
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    print_warning "This script is optimized for Apple Silicon (M-series chips). You have: $ARCH"
fi

print_info "System: macOS $(sw_vers -productVersion) on $ARCH"

# Step 1: Check Prerequisites
print_info "Step 1: Checking prerequisites..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    print_info "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

print_success "Python $PYTHON_VERSION found ✓"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

print_success "Homebrew found ✓"

# Step 2: Install MongoDB
print_info "Step 2: Setting up MongoDB..."

if ! command -v mongod &> /dev/null; then
    print_info "Installing MongoDB..."
    brew tap mongodb/brew
    brew install mongodb-community
else
    print_info "MongoDB already installed"
fi

# Create MongoDB data directory
sudo mkdir -p /opt/homebrew/var/mongodb
sudo chown $(whoami) /opt/homebrew/var/mongodb

# Start MongoDB service
print_info "Starting MongoDB service..."
brew services start mongodb/brew/mongodb-community

print_success "MongoDB setup complete ✓"

# Step 3: Check Neo4j Desktop
print_info "Step 3: Checking Neo4j Desktop..."

# Check if Neo4j Desktop is running
if lsof -i :7687 > /dev/null 2>&1; then
    print_success "Neo4j Desktop is running on port 7687 ✓"
else
    print_warning "Neo4j Desktop not detected on port 7687"
    print_info "Please ensure Neo4j Desktop is installed and running"
    print_info "Download from: https://neo4j.com/download-center/#desktop"
    print_warning "Make sure to start your Neo4j database before proceeding"
fi

# Step 4: Create project directory structure
print_info "Step 4: Creating project directory structure..."

# Create necessary directories
mkdir -p data/user_files
mkdir -p data/extracted
mkdir -p data/mongodb
mkdir -p logs
mkdir -p orchestrator_api/app/{api,core,db,models,services}
mkdir -p gpu_services/docling_service

print_success "Directory structure created ✓"

# Step 5: Set up Orchestrator API
print_info "Step 5: Setting up Orchestrator API..."

cd orchestrator_api

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment for Orchestrator API..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Orchestrator API environment created ✓"
else
    print_info "Virtual environment already exists for Orchestrator API"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file for Orchestrator API..."
    cat > .env << 'EOF'
# Orchestrator API Configuration for Mac Mini

# Security
SECRET_KEY=your-very-secure-secret-key-change-this-in-production-at-least-32-chars
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Local Database Connections (Mac Mini)
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=document_library

# Neo4j Desktop connection (update with your password)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Remote Services (Dell Laptop - Update IPs as needed)
MILVUS_HOST=192.168.100.42
MILVUS_PORT=19530
EMBEDDING_SERVICE_URL=http://192.168.100.42:8002
KNOWLEDGE_GRAPH_SERVICE_URL=http://192.168.100.42:8003

# Remote LLM Service (Lenovo Laptop - Update IP as needed)
LLM_SERVICE_URL=http://192.168.100.43:8001

# Local Services (Mac Mini)
DOCLING_SERVICE_URL=http://localhost:8004

# Project Configuration
PROJECT_NAME=Distributed Document Library System
API_V1_STR=/api/v1
DATA_DIR=./data
USER_FILES_DIR=./data/user_files
EXTRACTED_DIR=./data/extracted

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/orchestrator.log
EOF
    
    print_warning "Created .env file with default values"
    print_error "IMPORTANT: Please update the following in orchestrator_api/.env:"
    echo "  - SECRET_KEY: Generate a secure random key"
    echo "  - NEO4J_PASSWORD: Your actual Neo4j Desktop password"
    echo "  - IP addresses: Update to match your network setup"
fi

cd ..

# Step 6: Set up Granite Docling Service
print_info "Step 6: Setting up Granite Docling Service..."

cd gpu_services/docling_service

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment for Granite Docling Service..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    print_info "Installing MLX and docling dependencies..."
    # Install MLX framework first (Apple Silicon specific)
    pip install mlx mlx-vlm
    
    # Install docling with MLX support
    pip install "docling[vlm,mlx]"
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_success "Granite Docling Service environment created ✓"
else
    print_info "Virtual environment already exists for Granite Docling Service"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file for Granite Docling Service..."
    cat > .env << 'EOF'
# Granite Docling Service Configuration for Mac Mini

# AssemblyAI API key for audio processing (optional)
ASSEMBLYAI_API_KEY=

# Service configuration
HOST=0.0.0.0
PORT=8004

# Model configuration
MODEL_NAME=granite-docling-258M-mlx
MODEL_TYPE=vision-language-model
PLATFORM=Apple Silicon (MLX)

# Processing configuration
MAX_TOKENS=4096
TEMPERATURE=0.0

# Chunking configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MERGE_PEERS=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
EOF
    
    print_success "Created .env file for Granite Docling Service ✓"
fi

cd ../..

# Step 7: Create startup scripts
print_info "Step 7: Creating startup scripts..."

# Create database startup script
cat > start_databases.sh << 'EOF'
#!/bin/bash
# Start databases for Mac Mini

echo "Starting databases on Mac Mini..."

# Start MongoDB
echo "Starting MongoDB..."
brew services start mongodb/brew/mongodb-community

# Check Neo4j Desktop
if lsof -i :7687 > /dev/null 2>&1; then
    echo "✅ Neo4j Desktop is running"
else
    echo "⚠️  Neo4j Desktop not detected - please start it manually"
fi

echo "Databases startup complete!"
EOF

chmod +x start_databases.sh

# Create services startup script
cat > start_services.sh << 'EOF'
#!/bin/bash
# Start services for Mac Mini

echo "Starting services on Mac Mini..."

# Start Granite Docling Service
echo "Starting Granite Docling Service..."
cd gpu_services/docling_service
source venv/bin/activate
python main.py > ../../logs/docling.log 2>&1 &
DOCLING_PID=$!
echo $DOCLING_PID > ../../.docling.pid
echo "Granite Docling Service started (PID: $DOCLING_PID)"
cd ../..

# Wait a moment for docling to start
sleep 10

# Start Orchestrator API
echo "Starting Orchestrator API..."
cd orchestrator_api
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo $ORCHESTRATOR_PID > ../.orchestrator.pid
echo "Orchestrator API started (PID: $ORCHESTRATOR_PID)"
cd ..

echo ""
echo "Mac Mini services started successfully!"
echo ""
echo "Services running:"
echo "  ✅ MongoDB: mongodb://localhost:27017"
echo "  ✅ Neo4j Desktop: bolt://localhost:7687"
echo "  ✅ Granite Docling Service: http://localhost:8004"
echo "  ✅ Orchestrator API: http://localhost:8000"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo "Service Health Checks:"
echo "  curl http://localhost:8004/health"
echo "  curl http://localhost:8000/health"
EOF

chmod +x start_services.sh

# Create stop services script
cat > stop_services.sh << 'EOF'
#!/bin/bash
# Stop services for Mac Mini

echo "Stopping services on Mac Mini..."

# Stop Orchestrator API
if [ -f ".orchestrator.pid" ]; then
    ORCHESTRATOR_PID=$(cat .orchestrator.pid)
    if kill -0 $ORCHESTRATOR_PID 2>/dev/null; then
        kill $ORCHESTRATOR_PID
        echo "Orchestrator API stopped (PID: $ORCHESTRATOR_PID)"
    fi
    rm .orchestrator.pid
fi

# Stop Granite Docling Service
if [ -f ".docling.pid" ]; then
    DOCLING_PID=$(cat .docling.pid)
    if kill -0 $DOCLING_PID 2>/dev/null; then
        kill $DOCLING_PID
        echo "Granite Docling Service stopped (PID: $DOCLING_PID)"
    fi
    rm .docling.pid
fi

# Stop MongoDB (optional)
# brew services stop mongodb/brew/mongodb-community

echo "Mac Mini services stopped"
EOF

chmod +x stop_services.sh

# Step 8: Create health check script
print_info "Step 8: Creating health check script..."

cat > health_check.sh << 'EOF'
#!/bin/bash
# Health check for Mac Mini services

echo "=== Mac Mini Services Health Check ==="

# Check MongoDB
if lsof -i :27017 > /dev/null 2>&1; then
    echo "✅ MongoDB is running on port 27017"
else
    echo "❌ MongoDB is not running"
fi

# Check Neo4j Desktop
if lsof -i :7687 > /dev/null 2>&1; then
    echo "✅ Neo4j Desktop is running on port 7687"
else
    echo "❌ Neo4j Desktop is not running"
fi

# Check Granite Docling Service
if curl -s http://localhost:8004/health > /dev/null 2>&1; then
    echo "✅ Granite Docling Service is running on port 8004"
else
    echo "❌ Granite Docling Service is not running"
fi

# Check Orchestrator API
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Orchestrator API is running on port 8000"
else
    echo "❌ Orchestrator API is not running"
fi

echo ""
echo "Service URLs:"
echo "  - Orchestrator API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Granite Docling: http://localhost:8004"
echo "  - MongoDB: mongodb://localhost:27017"
echo "  - Neo4j Desktop: bolt://localhost:7687"
EOF

chmod +x health_check.sh

echo ""
print_success "=== Mac Mini Setup Complete! ==="
echo ""
echo "📁 Project structure created"
echo "🍃 MongoDB installed and configured"
echo "🔷 Neo4j Desktop integration ready"
echo "🤖 Granite Docling Service (granite-docling-258M-mlx) ready"
echo "🎯 Orchestrator API configured"
echo ""
print_warning "IMPORTANT NEXT STEPS:"
echo ""
echo "1. Update configuration files:"
echo "   - orchestrator_api/.env: Set NEO4J_PASSWORD and network IPs"
echo "   - gpu_services/docling_service/.env: Set ASSEMBLYAI_API_KEY if needed"
echo ""
echo "2. Ensure Neo4j Desktop is running:"
echo "   - Start Neo4j Desktop application"
echo "   - Create/start a database"
echo "   - Note the password for .env configuration"
echo ""
echo "3. Start the services:"
echo "   ./start_databases.sh"
echo "   ./start_services.sh"
echo ""
echo "4. Verify everything is working:"
echo "   ./health_check.sh"
echo ""
print_info "The Mac Mini is now ready to serve as the main hub for your distributed document library!"