#!/bin/bash
# setup_mac_mini.sh
# Automated setup script for Mac Mini (Main Backend)

set -e

echo "=== Mac Mini Document Library Setup ==="
echo "This script will set up the main backend services locally"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]; then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

print_status "Python version: $PYTHON_VERSION ✓"

# Check if MongoDB is available (for local deployment)
if command -v mongod &> /dev/null; then
    print_status "MongoDB found ✓"
else
    print_warning "MongoDB not found. Will use Docker container."
fi

# Check if Neo4j is available
if command -v neo4j &> /dev/null; then
    print_status "Neo4j found ✓"
else
    print_warning "Neo4j not found. Will use Docker container."
fi

# Create directory structure
print_status "Creating directory structure..."
mkdir -p data/user_files/{raw,extracted}
mkdir -p data/mongodb
mkdir -p data/neo4j
mkdir -p logs
mkdir -p orchestrator_api/logs
mkdir -p gpu_services/docling_service/logs

# Copy environment files if they don't exist
print_status "Setting up environment files..."

if [ ! -f "orchestrator_api/.env" ]; then
    if [ -f ".env.mac-mini.template" ]; then
        cp .env.mac-mini.template orchestrator_api/.env
        print_status "Created orchestrator_api/.env from template"
        print_warning "Please edit orchestrator_api/.env with your actual values"
    else
        print_error "Environment template not found!"
        exit 1
    fi
else
    print_status "orchestrator_api/.env already exists"
fi

if [ ! -f "gpu_services/docling_service/.env" ]; then
    cp .env.mac-mini.template gpu_services/docling_service/.env
    print_status "Created docling service .env from template"
fi

# Set up orchestrator API
print_status "Setting up Orchestrator API..."
cd orchestrator_api

if [ ! -d "venv" ]; then
    print_status "Creating virtual environment for orchestrator..."
    python3 -m venv venv
fi

source venv/bin/activate
print_status "Installing orchestrator dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ..

# Set up Docling service
print_status "Setting up Docling Service..."
cd gpu_services/docling_service

if [ ! -d "venv" ]; then
    print_status "Creating virtual environment for docling..."
    python3 -m venv venv
fi

source venv/bin/activate
print_status "Installing docling dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ../..

# Create startup scripts
print_status "Creating startup scripts..."

cat > start_databases.sh << 'EOF'
#!/bin/bash
# Start databases (MongoDB and Neo4j)

echo "Starting databases..."

# Start MongoDB
if command -v mongod &> /dev/null; then
    echo "Starting MongoDB locally..."
    mkdir -p data/mongodb
    mongod --dbpath ./data/mongodb --port 27017 &
    MONGO_PID=$!
    echo "MongoDB PID: $MONGO_PID"
else
    echo "Starting MongoDB with Docker..."
    docker run -d --name mongo_db -p 27017:27017 \
        -e MONGO_INITDB_ROOT_USERNAME=admin \
        -e MONGO_INITDB_ROOT_PASSWORD=password123 \
        -v $(pwd)/data/mongodb:/data/db \
        mongo:latest
fi

# Start Neo4j
if command -v neo4j &> /dev/null; then
    echo "Starting Neo4j locally..."
    neo4j start
else
    echo "Starting Neo4j with Docker..."
    docker run -d --name neo4j_db -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password123 \
        -v $(pwd)/data/neo4j:/data \
        neo4j:5.15-community
fi

echo "Databases started. Waiting 30 seconds for initialization..."
sleep 30

echo "Database startup complete!"
EOF

chmod +x start_databases.sh

cat > start_services.sh << 'EOF'
#!/bin/bash
# Start application services

# Function to run service in background
run_service() {
    local name=$1
    local command=$2
    local logfile=$3
    
    echo "Starting $name..."
    $command > $logfile 2>&1 &
    local pid=$!
    echo "$name PID: $pid (log: $logfile)"
    echo $pid > ".$name.pid"
}

echo "Starting Document Library Services..."

# Start Docling service
cd gpu_services/docling_service
source venv/bin/activate
run_service "docling" "python main.py" "../../logs/docling.log"
cd ../..

# Wait a bit for docling to initialize
sleep 10

# Start Orchestrator API
cd orchestrator_api
source venv/bin/activate
run_service "orchestrator" "uvicorn app.main:app --host 0.0.0.0 --port 8000" "../logs/orchestrator.log"
cd ..

echo ""
echo "All services started!"
echo "Access points:"
echo "- Main API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Docling Service: http://localhost:8004"
echo "- MongoDB: mongodb://localhost:27017"
echo "- Neo4j Browser: http://localhost:7474"
echo ""
echo "Logs:"
echo "- Orchestrator: logs/orchestrator.log"
echo "- Docling: logs/docling.log"
echo ""
echo "To stop services, run: ./stop_services.sh"
EOF

chmod +x start_services.sh

cat > stop_services.sh << 'EOF'
#!/bin/bash
# Stop all services

echo "Stopping Document Library Services..."

# Function to stop service by PID file
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

stop_service "orchestrator"
stop_service "docling"

# Stop Docker containers if they exist
docker stop mongo_db neo4j_db 2>/dev/null || true
docker rm mongo_db neo4j_db 2>/dev/null || true

# Stop local MongoDB if running
pkill mongod || true

# Stop local Neo4j if running
neo4j stop || true

echo "All services stopped."
EOF

chmod +x stop_services.sh

print_status "Setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Edit orchestrator_api/.env with your actual configuration"
print_status "2. Update IP addresses for remote services"
print_status "3. Run: ./start_databases.sh"
print_status "4. Run: ./start_services.sh"
print_status ""
print_status "For Docker deployment instead, run:"
print_status "docker-compose -f docker-compose.mac.yml up -d"