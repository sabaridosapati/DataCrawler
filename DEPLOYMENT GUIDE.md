# DISTRIBUTED DOCUMENT LIBRARY DEPLOYMENT GUIDE

This guide will help you deploy the distributed document library system across three machines:
- **Mac Mini (16GB RAM)**: Main backend, MongoDB, Neo4j, Docling service
- **Lenovo Laptop (RTX 4070, 32GB RAM)**: LLM service with Gemma 3-4B
- **Dell Laptop (RTX 4050, 16GB RAM)**: Embedding service, Vector database (Milvus)

## Prerequisites

### All Machines
- Python 3.10+
- Git
- Docker and Docker Compose (for container deployment)
- Network connectivity between all machines

### GPU Machines (Lenovo & Dell)
- NVIDIA drivers installed
- CUDA 12.1+ toolkit
- nvidia-docker runtime (for container deployment)

## Network Configuration

Update the IP addresses in the configuration files to match your network:
- **Mac Mini**: `192.168.100.41`
- **Dell Laptop**: `192.168.100.42`
- **Lenovo Laptop**: `192.168.100.43`

## Deployment Steps

### 1. Mac Mini Setup (Main Backend)

```bash
# Clone the repository
git clone <your-repo-url>
cd document-library

# Copy and configure environment
cp .env.mac-mini.template orchestrator_api/.env
# Edit orchestrator_api/.env with your values

# Copy Docling service environment
cp .env.mac-mini.template gpu_services/docling_service/.env
# Edit as needed

# Option A: Docker Deployment
docker-compose -f docker-compose.mac.yml up -d

# Option B: Local Deployment
# Install dependencies for orchestrator
cd orchestrator_api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install dependencies for docling service
cd ../gpu_services/docling_service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start services
# Terminal 1: MongoDB
mongod --dbpath ./data/mongodb

# Terminal 2: Neo4j
neo4j start

# Terminal 3: Docling Service
cd gpu_services/docling_service
source venv/bin/activate
python main.py

# Terminal 4: Orchestrator API
cd orchestrator_api
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Lenovo Laptop Setup (LLM Service)

```bash
# Clone the repository
git clone <your-repo-url>
cd document-library/gpu_services/llm_service

# Copy and configure environment
cp ../../.env.lenovo.template .env
# Edit .env with your values

# Option A: Docker Deployment
docker-compose -f ../../docker-compose.lenovo.yml up -d

# Option B: Local Deployment
chmod +x setup_local.sh start_local.sh
./setup_local.sh
./start_local.sh
```

### 3. Dell Laptop Setup (Embedding & Vector Services)

```bash
# Clone the repository
git clone <your-repo-url>
cd document-library

# Copy and configure environments
cp .env.dell.template gpu_services/embedding_service/.env
cp .env.dell.template gpu_services/knowledge_graph_service/.env
# Edit both .env files as needed

# Option A: Docker Deployment
docker-compose -f docker-compose.dell.yml up -d

# Option B: Local Deployment
# Terminal 1: Start Milvus dependencies
docker run -d --name etcd -p 2379:2379 quay.io/coreos/etcd:v3.5.5
docker run -d --name minio -p 9000:9000 -p 9001:9001 minio/minio:latest
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.5.18

# Terminal 2: Embedding Service
cd gpu_services/embedding_service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Terminal 3: Knowledge Graph Service
cd gpu_services/knowledge_graph_service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Service Health Checks

After deployment, verify all services are running:

### Mac Mini (192.168.100.41)
- Orchestrator API: http://192.168.100.41:8000/health
- Docling Service: http://192.168.100.41:8004/health
- MongoDB: mongodb://192.168.100.41:27017
- Neo4j Browser: http://192.168.100.41:7474

### Lenovo Laptop (192.168.100.43)
- LLM Service: http://192.168.100.43:8001/health
- API Documentation: http://192.168.100.43:8001/docs

### Dell Laptop (192.168.100.42)
- Embedding Service: http://192.168.100.42:8002/health
- Knowledge Graph Service: http://192.168.100.42:8003/health
- Milvus: http://192.168.100.42:19530 (use SDK to test)

## Testing the System

1. **Register a user**:
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username": "test@example.com", "password": "password123"}'
```

2. **Login to get token**:
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=password123"
```

3. **Upload a document**:
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_document.pdf"
```

4. **Query the system**:
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/query/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the key points in my documents?"}'
```

## Troubleshooting

### Common Issues

1. **GPU out of memory**:
   - Reduce `GPU_MEMORY_FRACTION` in environment files
   - Restart the GPU services

2. **Network connectivity issues**:
   - Check firewall settings on all machines
   - Verify IP addresses in configuration files
   - Test connectivity with `ping` and `telnet`

3. **Model download issues**:
   - Ensure internet connectivity
   - Check Hugging Face token if using private models
   - Monitor disk space for model downloads

4. **Database connection issues**:
   - Verify database services are running
   - Check credentials in environment files
   - Ensure ports are not blocked by firewall

### Logs

Check service logs for debugging:

#### Docker Deployment
```bash
# Mac Mini
docker-compose -f docker-compose.mac.yml logs -f

# Lenovo
docker-compose -f docker-compose.lenovo.yml logs -f

# Dell
docker-compose -f docker-compose.dell.yml logs -f
```

#### Local Deployment
- Check individual service log files
- Use `journalctl` for system service logs
- Monitor GPU usage with `nvidia-smi`

## Production Considerations

1. **Security**:
   - Change default passwords
   - Use proper SSL certificates
   - Configure firewall rules
   - Use environment variables for secrets

2. **Monitoring**:
   - Set up log aggregation
   - Monitor GPU and CPU usage
   - Set up health check alerts

3. **Backup**:
   - Regular database backups (MongoDB, Neo4j)
   - Model and configuration backups
   - Document storage backups

4. **Scaling**:
   - Consider load balancers for high traffic
   - Multiple GPU instances for LLM service
   - Database clustering for high availability

## Support

For issues and questions:
1. Check the logs first
2. Verify network connectivity
3. Ensure all dependencies are installed
4. Check GPU memory and availability