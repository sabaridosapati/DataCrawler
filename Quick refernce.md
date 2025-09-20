# ⚡ Quick Reference - Updated System

## 🎯 Key Changes Made
1. **Neo4j**: Uses your existing Neo4j Desktop (no Docker Neo4j)
2. **Milvus**: Exact v2.6.2 with official setup
3. **Enhanced Embedding**: RTX 4050 optimized + docling data handling  
4. **Better Data Pipeline**: Improved docling → embedding → vector flow

## 🚀 Quick Deployment Commands

### Mac Mini (Main Backend)
```bash
# Setup
cp .env.mac-mini.template orchestrator_api/.env
# Edit NEO4J_PASSWORD in orchestrator_api/.env
./setup_mac_mini.sh
./start_databases.sh  # Only MongoDB (Neo4j Desktop already running)
./start_services.sh

# Health Check
curl http://localhost:8000/health
```

### Dell Laptop (Vector & Embedding)
```bash
# Setup Milvus v2.6.2 FIRST
./setup_milvus_v2.6.2.sh

# Setup services
cp .env.dell.template gpu_services/embedding_service/.env
cp .env.dell.template gpu_services/knowledge_graph_service/.env
# Edit NEO4J_PASSWORD in both .env files
./setup_dell_laptop.sh
./start_services.sh

# Health Checks
curl http://localhost:8002/health  # Embedding
curl http://localhost:8003/health  # Knowledge Graph
curl http://127.0.0.1:9091/webui/  # Milvus WebUI
```

### Lenovo Laptop (LLM Service)
```bash
# Setup
cd gpu_services/llm_service
cp ../../.env.lenovo.template .env
./setup_local.sh
./start_local.sh

# Health Check
curl http://localhost:8001/health
```

## 🧪 Full System Test
```bash
# From any machine
./test_distributed_system.sh
```

## 🔧 Important Configuration Changes

### Mac Mini `.env` Updates:
```env
# Your Neo4j Desktop password
NEO4J_PASSWORD=your-actual-neo4j-desktop-password

# Network IPs (update as needed)
MILVUS_HOST=192.168.100.42
LLM_SERVICE_URL=http://192.168.100.43:8001
```

### Dell `.env` Updates:
```env
# Neo4j Desktop on Mac Mini
NEO4J_URI=bolt://192.168.100.41:7687
NEO4J_PASSWORD=your-actual-neo4j-desktop-password

# RTX 4050 optimization
GPU_MEMORY_FRACTION=0.85
BATCH_SIZE=16
```

## 📊 Service Endpoints

| Machine | Service | URL | Purpose |
|---------|---------|-----|---------|
| Mac Mini | Orchestrator | http://192.168.100.41:8000 | Main API |
| Mac Mini | Docling | http://192.168.100.41:8004 | Document processing |
| Mac Mini | Neo4j Desktop | bolt://192.168.100.41:7687 | Graph database |
| Mac Mini | MongoDB | mongodb://192.168.100.41:27017 | User data |
| Lenovo | LLM Service | http://192.168.100.43:8001 | Gemma 3-4B |
| Dell | Embedding | http://192.168.100.42:8002 | EmbeddingGemma |
| Dell | Knowledge Graph | http://192.168.100.42:8003 | Graph builder |
| Dell | Milvus v2.6.2 | http://192.168.100.42:19530 | Vector database |
| Dell | Milvus WebUI | http://192.168.100.42:9091/webui/ | Vector DB UI |

## 🔍 Enhanced Data Flow
```
Document Upload (Mac) 
    ↓
Docling Processing (Mac) - Granite-Docling MLX
    ↓
Entity Extraction (Dell → Lenovo) - Gemma 3-4B
    ↓  
Neo4j Storage (Mac) - Your Neo4j Desktop
    ↓
Enhanced Embedding (Dell) - EmbeddingGemma + Docling cleaning
    ↓
Milvus v2.6.2 Storage (Dell) - HNSW indexing
```

## ⚠️ Critical Notes
- **Neo4j Desktop**: Must be running before starting other services
- **Milvus v2.6.2**: Run `./setup_milvus_v2.6.2.sh` BEFORE other Dell services  
- **Docker Desktop**: Required on Dell for GPU support with Milvus
- **Network**: Update all IP addresses in config files to match your setup
- **GPU Memory**: Reduce batch sizes if RTX 4050 runs out of memory

## 🐛 Quick Fixes
```bash
# Neo4j connection issues
neo4j status  # Check if Desktop is running

# Milvus v2.6.2 issues  
cd milvus-v2.6.2 && docker compose ps

# GPU memory issues
nvidia-smi  # Check GPU usage
# Then reduce GPU_MEMORY_FRACTION in .env

# Service not responding
curl http://IP:PORT/health  # Test individual services
```

## 📝 Test Document Upload
```bash
# Register user
curl -X POST "http://192.168.100.41:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username": "test@example.com", "password": "testpass123"}'

# Login and upload
TOKEN=$(curl -X POST "http://192.168.100.41:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=testpass123" | jq -r .access_token)

curl -X POST "http://192.168.100.41:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test-document.pdf"
```

---
**🚀 Your enhanced distributed document library is ready to deploy!**