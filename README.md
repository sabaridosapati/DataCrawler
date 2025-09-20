# Distributed Document Library System

A high-performance, scalable document intelligence system distributed across multiple machines with local AI models for complete data privacy and confidentiality.

## 🏗️ System Architecture

### Distributed Deployment
- **Mac Mini (16GB RAM)**: Main backend, orchestrator API, document processing, MongoDB, Neo4j
- **Lenovo Laptop (RTX 4070, 32GB RAM)**: LLM service with Gemma 3-4B model  
- **Dell Laptop (RTX 4050, 16GB RAM)**: Embedding service, knowledge graph service, Milvus vector database

### Key Features
- **100% Local AI**: All processing happens on your machines - no data leaves your network
- **Multi-modal Processing**: Handles PDF, DOCX, PPTX, HTML, CSV, Excel, images, and audio
- **Advanced Knowledge Graphs**: Automatic entity extraction and relationship mapping using Neo4j
- **Semantic Search**: High-performance vector search with HNSW indexing in Milvus
- **JWT Authentication**: Secure user management with proper session handling
- **Real-time Processing**: Background document processing with status tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ on all machines
- NVIDIA drivers and CUDA 12.1+ on GPU machines
- Docker and Docker Compose (optional)
- Network connectivity between all machines

### 1. Clone and Setup

```bash
# On each machine
git clone <your-repo-url>
cd document-library
```

### 2. Configure Network

Update IP addresses in configuration files to match your network:
- Mac Mini: `192.168.100.41`
- Dell Laptop: `192.168.100.42` 
- Lenovo Laptop: `192.168.100.43`

### 3. Deploy by Machine

#### Mac Mini (Main Backend)
```bash
# Copy and edit environment
cp .env.mac-mini.template orchestrator_api/.env
# Edit orchestrator_api/.env with your settings

# Option A: Automated local setup
chmod +x setup_mac_mini.sh
./setup_mac_mini.sh
./start_databases.sh
./start_services.sh

# Option B: Docker deployment
docker-compose -f docker-compose.mac.yml up -d
```

#### Lenovo Laptop (LLM Service)
```bash
cd gpu_services/llm_service
cp ../../.env.lenovo.template .env
# Edit .env as needed

# Option A: Local setup
chmod +x setup_local.sh start_local.sh
./setup_local.sh
./start_local.sh

# Option B: Docker deployment
docker-compose -f ../../docker-compose.lenovo.yml up -d
```

#### Dell Laptop (Embedding & Vector Services)
```bash
cp .env.dell.template gpu_services/embedding_service/.env
cp .env.dell.template gpu_services/knowledge_graph_service/.env
# Edit both .env files

# Option A: Local setup
chmod +x setup_dell_laptop.sh
./setup_dell_laptop.sh
./start_services.sh

# Option B: Docker deployment  
docker-compose -f docker-compose.dell.yml up -d
```

### 4. Test the System

```bash
chmod +x test_distributed_system.sh
./test_distributed_system.sh
```

## 📊 Service Endpoints

### Mac Mini (192.168.100.41)
- **Main API**: http://192.168.100.41:8000
- **API Documentation**: http://192.168.100.41:8000/docs
- **Docling Service**: http://192.168.100.41:8004
- **MongoDB**: mongodb://192.168.100.41:27017
- **Neo4j Browser**: http://192.168.100.41:7474

### Lenovo Laptop (192.168.100.43)
- **LLM Service**: http://192.168.100.43:8001
- **Health Check**: http://192.168.100.43:8001/health
- **API Docs**: http://192.168.100.43:8001/docs

### Dell Laptop (192.168.100.42)
- **Embedding Service**: http://192.168.100.42:8002
- **Knowledge Graph Service**: http://192.168.100.42:8003
- **Milvus**: localhost:19530 (internal)

## 🔧 API Usage Examples

### 1. User Registration
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "securepass123"}'
```

### 2. Login
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepass123"
```

### 3. Upload Document
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### 4. Check Processing Status
```bash
curl "http://192.168.100.41:8000/api/v1/documents/DOC_ID/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Query Documents
```bash
curl -X POST "http://192.168.100.41:8000/api/v1/query/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the main topics in my documents?"}'
```

## 🔍 Processing Pipeline

1. **Document Upload**: User uploads file through orchestrator API
2. **Content Extraction**: Docling service extracts text, images, tables using Granite-Docling MLX
3. **Chunking**: Content is intelligently chunked with context preservation
4. **Entity Extraction**: Gemma 3-4B model extracts entities and relationships
5. **Knowledge Graph**: Entities stored in Neo4j with relationships
6. **Embeddings**: Text chunks vectorized using EmbeddingGemma-300M
7. **Vector Storage**: Embeddings stored in Milvus with user isolation
8. **Query Processing**: Semantic search + graph RAG for comprehensive answers

## 🛠️ Technology Stack

### AI Models (All Local)
- **LLM**: Google Gemma 3-4B Instruct (vision-capable)
- **Embeddings**: Google EmbeddingGemma-300M
- **Document Processing**: IBM Granite-Docling-258M MLX

### Databases
- **Vector Database**: Milvus 2.5 with HNSW indexing
- **Graph Database**: Neo4j 5.15 Community
- **Document Store**: MongoDB with Motor async driver

### Backend
- **Framework**: FastAPI with async/await
- **Authentication**: JWT with bcrypt password hashing
- **Processing**: Background task queues
- **API**: OpenAPI/Swagger documentation

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **GPU Acceleration**: NVIDIA CUDA, MLX for Apple Silicon
- **Networking**: HTTP REST APIs with proper error handling

## 📁 Project Structure

```
document-library/
├── orchestrator_api/           # Main FastAPI backend (Mac Mini)
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── core/              # Security & config
│   │   ├── db/                # Database handlers
│   │   ├── models/            # Pydantic models
│   │   └── services/          # Business logic
│   └── requirements.txt
├── gpu_services/              # GPU-powered microservices
│   ├── llm_service/          # Gemma 3-4B service (Lenovo)
│   ├── embedding_service/    # EmbeddingGemma service (Dell)
│   ├── knowledge_graph_service/ # Graph builder (Dell)
│   └── docling_service/      # Document processor (Mac)
├── docker-compose.mac.yml     # Mac Mini deployment
├── docker-compose.lenovo.yml  # Lenovo deployment
├── docker-compose.dell.yml    # Dell deployment
└── setup scripts & configs
```

## 🔒 Security & Privacy

- **Data Isolation**: All processing happens locally on your hardware
- **User Segregation**: Each user's data is completely isolated
- **Secure Authentication**: JWT tokens with configurable expiration
- **Network Security**: Services communicate over private network
- **No External APIs**: No data sent to cloud services (except optional AssemblyAI for audio)

## 📈 Performance & Scaling

### Current Capacity
- **LLM Service**: ~2-4 requests/minute on RTX 4070
- **Embedding Service**: ~100-500 documents/minute on RTX 4050
- **Vector Search**: Sub-second queries on millions of vectors
- **Document Processing**: 1-10 documents/minute depending on size

### Optimization Tips
- Adjust `GPU_MEMORY_FRACTION` based on available VRAM
- Use SSD storage for better I/O performance
- Consider CPU-only deployment for embedding service if needed
- Scale horizontally by adding more GPU machines

## 🐛 Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   ```bash
   # Reduce memory fraction in .env files
   GPU_MEMORY_FRACTION=0.8
   
   # Or restart GPU services
   docker restart llm_service
   ```

2. **Network Connectivity**
   ```bash
   # Test connectivity
   ping 192.168.100.41
   telnet 192.168.100.43 8001
   
   # Check firewall
   sudo ufw status
   ```

3. **Service Dependencies**
   ```bash
   # Check service startup order
   # 1. Databases first (Mac Mini)
   # 2. LLM service (Lenovo) 
   # 3. Embedding/KG services (Dell)
   # 4. Orchestrator (Mac Mini)
   ```

4. **Model Download Issues**
   ```bash
   # Clear Hugging Face cache
   rm -rf ~/.cache/huggingface
   
   # Set HF_TOKEN for private models
   export HF_TOKEN=your_token_here
   ```

### Log Locations
- **Docker**: `docker-compose logs -f service_name`
- **Local**: Check `logs/` directory in each service
- **System**: Use `journalctl -f` for system services

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on all three machine types
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **IBM** for the Granite-Docling model
- **Google** for the Gemma model family
- **Milvus** for the high-performance vector database
- **Neo4j** for graph database capabilities
- **FastAPI** for the excellent async framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite: `./test_distributed_system.sh`
3. Check individual service logs
4. Verify network connectivity and GPU status

---

**Built for privacy, performance, and scalability** 🚀