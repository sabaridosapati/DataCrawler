# 📋 Deployment Checklist

Use this checklist to ensure your distributed document library system is properly deployed and configured.

## Pre-Deployment Preparation

### Network Setup
- [ ] Assign static IP addresses to all machines:
  - [ ] Mac Mini: 192.168.100.41 (or update configs with your IPs)
  - [ ] Dell Laptop: 192.168.100.42 
  - [ ] Lenovo Laptop: 192.168.100.43
- [ ] Test network connectivity between all machines (`ping` test)
- [ ] Configure firewall rules to allow inter-machine communication
- [ ] Ensure all required ports are open:
  - Mac Mini: 8000, 8004, 27017, 7474, 7687
  - Lenovo: 8001
  - Dell: 8002, 8003, 19530, 9000, 9001, 2379

### Hardware Prerequisites
- [ ] **Mac Mini (16GB RAM)**:
  - [ ] macOS with Python 3.10+
  - [ ] Docker installed (optional)
  - [ ] At least 50GB free storage
  
- [ ] **Lenovo Laptop (RTX 4070, 32GB RAM)**:
  - [ ] NVIDIA drivers installed
  - [ ] CUDA 12.1+ toolkit installed
  - [ ] Python 3.10+ installed
  - [ ] At least 20GB free storage for models
  
- [ ] **Dell Laptop (RTX 4050, 16GB RAM)**:
  - [ ] NVIDIA drivers installed
  - [ ] CUDA 12.1+ toolkit installed
  - [ ] Python 3.10+ installed
  - [ ] Docker installed for Milvus dependencies
  - [ ] At least 30GB free storage

## Deployment Steps

### Step 1: Mac Mini Setup
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Copy environment template: `cp .env.mac-mini.template orchestrator_api/.env`
- [ ] Edit `orchestrator_api/.env` with your configuration:
  - [ ] Set `SECRET_KEY` to a secure random value
  - [ ] Update remote service URLs with correct IPs
  - [ ] Set `ASSEMBLYAI_API_KEY` if using audio processing
- [ ] Run setup script: `./setup_mac_mini.sh`
- [ ] Start databases: `./start_databases.sh`
- [ ] Start services: `./start_services.sh`
- [ ] Verify health: `curl http://localhost:8000/health`

### Step 2: Lenovo Laptop Setup (LLM Service)
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Navigate to LLM service: `cd gpu_services/llm_service`
- [ ] Copy environment: `cp ../../.env.lenovo.template .env`
- [ ] Edit `.env` if needed (default settings should work)
- [ ] Run setup: `./setup_local.sh`
- [ ] Start service: `./start_local.sh`
- [ ] Verify health: `curl http://localhost:8001/health`
- [ ] Test GPU usage: `nvidia-smi` (should show model loaded)

### Step 3: Dell Laptop Setup (Embedding & Vector)
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Copy environment templates:
  - [ ] `cp .env.dell.template gpu_services/embedding_service/.env`
  - [ ] `cp .env.dell.template gpu_services/knowledge_graph_service/.env`
- [ ] Edit both `.env` files:
  - [ ] Verify Neo4j URI points to Mac Mini
  - [ ] Verify LLM service URL points to Lenovo laptop
- [ ] Run setup: `./setup_dell_laptop.sh`
- [ ] Start services: `./start_services.sh`
- [ ] Verify health: 
  - [ ] `curl http://localhost:8002/health`
  - [ ] `curl http://localhost:8003/health`

## Post-Deployment Verification

### System Health Checks
- [ ] Run comprehensive test: `./test_distributed_system.sh`
- [ ] All services return HTTP 200 on health endpoints
- [ ] GPU services show model loading in logs
- [ ] Database connections established successfully

### Functional Testing
- [ ] User registration works: Test signup endpoint
- [ ] Authentication works: Test login and get JWT token
- [ ] Document upload works: Upload a test PDF/DOCX
- [ ] Document processing completes: Check status endpoint
- [ ] Query system responds: Test basic query functionality

### Performance Verification
- [ ] **LLM Service**: Response time < 30 seconds for simple queries
- [ ] **Embedding Service**: Batch processing works without OOM errors
- [ ] **Docling Service**: Document extraction completes successfully
- [ ] **Vector Search**: Query response time < 2 seconds
- [ ] **Knowledge Graph**: Entity extraction produces reasonable results

### Resource Monitoring
- [ ] Check GPU memory usage: `nvidia-smi` on both GPU machines
- [ ] Monitor CPU usage during processing
- [ ] Verify sufficient disk space on all machines
- [ ] Check network bandwidth during large file transfers

## Production Readiness

### Security
- [ ] Change default passwords:
  - [ ] MongoDB: `mongo_data` volume or connection string
  - [ ] Neo4j: Update in orchestrator config
- [ ] Generate secure JWT secret key (min 32 characters)
- [ ] Configure HTTPS/SSL certificates for production
- [ ] Set up proper firewall rules
- [ ] Enable service monitoring and alerting

### Backup Strategy
- [ ] MongoDB backup script configured
- [ ] Neo4j backup script configured  
- [ ] Document file storage backup configured
- [ ] Configuration files backed up
- [ ] Model cache backup strategy (optional)

### Monitoring
- [ ] Log aggregation configured
- [ ] Health check monitoring set up
- [ ] GPU usage monitoring configured
- [ ] Disk space monitoring enabled
- [ ] Network connectivity monitoring enabled

## Common Issues Checklist

If you encounter problems, check these items:

### Service Won't Start
- [ ] Check Python version (must be 3.10+)
- [ ] Verify virtual environment activated
- [ ] Check all dependencies installed (`pip install -r requirements.txt`)
- [ ] Verify GPU drivers and CUDA installation
- [ ] Check port availability (`netstat -tulpn | grep PORT`)

### GPU Out of Memory
- [ ] Reduce `GPU_MEMORY_FRACTION` in service configs
- [ ] Close other GPU applications
- [ ] Restart GPU services to clear memory
- [ ] Consider reducing batch sizes in processing

### Network Connectivity Issues
- [ ] Verify IP addresses in all config files
- [ ] Test with `ping` and `telnet` commands
- [ ] Check firewall settings on all machines
- [ ] Verify DNS resolution if using hostnames

### Database Connection Issues
- [ ] Check database services are running
- [ ] Verify connection strings in config files
- [ ] Test database connectivity directly
- [ ] Check user permissions and credentials

### Model Loading Issues
- [ ] Verify internet connectivity for model downloads
- [ ] Check Hugging Face token if using private models
- [ ] Ensure sufficient disk space for model cache
- [ ] Check model compatibility with hardware

## Support Resources

- **Documentation**: README.md for detailed setup instructions
- **Testing**: Run `./test_distributed_system.sh` for automated diagnostics
- **Logs**: Check service logs in `logs/` directories
- **Community**: Create issues on the project repository

---

✅ **System deployment complete when all items are checked!**

**Next Steps**: Start uploading documents and building your private AI-powered document library!