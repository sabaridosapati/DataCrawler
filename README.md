# DataCrawler

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload documents, build a knowledge graph, and chat with your data using hybrid semantic search and Google Gemini AI.

---

## Features

- **Multi-format document ingestion** — PDF, DOCX, PPTX, HTML, Markdown, TXT, images
- **Hybrid RAG retrieval** — HyDE query expansion + BM25 full-text + vector search + Reciprocal Rank Fusion (RRF)
- **Knowledge graph extraction** — Entities and relationships automatically extracted and stored in Neo4j
- **User-isolated data** — Each user's documents, embeddings, and chat history are fully separated
- **JWT authentication** — Secure signup/login with bcrypt password hashing
- **Real-time processing status** — Documents tracked through QUEUED → PROCESSING → COMPLETED/FAILED
- **Streamlit UI** — Chat, document management, semantic search, and system dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│                   localhost:<portno>                     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│              Orchestrator API (FastAPI)              │
│                   localhost:<portno>                     │
│  Auth · Documents · Query · Hybrid Retriever         │
└────┬──────────┬──────────┬──────────────────────────┘
     │          │          │
     ▼          ▼          ▼
  Docling   Embedding    LLM         Knowledge Graph
  Service    Service    Service          Service
  :<portno>      :<portno>       :<portno>            :<portno>
     │          │          │                │
     │          │    Google Gemini API       │
     │          │   (cloud, no local GPU)    │
     ▼          ▼                            ▼
  MongoDB    Milvus                        Neo4j
  (docs,     (vectors,                   (entities,
  users,     3072-dim                  relationships)
  chats)     COSINE)
```

### Services

| Service | Port | Purpose |
|---|---|---|
| Streamlit Frontend | <portno> | Chat UI, document upload, search, dashboard |
| Orchestrator API | <portno> | Main FastAPI backend — auth, documents, RAG queries |
| LLM Service | <portno> | Text generation via Gemini `gemini-2.0-flash-lite` |
| Embedding Service | <portno> | Vector embeddings via Gemini `gemini-embedding-001` (3072-dim) |
| Knowledge Graph Service | <portno> | Entity/relation extraction → Neo4j |
| Docling Service | <portno> | Document parsing and chunking |

---

## Technology Stack

- **Backend:** FastAPI, Python 3.10+, Motor (async MongoDB), Neo4j async driver
- **Frontend:** Streamlit
- **AI:** Google Gemini API (LLM + Embeddings — no local GPU required)
- **Vector DB:** Milvus (Docker) — COSINE similarity, HNSW index, 3072 dimensions
- **Graph DB:** Neo4j 5.15+
- **Document Store:** MongoDB
- **Auth:** JWT + bcrypt
- **Document Processing:** Docling 2.x

---

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for Milvus, MongoDB)
- Neo4j instance (local or cloud)
- Google Gemini API key — [get one here](https://aistudio.google.com/app/apikey)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sabaridosapati/DataCrawler.git
cd DataCrawler
```

### 2. Create a virtual environment

```bash
python -m venv datacrawlerenv
# Windows
datacrawlerenv\Scripts\activate
# macOS/Linux
source datacrawlerenv/bin/activate
```

### 3. Install dependencies

```bash
# Orchestrator API
pip install -r orchestrator_api/requirements.txt

# GPU Services
pip install -r gpu_services/llm_service/requirements.txt
pip install -r gpu_services/embedding_service/requirements.txt
pip install -r gpu_services/knowledge_graph_service/requirements.txt
pip install -r gpu_services/docling_service/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

### 4. Configure environment variables

Copy the template and fill in your values:

```bash
cp .env.template orchestrator_api/.env
```

Key variables to set:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=change-me-to-a-random-string

# Databases
MONGODB_URL=mongodb://localhost:<portno>
NEO4J_URI=bolt://localhost:<portno>
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Service URLs (defaults for single-machine)
LLM_SERVICE_URL=http://localhost:<portno>
EMBEDDING_SERVICE_URL=http://localhost:<portno>
KNOWLEDGE_GRAPH_SERVICE_URL=http://localhost:<portno>
DOCLING_SERVICE_URL=http://localhost:<portno>
```

Also set `GEMINI_API_KEY` in each GPU service `.env` file:
- `gpu_services/llm_service/.env`
- `gpu_services/embedding_service/.env`
- `gpu_services/knowledge_graph_service/.env`

### 5. Start the databases

```bash
# Milvus + MongoDB via Docker Compose
docker-compose -f docker-compose.databases.yml up -d
```

### 6. Start all services

```bash
python start.py
```

This opens separate terminal windows for each service with `--reload` enabled. The frontend will be available at `http://localhost:<portno>`.

To stop all services:

```bash
python stop.py
```

---

## Usage

### Web Interface

Open `http://localhost:<portno>` in your browser.

1. **Register / Login** — Create an account or sign in
2. **Upload Documents** — Go to the Documents tab, upload PDF/DOCX/etc.
3. **Wait for processing** — Status updates from QUEUED → COMPLETED (usually seconds to minutes)
4. **Chat** — Ask questions about your documents in the Chat tab
5. **Search** — Use the Search tab for direct semantic search without LLM generation
6. **Dashboard** — Check service health and stats

### API

Base URL: `http://localhost:<portno>/api/v1`

Interactive docs: `http://localhost:<portno>/docs`

```bash
# Register
curl -X POST http://localhost:<portno>/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "yourpassword"}'

# Login
curl -X POST http://localhost:<portno>/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=yourpassword"

# Upload a document
curl -X POST http://localhost:<portno>/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"

# Query your documents
curl -X POST http://localhost:<portno>/api/v1/query/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize the key findings", "use_hyde": true, "use_bm25": true, "top_k": 5}'
```

---

## Document Processing Pipeline

```
Upload → Docling (extract + chunk) → Knowledge Graph (Neo4j) → Embeddings (Gemini) → Milvus
```

1. **Docling Service** parses the file and returns markdown + JSON chunks
2. **Knowledge Graph Service** extracts entities and relationships, stores them in Neo4j
3. **Embedding Service** generates 3072-dim vectors via Gemini `gemini-embedding-001`
4. Vectors are indexed in **Milvus** under the user's collection

## Query Pipeline

```
User prompt → HyDE expansion → Vector search (Milvus) + BM25 → RRF fusion → LLM (Gemini)
```

1. **HyDE**: Generates a hypothetical answer to expand the query
2. **Vector search**: Milvus COSINE similarity search (min threshold: 0.6)
3. **BM25**: Full-text search over all user chunks
4. **RRF**: Combines both result lists using Reciprocal Rank Fusion
5. **LLM**: Gemini generates the final answer with retrieved context

If no relevant documents are found (all scores below threshold), the LLM answers from general knowledge and says so.

---

## Project Structure

```
Data-Crawler/
├── orchestrator_api/
│   └── app/
│       ├── api/           # auth.py, documents.py, query.py
│       ├── core/          # config.py, security.py
│       ├── db/            # milvus_handler.py, mongo_handler.py, neo4j_handler.py
│       ├── models/        # Pydantic models
│       └── services/      # hybrid_retriever.py, processing_pipeline.py, gpu_node_client.py
├── gpu_services/
│   ├── llm_service/       # Gemini chat completions (OpenAI-compatible)
│   ├── embedding_service/ # Gemini embeddings
│   ├── knowledge_graph_service/ # Entity extraction → Neo4j
│   └── docling_service/   # Document parsing
├── frontend/
│   └── app.py             # Streamlit UI
├── data/                  # Runtime: user files, extracted chunks
├── docker-compose.databases.yml
├── start.py
├── stop.py
└── .env.template
```

---

## Troubleshooting

**Services not starting**
- Make sure Docker is running before starting databases
- Check that all `.env` files have `GEMINI_API_KEY` set
- Verify Neo4j is running and the password matches your config

**Document stuck in PROCESSING**
- Check the orchestrator API logs for errors
- Verify each GPU service is running (`http://localhost:800X/health`)
- Gemini API quota errors will cause processing to fail — check your API usage

**No results for relevant queries**
- The minimum similarity threshold is 0.6 — very short or vague queries may not match
- Make sure the document status is COMPLETED before querying
- Try disabling HyDE (`use_hyde: false`) to test with the raw query

**Relevance scores look wrong**
- Scores are Milvus COSINE similarity values (higher = more similar)
- Scores below 0.6 are filtered out
- The UI shows these as percentages (e.g. 0.72 → 72%)
