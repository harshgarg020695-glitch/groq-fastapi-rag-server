# Groq FastAPI RAG Server

## What it does

Enterprise document Q&A systems typically require expensive cloud infrastructure and per-query API budgets that scale poorly. This project solves that by exposing a production-ready REST API that answers natural language questions against a private knowledge base — using Groq's LPU inference for sub-second response times and FAISS for in-memory vector search. Any front-end, mobile app, or internal tool can integrate via the `/chat` endpoint without touching a vector database subscription.

## Architecture

| Component | Choice | Reason |
|---|---|---|
| LLM inference | Groq (`llama-3.1-8b-instant`) | ~10x faster than OpenAI on equivalent queries; generous free tier |
| Vector search | FAISS (in-memory) | Zero infrastructure cost; suitable for single-document or small KB use cases |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) | No embedding API cost; runs on CPU in under 100ms |
| Framework | FastAPI + Uvicorn | Auto-generates Swagger UI at `/docs`; async-native for LLM I/O |
| RAG chain | LangChain LCEL pipe | Composable, readable; supports easy swap of retriever or LLM |

The server loads `data.txt` at startup, chunks it with `RecursiveCharacterTextSplitter`, builds a FAISS index, and keeps the RAG chain in memory for the lifetime of the process.

## How to run

**Prerequisites:** Python 3.10+, a [Groq API key](https://console.groq.com/keys)

```bash
# 1. Clone and enter repo
git clone https://github.com/harshgarg020695-glitch/groq-fastapi-rag-server
cd groq-fastapi-rag-server

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 5. Start the server
python main.py
# or: uvicorn main:app --reload
```

**Endpoints:**
- `GET /` — Chat UI (browser)
- `POST /chat` — JSON API: `{"question": "your question here"}`
- `GET /docs` — Swagger UI (interactive API explorer)

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | From [console.groq.com](https://console.groq.com/keys) |

## Live demo

Live API: https://groq-fastapi-rag-server.onrender.com/docs

> Note: Hosted on Render free tier. The first request after a period of inactivity may take 30–60 seconds while the service cold-starts. Subsequent requests respond normally.
