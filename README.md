# 🤖 Stack Overflow AI Assistant

A full-stack RAG-based AI assistant grounded in 500K real Stack Overflow questions and 1.2M answers. Ask any programming question and get a synthesized answer with real Stack Overflow sources.

🔗 **Live Demo:** [stack-overflow-ai-assistant.vercel.app](https://stack-overflow-ai-assistant.vercel.app)

---

## 📸 Screenshots

> <img width="1877" height="907" alt="image" src="https://github.com/user-attachments/assets/f281f427-2c91-4de5-ad04-03673b4acd6f" />


---

## 🧠 How It Works

```
User Query
    ↓
Guardrail Check (tech or not?)
    ↓
Embed query → Search FAISS (498K vectors)
    ↓
Fetch real SO answers by question_id
    ↓
Build context → Groq LLaMA 3.1
    ↓
Synthesized answer + Source cards
```

### Five Response Paths

| Path | Trigger | Response |
|---|---|---|
| `greeting` | "hi", "hello" | Friendly intro message |
| `rag` | Tech + found in dataset | Answer + 3 source cards |
| `general` | Tech + not in dataset | LLM answer from own knowledge |
| `no_answers` | Found questions but no answers | LLM general answer |
| `rejected` | Non-tech query | Polite rejection message |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (Vercel)                  │
│         React + Tailwind CSS + Marked.js        │
└─────────────────────┬───────────────────────────┘
                      │ POST /chat
┌─────────────────────▼───────────────────────────┐
│           Backend (HuggingFace Spaces)          │
│                  FastAPI                        │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │            RAG Pipeline                 │   │
│  │  Guardrail → FAISS → AnswerFetcher      │   │
│  │  ContextBuilder → Groq LLaMA 3.1        │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Data (HuggingFace Datasets)           │
│  questions.index  │  questions_docs.pkl         │
│  answers_cleaned.parquet                        │
└─────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| File | Size | Description |
|---|---|---|
| `questions_cleaned.parquet` | ~320 MB | 498,643 cleaned SO questions |
| `answers_cleaned.parquet` | ~600 MB | 1,199,928 cleaned SO answers |
| `top_tags.parquet` | ~1 KB | Top 30 SO tags with counts |
| `questions.index` | ~730 MB | FAISS IVFFlat index |
| `questions_docs.pkl` | ~400 MB | LangChain Documents with metadata |

**Data Source:** Google BigQuery public Stack Overflow dataset  
**Stratified sampling:** Top 30 programming tags, 500K questions total

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| Top-1 Accuracy | 65.1% |
| Top-3 Accuracy | 84.7% |
| Top-5 Accuracy | 90.0% |
| Score Ranking Accuracy | 38.3% |
| **Overall Accuracy** | **79.9%** |

---

## 🛠️ Tech Stack

**ML / Data**
- `sentence-transformers` — all-MiniLM-L6-v2 embeddings (384 dims)
- `FAISS` — IVFFlat index for semantic search
- `LangChain` — Document ingestion and pipeline
- `Groq` — LLaMA 3.1 8B for answer generation
- `Google BigQuery` — Data extraction

**Backend**
- `FastAPI` — REST API
- `HuggingFace Spaces` — Deployment (Docker)
- `HuggingFace Hub` — Data storage

**Frontend**
- `HTML/CSS/JavaScript` — Chat interface
- `Tailwind CSS` — Styling
- `Marked.js` — Markdown rendering
- `Vercel` — Deployment

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11+
- GROQ API key (free at [console.groq.com](https://console.groq.com))
- HuggingFace account

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/retronoob99/stack-overflow-ai-assistant
cd stack-overflow-ai-assistant/backend

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "VECTOR_STORE_PATH=../data/vector_store" >> .env
echo "ANSWERS_PATH=../data/answers_cleaned.parquet" >> .env

# Run the backend
python app.py
```

### Frontend Setup

```bash
# Open index.html directly in browser
# Or use Live Server in VS Code

# Update API URL in index.html to localhost
# Find: fetch("https://...hf.space/chat"
# Replace with: fetch("http://localhost:7860/chat"
```

### Download Data

```python
from huggingface_hub import hf_hub_download

# Download FAISS index
hf_hub_download(
    repo_id="retronoob99/stackoverflow-ai-data",
    filename="questions.index",
    repo_type="dataset",
    local_dir="data/vector_store"
)

# Download documents
hf_hub_download(
    repo_id="retronoob99/stackoverflow-ai-data",
    filename="questions_docs.pkl",
    repo_type="dataset",
    local_dir="data/vector_store"
)

# Download answers
hf_hub_download(
    repo_id="retronoob99/stackoverflow-ai-data",
    filename="answers_cleaned.parquet",
    repo_type="dataset",
    local_dir="data"
)
```

---

## 📁 Project Structure

```
stack-overflow-ai-assistant/
│
├── backend/
│   ├── app.py                  ← FastAPI app + endpoints
│   ├── rag_pipeline.py         ← RAG logic (AnswerFetcher, ContextBuilder, LLMCaller, Guardrail)
│   ├── utils.py                ← EmbeddingManager, VectorStore, RAGRetriever
│   ├── Dockerfile              ← HuggingFace Spaces deployment
│   └── requirements.txt
│
├── frontend/
│   └── index.html              ← Full chat UI (HTML + CSS + JS)
│
├── notebook/
│   ├── vector_indexing_pipeline.ipynb   ← One-time embedding + FAISS setup
│   └── Data_Cleaning_EDA.ipynb          ← Data cleaning pipeline
│
└── README.md
```

---

## 🔌 API Endpoints

### `POST /chat`

```json
// Request
{
    "query": "How do I reverse a list in Python?",
    "top_k": 5
}

// Response
{
    "answer": "You can reverse a list using [::-1]...",
    "is_relevant": true,
    "is_tech": true,
    "sources": [
        {
            "rank": 1,
            "question_id": 3940128,
            "title": "How do I reverse a list or loop over it backwards?",
            "primary_tag": "python",
            "top_answers": ["To get a new reversed list..."]
        }
    ],
    "path": "rag"
}
```

### `GET /health`

```json
{
    "status": "ok",
    "pipeline_ready": true,
    "total_vectors": 498643,
    "total_answers": 1199928,
    "model": "all-MiniLM-L6-v2",
    "score_threshold": 0.5
}
```

---

## 📖 Related Research

- Lewis et al. (2020) — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Johnson et al. (2017) — [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
- Reimers & Gurevych (2019) — [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- Greco (2018) — [A Behavior-Driven Recommendation System for Stack Overflow Posts](https://scholarscompass.vcu.edu/etd/5396)

---

## 🙏 Acknowledgements

- [Stack Overflow](https://stackoverflow.com) for the public dataset via Google BigQuery
- [HuggingFace](https://huggingface.co) for free hosting
- [Groq](https://groq.com) for blazing fast LLM inference
- [Vercel](https://vercel.com) for frontend deployment

---

## 📄 License

MIT License — feel free to use, modify and build on this project.




