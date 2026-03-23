from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
import uvicorn
from huggingface_hub import hf_hub_download
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


# Initialize RAG Pipeline
pipeline: Optional[RAGPipeline] = None

executor = ThreadPoolExecutor(max_workers=1)

def download_and_initialize():
    global pipeline

    data_dir         = "/tmp/data"
    vector_store_dir = f"{data_dir}/vector_store"
    answers_path     = f"{data_dir}/answers_cleaned.parquet"

    os.makedirs(vector_store_dir, exist_ok=True)

    print("📥 Downloading data from Hugging Face...")

    hf_token = os.getenv("HF_TOKEN")     # ← add this

    hf_hub_download(
        repo_id   = "retronoob99/stackoverflow-ai-data",
        filename  = "questions.index",
        repo_type = "dataset",
        local_dir = vector_store_dir,
        token     = hf_token             # ← add token
    )
    hf_hub_download(
        repo_id   = "retronoob99/stackoverflow-ai-data",
        filename  = "questions_docs.pkl",
        repo_type = "dataset",
        local_dir = vector_store_dir,
        token     = hf_token             # ← add token
    )
    hf_hub_download(
        repo_id   = "retronoob99/stackoverflow-ai-data",
        filename  = "answers_cleaned.parquet",
        repo_type = "dataset",
        local_dir = data_dir,
        token     = hf_token             # ← add token
    )

    print("✅ Data downloaded!")

    pipeline = RAGPipeline(
        vector_store_path = vector_store_dir,
        answers_path      = answers_path,
        score_threshold   = 0.5
    )
    print("✅ RAG Pipeline ready!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Start download in background thread ───────────────────────────────
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, download_and_initialize)
    print("🚀 Server started! Pipeline initializing in background...")
    yield
    print("🛑 Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Stack Overflow AI Assistant API",
    description="API for the Stack Overflow AI Assistant using RAG pipeline",
    version="1.0.0",
    lifespan = lifespan
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development, restrict in production
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
    allow_credentials=True,
)

# Request model
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class SourceCard(BaseModel):
    rank: int
    question_id: int
    title: str
    primary_tag: str
    top_answers: List[str]

class ChatResponse(BaseModel):
    answer: str
    is_relevant: bool
    is_tech: bool
    sources: List[SourceCard]
    path: str

# Helath check endpoint
@app.get("/health")
async def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    return {
        "status": "ok",
        "pipeline_ready": True,
        "total_vectors": pipeline.vector_store.index.ntotal,
        "total_answers": len(pipeline.answer_fetcher.answers_df),
        "model": pipeline.embedding_manager.model_name,
        "score_threshold": pipeline.score_threshold
    }

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = pipeline.run(
            query=request.query,
            top_k=request.top_k
        )

        source_cards = []
        for source in result["sources"]:
            source_cards.append(SourceCard(
                rank=source["rank"],
                question_id=source["question_id"],
                title=source["title"],
                primary_tag=source["primary_tag"],
                top_answers=source["top_answers"]
            ))

        return ChatResponse(
            answer = result["answer"],
            is_relevant = result["is_relevant"],
            is_tech = result.get("is_tech", True),  # Default to True if not provided
            sources = source_cards,
            path = result["path"]
        )
    
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
