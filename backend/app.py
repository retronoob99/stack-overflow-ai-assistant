from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Stack Overflow AI Assistant API",
    description="API for the Stack Overflow AI Assistant using RAG pipeline",
    version="1.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development, restrict in production
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
    allow_credentials=True,
)

# Initialize RAG Pipeline
pipeline: Optional[RAGPipeline] = None

# Startup event to initialize the RAG pipeline
@app.on_event("startup")
async def startup_event():
    global pipeline
    
    print("🚀 Starting Stack Overflow AI Assistant API...")

    pipeline = RAGPipeline(
        vector_store_path=os.getenv("VECTOR_STORE_PATH"),
        answers_path=os.getenv("ANSWERS_PATH"),
        score_threshold=0.5,
    )
    print("✅ RAG Pipeline initialized successfully!")

    yield

    print("🛑 Shutting down Stack Overflow AI Assistant API..." )

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