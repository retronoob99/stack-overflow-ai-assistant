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
    alloqw_credentials=True,
)