import pandas as pd
import numpy as np
from langchain_core.documents import Document
import torch
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
import pickle

# Embedding Manager
class EmbeddingManager:
    '''Manages embedding generation and storage using SentenceTransformer'''

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        '''Loads the SentenceTransformer model'''
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Loaded embedding model: {self.model_name}")
            print(f"Model device: {self.device}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generates embeddings from a list of LangChain Documents"""

        if not self.model:
            raise ValueError("Model not loaded successfully.")
        
        # Extract page_content from each Document
        texts = [doc.page_content for doc in documents]

        print(f"\nGenerating embeddings for {len(texts):,} documents on {self.device}...")
        batch_size = 512 if self.device == "cuda" else 32
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=self.device,
            convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print(f"GPU cache cleared")
        return embeddings
    

# Vector Store
class VectorStore:
    '''Manages vector storage and retrieval using FAISS with GPU support'''

    def __init__(self, 
                 index_name: str = "questions",
                 persist_directory: str = "../data/vector_store"):
        
        self.index_name        = index_name
        self.persist_directory = persist_directory
        self.index             = None
        self.documents         = []
        self.use_gpu           = faiss.get_num_gpus() > 0
        
        os.makedirs(self.persist_directory, exist_ok=True)
        print(f"✅ Vector store initialized")
        print(f"✅ FAISS GPU available : {self.use_gpu}")
        print(f"✅ Persist directory  : {self.persist_directory}")

    def _build_index(self, embeddings: np.ndarray):
        '''Builds FAISS index from embeddings'''

        dimension = embeddings.shape[1]          # 384 for all-MiniLM-L6-v2

        # IVFFlat index — faster search on large datasets (500K+)
        # nlist = number of clusters (rule of thumb: sqrt of total docs)
        nlist      = int(np.sqrt(len(embeddings)))
        quantizer  = faiss.IndexFlatIP(dimension)         # Inner product (cosine if normalized)
        index      = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # Move to GPU if available
        if self.use_gpu:
            res        = faiss.StandardGpuResources()
            index      = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"✅ FAISS index moved to GPU")

        # Train index (required for IVFFlat)
        print(f"Training FAISS index on {len(embeddings):,} vectors...")
        index.train(embeddings.astype(np.float32))

        # Add embeddings with their integer IDs
        index.add_with_ids(
            embeddings.astype(np.float32),
            np.arange(len(embeddings))            # IDs = 0, 1, 2, ... n
        )

        print(f"✅ FAISS index built — total vectors: {index.ntotal:,}")
        return index

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        '''Builds FAISS index and stores documents'''

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        print(f"\nBuilding FAISS index for {len(documents):,} documents...")

        # Store documents
        self.documents = documents

        # Build FAISS index
        self.index = self._build_index(embeddings)

        print(f"✅ Successfully added {len(documents):,} documents")
        print(f"✅ Index total vectors: {self.index.ntotal:,}")

    def save(self):
        '''Saves FAISS index and documents locally'''

        index_path    = os.path.join(self.persist_directory, f"{self.index_name}.index")
        docs_path     = os.path.join(self.persist_directory, f"{self.index_name}_docs.pkl")

        # Move back to CPU before saving
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, index_path)

        # Save documents
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"✅ FAISS index saved  : {index_path}")
        print(f"✅ Documents saved    : {docs_path}")

    def load(self):
        '''Loads FAISS index and documents from disk'''

        index_path = os.path.join(self.persist_directory, f"{self.index_name}.index")
        docs_path  = os.path.join(self.persist_directory, f"{self.index_name}_docs.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved index found at {index_path}")

        # Load index
        cpu_index = faiss.read_index(index_path)

        # Move to GPU if available
        if self.use_gpu:
            res        = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"✅ FAISS index loaded to GPU")
        else:
            self.index = cpu_index

        # Load documents
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        print(f"✅ Loaded index with {self.index.ntotal:,} vectors")
        print(f"✅ Loaded {len(self.documents):,} documents")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, nprobe: int = 10):
        '''Searches for similar documents given a query embedding'''

        if self.index is None:
            raise ValueError("Index not built. Run add_documents() or load() first.")

        # nprobe = number of clusters to search (higher = more accurate but slower)
        self.index.nprobe = nprobe

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "document" : self.documents[idx],
                "score"    : float(dist),
                "index"    : int(idx)
            })
        return results

    def get_stats(self):
        '''Prints vector store statistics'''
        print(f"\n{'=' * 40}")
        print(f"   VECTOR STORE STATS")
        print(f"{'=' * 40}")
        print(f"   Index name   : {self.index_name}")
        print(f"   Total vectors: {self.index.ntotal:,}" if self.index else "   Index        : Not built")
        print(f"   Documents    : {len(self.documents):,}")
        print(f"   GPU enabled  : {self.use_gpu}")
        print(f"   Directory    : {self.persist_directory}")
        print(f"{'=' * 40}")

# RAG Retriever
class RAGretriever:
    '''Handles query based retrieval from the vector store'''

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.5, nprobe: int = 10):
        '''Retrieve relevant documents for a query'''
        print(f"\nQuery          : '{query}'")
        print(f"Top-K          : {top_k}")
        print(f"Score threshold: {score_threshold}")
        
        # Generate embedding for the query
        query_embedding = self.embedding_manager.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # Search in vector store
        try:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                nprobe=nprobe
            )

            # Filter results based on score threshold
            retrieved_docs = []
            for rank, result in enumerate(results):
                if result["score"] >= score_threshold:
                    doc = result["document"]
                    metadata = doc.metadata

                    retrieved_docs.append({
                        "rank": rank + 1,
                        "score": round(result["score"], 4),
                        "question_id": metadata.get("question_id"),
                        "title": metadata.get("title"),
                        "primary_tag": metadata.get("primary_tag"),
                        "tags": metadata.get("tags"),
                        "answer_count": metadata.get("answer_count"),
                        "has_accepted_answer": metadata.get("has_accepted_answer"),
                        "score_votes": metadata.get("score"),
                        "view_count": metadata.get("view_count"),
                        "has_code": metadata.get("has_code"),
                        "creation_date": metadata.get("creation_date"),
                        "tag_popularity": metadata.get("tag_popularity"),
                        "page_content": doc.page_content
                    })

            print(f"Retrieved {len(retrieved_docs)} documents\n")

            for doc in retrieved_docs:
                print(f"  Rank {doc['rank']} | Score: {doc['score']:.4f}")
                print(f"  Title : {doc['title']}")
                print(f"  Tag   : {doc['primary_tag']} | Answers: {doc['answer_count']} | Accepted: {doc['has_accepted_answer']}")
                print(f"  {'─' * 55}")

            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise

    def retrieve_by_tag(self, query: str, tag: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''Retrieves documents filtered by a specific tag'''

        results = self.retrieve(query, top_k=top_k * 3)  # fetch more, then filter

        filtered = [r for r in results if r["primary_tag"] == tag][:top_k]

        print(f"Filtered to {len(filtered)} results for tag: '{tag}'")
        return filtered

    def retrieve_answered_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''Retrieves only questions that have accepted answers'''

        results = self.retrieve(query, top_k=top_k * 3)  # fetch more, then filter

        filtered = [r for r in results if r["has_accepted_answer"]][:top_k]

        print(f"Filtered to {len(filtered)} answered questions")
        return filtered