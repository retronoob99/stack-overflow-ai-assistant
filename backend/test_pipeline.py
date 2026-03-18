import os
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    vector_store_path = "C:\\Users\\satya\\Desktop\\Stack_Overflow_AI_Assistant\\data\\vector_store",
    answers_path      = "C:\\Users\\satya\\Desktop\\Stack_Overflow_AI_Assistant\\data\\answers_cleaned.parquet",
    score_threshold   = 0.5
)

# ─── Test 1 — Relevant tech query (RAG path) ──────────────────────────────────
print("\n" + "=" * 60)
print("TEST 1 — Relevant tech query")
print("=" * 60)
result = pipeline.run("how to reverse a list in python?")
print(f"Path     : {result['path']}")
print(f"Answer   : {result['answer'][:300]}...")
print(f"Sources  : {len(result['sources'])} found")

# ─── Test 2 — Tech query not in dataset (General LLM path) ───────────────────
print("\n" + "=" * 60)
print("TEST 2 — Tech query not in dataset")
print("=" * 60)
result = pipeline.run("explain quantum computing algorithms")
print(f"Path     : {result['path']}")
print(f"Answer   : {result['answer'][:300]}...")
print(f"Sources  : {len(result['sources'])} found")

# ─── Test 3 — Non tech query (Rejected path) ─────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3 — Non tech query")
print("=" * 60)
result = pipeline.run("what should i eat for dinner?")
print(f"Path     : {result['path']}")
print(f"Answer   : {result['answer'][:300]}")
print(f"Sources  : {len(result['sources'])} found")