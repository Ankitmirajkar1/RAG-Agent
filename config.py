# =====================================================
# CONFIG.PY - Load Environment Variables & Settings
# =====================================================
# Load .env file → Extract API keys and paths
# Used by all other files (backend, frontend, rag_core)

import os
from pathlib import Path
from dotenv import load_dotenv

# =====================================================
# LOAD .ENV FILE
# =====================================================
# This reads the .env file and loads all variables into os.environ
# .env file contains: OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
load_dotenv()

# =====================================================
# API KEYS (from .env file)
# =====================================================
# OpenAI API key - used to create embeddings and generate answers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Which OpenAI model to use (gpt-4, gpt-3.5-turbo, etc.)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# LangChain API key - for tracing/monitoring (optional)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# =====================================================
# FILE PATHS
# =====================================================
# Get the project root directory (where config.py is located)
BASE_DIR = Path(__file__).parent

# Path to the PDF document we're asking questions about
# Location: data/Rainbow-Bazaar-Return-Refund-&-Cancellation-Policy.pdf
PDF_PATH = BASE_DIR / "data" / "Rainbow-Bazaar-Return-Refund-&-Cancellation-Policy.pdf"

# Path where we save the vector store (cached embeddings)
# After first run, embeddings are saved here for fast retrieval
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"

# =====================================================
# CREATE DIRECTORIES
# =====================================================
# Make sure data and vector_store folders exist
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# VALIDATION
# =====================================================
# Check if PDF file exists
if not PDF_PATH.exists():
    print(f"⚠️ WARNING: PDF not found at {PDF_PATH}")
else:
    print(f"✅ PDF found: {PDF_PATH}")

# Check if API key is set
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set in .env file")
else:
    print("✅ OpenAI API key loaded")