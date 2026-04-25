# =====================================================
# BACKEND/MAIN.PY - FastAPI Backend
# =====================================================
# REST API endpoints:
# - GET  /health      → Check if backend is running
# - POST /ask         → Ask a question (returns answer)
# - GET  /docs        → Interactive API documentation
#
# Run: uv run python -m uvicorn backend.main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path so we can import rag_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core.simple_rag import rag

# =====================================================
# FASTAPI APP SETUP
# =====================================================
# Create FastAPI application
app = FastAPI(
    title="Rainbow Bazaar RAG API",
    description="API for asking questions about returns policy"
)

# =====================================================
# CORS MIDDLEWARE
# =====================================================
# Allow frontend to make requests to this backend
# CORS = Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# DATA MODELS (Request/Response)
# =====================================================
# Define the structure of API requests and responses

class QuestionRequest(BaseModel):
    """Model for question input"""
    question: str  # The user's question

class AnswerResponse(BaseModel):
    """Model for answer output"""
    answer: str  # The AI's answer
    status: str  # "success" or "error"


# =====================================================
# STARTUP EVENT
# =====================================================
# This runs once when the backend starts
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when backend starts"""
    print("\n" + "="*50)
    print("🚀 Starting Backend API")
    print("="*50)
    
    # Setup RAG (load or create vector store)
    rag.setup()
    
    print("\n✅ Backend ready to receive requests")
    print("📝 API Docs: http://localhost:8000/docs")
    print("="*50 + "\n")


# =====================================================
# HEALTH CHECK ENDPOINT
# =====================================================
# Used by frontend to check if backend is running
@app.get("/health")
async def health():
    """
    Health check endpoint
    Returns: {"status": "ok"}
    """
    return {"status": "ok"}


# =====================================================
# QUESTION/ANSWER ENDPOINT
# =====================================================
# Main endpoint - answers questions about the PDF
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about Rainbow Bazaar returns policy
    
    Request:
        {
            "question": "What is the return policy?"
        }
    
    Response:
        {
            "answer": "The answer...",
            "status": "success"
        }
    """
    try:
        # Get the question from request
        question = request.question
        print(f"\n❓ Question: {question}")
        
        # Use RAG to answer the question
        answer = rag.ask(question)
        
        print(f"✅ Answer: {answer[:100]}...")
        
        # Return the answer
        return AnswerResponse(
            answer=answer,
            status="success"
        )
    
    except Exception as e:
        # If error occurs, return error message
        print(f"❌ Error: {str(e)}")
        return AnswerResponse(
            answer=f"Error: {str(e)}",
            status="error"
        )


# =====================================================
# RUN BACKEND
# =====================================================
# Command: uv run python -m uvicorn backend.main:app --reload
# This starts the server on http://localhost:8000