# =====================================================
# RAG_CORE/SIMPLE_RAG.PY - Core RAG Logic
# =====================================================
# How RAG works:
# 1. Load PDF document
# 2. Split text into chunks
# 3. Convert chunks to vectors (embeddings) using OpenAI
# 4. Store vectors in FAISS database for fast search
# 5. When user asks: find relevant chunks + ask LLM to answer
#
# This makes the LLM aware of YOUR specific document

import sys
from pathlib import Path

# Add parent directory to Python path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import configuration
from config import PDF_PATH, VECTOR_STORE_PATH, OPENAI_API_KEY, OPENAI_MODEL


class SimpleRAG:
    """
    Simple RAG class - handles everything for question answering
    
    Methods:
    - __init__(): Initialize embeddings and LLM
    - setup(): Load or create vector store
    - process_pdf(): Load PDF and create embeddings
    - load_vector_store(): Load cached embeddings
    - ask(): Answer a question using RAG
    """
    
    def __init__(self):
        """Initialize RAG components"""
        print("📦 Initializing RAG...")
        
        # Create embeddings object - converts text to vectors using OpenAI
        # Vectors help find similar text chunks (semantic search)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Vector store will hold all PDF embeddings
        self.vector_store = None
        
        # Create LLM object - this answers questions
        # temperature=0.7: medium creativity (0=consistent, 1=creative)
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.7
        )
    
    def setup(self):
        """Setup RAG - either load existing or create new"""
        print("\n🚀 Setting up RAG system...")
        
        # Check if vector store already exists (from previous run)
        if VECTOR_STORE_PATH.exists():
            print("📂 Loading cached vector store...")
            self.load_vector_store()
        else:
            print("🔄 Creating new vector store from PDF...")
            self.process_pdf()
    
    def process_pdf(self):
        """Load PDF, create embeddings, store in FAISS"""
        print(f"📄 Loading PDF: {PDF_PATH}")
        
        # Load PDF document
        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDF")
        
        # Split text into chunks
        # chunk_size=500: each chunk is ~500 characters
        # overlap=50: 50 character overlap between chunks (keeps context)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Create embeddings and store in FAISS
        print("🔄 Creating embeddings and storing in FAISS...")
        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings
        )
        
        # Save vector store for next time
        self.vector_store.save_local(str(VECTOR_STORE_PATH))
        print(f"💾 Vector store saved to {VECTOR_STORE_PATH}")
    
    def load_vector_store(self):
        """Load previously saved vector store"""
        print(f"📂 Loading vector store from {VECTOR_STORE_PATH}")
        self.vector_store = FAISS.load_local(
            str(VECTOR_STORE_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded")
    
    def ask(self, question):
        """Answer a question using RAG"""
        if not self.vector_store:
            return "Error: Vector store not initialized"
        
        # Create the RAG chain
        # Step 1: Retrieve relevant chunks from vector store
        retriever = self.vector_store.as_retriever(k=3)  # Get top 3 relevant chunks
        
        # Step 2: Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """Use the following context from Rainbow Bazaar's policy document to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Step 3: Create the chain: retrieve → format → ask LLM
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute the chain and get answer
        answer = chain.invoke(question)
        return answer


# =====================================================
# CREATE GLOBAL RAG INSTANCE
# =====================================================
# This is used by the backend API
# Initialize once when backend starts
rag = SimpleRAG()