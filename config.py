"""
Configuration settings for the RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenAI Configuration
    # To enable conversational AI responses, provide your OpenAI API key:
    # Option 1: export OPENAI_API_KEY="your-api-key"
    # Option 2: leave the environment variable unset and replace the value below
    # The default "sk-your-api-key" acts as a placeholder and will trigger local mode
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key")
    USE_OPENAI_EMBEDDINGS = True  # Will fallback to local if API key not available
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Embedding settings - user prefers text-embedding-3-small
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback local model
    
    # Vector store settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "mif_documents"
    
    # Chunking settings
    CHUNK_SIZE = 1000  # Characters
    CHUNK_OVERLAP = 200  # Characters
    MIN_CHUNK_SIZE = 100  # Minimum chunk size
    MAX_CHUNK_SIZE = 2000  # Maximum chunk size
    
    # Search settings
    SEMANTIC_SEARCH_K = 5  # Number of semantic results
    KEYWORD_SEARCH_K = 5   # Number of keyword results
    HYBRID_ALPHA = 0.7     # Weight for semantic search (0.0 = keyword only, 1.0 = semantic only)
    
    # Document processing
    DOCUMENTS_PATH = "./Documents"
    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]
    
    # Metadata fields
    METADATA_FIELDS = [
        "source",
        "page",
        "document_type",
        "title",
        "section",
        "chunk_index",
        "total_chunks"
    ] 
