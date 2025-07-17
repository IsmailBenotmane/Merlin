"""
Vector store implementation using ChromaDB with OpenAI embeddings
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import json
import hashlib
import time
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from document_parser import DocumentChunk
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings using OpenAI or local models"""
    
    def __init__(self, use_openai: bool = None, api_key: str = None):
        self.use_openai = use_openai if use_openai is not None else Config.USE_OPENAI_EMBEDDINGS
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.local_model = None
        
        if self.use_openai and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
                self.model_name = Config.EMBEDDING_MODEL
                logger.info(f"Using OpenAI embeddings: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.use_openai = False
                self._init_local_model()
        else:
            # Fallback to local model
            self.use_openai = False
            self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local embedding model"""
        if self.local_model is None:
            try:
                self.model_name = Config.LOCAL_EMBEDDING_MODEL
                self.local_model = SentenceTransformer(self.model_name)
                logger.info(f"Local embedding model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                # Fallback to a simpler model
                self.model_name = 'all-MiniLM-L6-v2'
                self.local_model = SentenceTransformer(self.model_name)
                logger.info(f"Using fallback model: {self.model_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.use_openai:
            return self._generate_openai_embeddings(texts)
        else:
            return self._generate_local_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Process in batches to handle rate limits
            batch_size = 100  # OpenAI's batch limit
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            logger.info("Falling back to local embeddings")
            return self._generate_local_embeddings(texts)
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        try:
            # Initialize local model if not already done (fallback case)
            if self.local_model is None:
                self.model_name = Config.LOCAL_EMBEDDING_MODEL
                self._init_local_model()
            
            embeddings = self.local_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise

class ChromaVectorStore:
    """ChromaDB vector store for semantic search"""
    
    def __init__(self, 
                 persist_directory: str = Config.CHROMA_PERSIST_DIRECTORY,
                 collection_name: str = Config.COLLECTION_NAME,
                 embedding_generator: EmbeddingGenerator = None):
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "MIF Documents RAG System"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> None:
        """Add document chunks to the vector store"""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._add_batch(batch, i // batch_size + 1, (len(chunks) - 1) // batch_size + 1)
    
    def _add_batch(self, chunks: List[DocumentChunk], batch_num: int, total_batches: int) -> None:
        """Add a batch of chunks to the vector store"""
        texts = [chunk.text for chunk in chunks]
        ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for batch {batch_num}/{total_batches}")
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to collection
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added batch {batch_num}/{total_batches} to vector store")
        except Exception as e:
            logger.error(f"Error adding batch {batch_num}: {e}")
            raise
    
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk"""
        # Create a hash based on source, page, and chunk index
        content = f"{chunk.metadata.get('source', '')}_{chunk.page_number}_{chunk.chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (only string, int, float, bool)"""
        metadata = {}
        
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif value is not None:
                metadata[key] = str(value)
        
        # Ensure required fields
        metadata['chunk_text_length'] = len(chunk.text)
        if chunk.page_number:
            metadata['page_number'] = chunk.page_number
        if chunk.chunk_index is not None:
            metadata['chunk_index'] = chunk.chunk_index
            
        return metadata
    
    def similarity_search(self, 
                         query: str, 
                         k: int = Config.SEMANTIC_SEARCH_K,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform similarity search
        
        Returns:
            List of tuples (DocumentChunk, similarity_score)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            
            # Build where clause for filtering
            where_clause = self._build_where_clause(filter_dict) if filter_dict else None
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to DocumentChunk objects
            chunks_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (closer to 1 is more similar)
                    similarity = 1.0 - distance
                    
                    # Reconstruct DocumentChunk
                    chunk = DocumentChunk(
                        text=doc,
                        metadata=metadata,
                        page_number=metadata.get('page_number'),
                        chunk_index=metadata.get('chunk_index')
                    )
                    
                    chunks_with_scores.append((chunk, similarity))
            
            logger.info(f"Similarity search returned {len(chunks_with_scores)} results")
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _build_where_clause(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filter dictionary"""
        where_clause = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                # Handle list values with $in operator
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict) and any(op in value for op in ['$gt', '$gte', '$lt', '$lte', '$ne']):
                # Handle range queries
                where_clause[key] = value
            else:
                # Exact match
                where_clause[key] = value
        
        return where_clause
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.get(
                limit=min(100, count),
                include=['metadatas']
            )
            
            stats = {
                'total_chunks': count,
                'embedding_model': self.embedding_generator.model_name,
                'use_openai': self.embedding_generator.use_openai,
                'collection_name': self.collection_name
            }
            
            if sample_results['metadatas']:
                # Analyze document types
                doc_types = {}
                sources = set()
                
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('document_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    sources.add(metadata.get('source', 'unknown'))
                
                stats.update({
                    'document_types': doc_types,
                    'unique_sources': list(sources),
                    'total_sources': len(sources)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "MIF Documents RAG System"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source document"""
        try:
            # Get all chunks from the source
            results = self.collection.get(
                where={"source": source},
                include=['ids']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from source: {source}")
            else:
                logger.info(f"No chunks found for source: {source}")
                
        except Exception as e:
            logger.error(f"Error deleting chunks from source {source}: {e}")
            raise

class VectorStoreManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, vector_store: ChromaVectorStore = None):
        self.vector_store = vector_store or ChromaVectorStore()
    
    def index_documents(self, chunks: List[DocumentChunk], replace_existing: bool = False) -> Dict[str, Any]:
        """Index a list of document chunks"""
        if replace_existing:
            sources = set(chunk.metadata.get('source', '') for chunk in chunks)
            for source in sources:
                if source:
                    self.vector_store.delete_by_source(source)
        
        # Add chunks to vector store
        start_time = time.time()
        self.vector_store.add_chunks(chunks)
        end_time = time.time()
        
        # Return indexing statistics
        stats = {
            'chunks_indexed': len(chunks),
            'indexing_time': end_time - start_time,
            'chunks_per_second': len(chunks) / (end_time - start_time) if end_time > start_time else 0,
            'collection_stats': self.vector_store.get_collection_stats()
        }
        
        logger.info(f"Indexing completed: {stats}")
        return stats
    
    def search_similar(self, 
                      query: str, 
                      k: int = Config.SEMANTIC_SEARCH_K,
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks with optional filtering"""
        return self.vector_store.similarity_search(query, k, filters)

# Example usage and testing
if __name__ == "__main__":
    import time
    from document_parser import DocumentParser
    from chunking_strategy import DocumentChunker
    
    # Test the vector store
    logger.info("Testing Vector Store")
    
    # Initialize components
    parser = DocumentParser()
    chunker = DocumentChunker()
    vector_store_manager = VectorStoreManager()
    
    # Parse and chunk documents
    sample_path = "./Documents"
    if Path(sample_path).exists():
        logger.info("Parsing documents...")
        doc_chunks = parser.parse_directory(sample_path)
        
        logger.info("Chunking documents...")
        text_chunks = chunker.chunk_documents(doc_chunks)
        
        logger.info("Indexing chunks...")
        stats = vector_store_manager.index_documents(text_chunks, replace_existing=True)
        print(f"Indexing stats: {stats}")
        
        # Test search
        test_queries = [
            "Master in International Finance program",
            "career opportunities",
            "course curriculum",
            "admission requirements"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = vector_store_manager.search_similar(query, k=3)
            
            for i, (chunk, score) in enumerate(results):
                print(f"  Result {i+1} (score: {score:.3f}):")
                print(f"    Source: {chunk.metadata.get('source', 'unknown')}")
                print(f"    Page: {chunk.metadata.get('page', 'unknown')}")
                print(f"    Text: {chunk.text[:150]}...")
                print()
    else:
        logger.warning(f"Documents directory not found: {sample_path}") 
