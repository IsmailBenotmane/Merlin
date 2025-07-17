"""
BM25 keyword search implementation for hybrid search
"""
import re
import json
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import logging
import nltk
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from document_parser import DocumentChunk
from config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for keyword search"""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            logger.warning(f"Stopwords for {language} not found, using basic stopwords")
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }
        
        # Add custom academic/business stopwords
        self.stop_words.update({
            'also', 'may', 'can', 'must', 'shall', 'might', 'every', 'each', 'any',
            'some', 'many', 'much', 'more', 'most', 'very', 'quite', 'rather',
            'program', 'course', 'student', 'university', 'school'  # Domain-specific
        })
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword search"""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Filter and process tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens and stopwords
            if len(token) < 2 or token in self.stop_words:
                continue
            
            # Handle hyphenated words
            if '-' in token:
                parts = token.split('-')
                for part in parts:
                    if len(part) >= 2 and part not in self.stop_words:
                        processed_tokens.append(self.stemmer.stem(part))
            else:
                # Stem the token
                processed_tokens.append(self.stemmer.stem(token))
        
        return processed_tokens
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that should be preserved"""
        key_phrases = []
        
        # Common academic/business phrases
        phrase_patterns = [
            r'\b(?:master|bachelor|phd|doctorate)\s+(?:of|in)\s+\w+(?:\s+\w+)*',
            r'\b(?:international|global)\s+\w+(?:\s+\w+)*',
            r'\b(?:career|job|employment)\s+\w+(?:\s+\w+)*',
            r'\b(?:admission|application)\s+\w+(?:\s+\w+)*',
            r'\b\w+\s+(?:program|course|curriculum)',
            r'\b(?:financial|finance|business|management)\s+\w+(?:\s+\w+)*'
        ]
        
        text_lower = text.lower()
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text_lower)
            key_phrases.extend(matches)
        
        # Clean and deduplicate
        cleaned_phrases = []
        for phrase in key_phrases:
            phrase = phrase.strip()
            if len(phrase) > 5 and phrase not in cleaned_phrases:
                cleaned_phrases.append(phrase)
        
        return cleaned_phrases

class BM25KeywordSearch:
    """BM25-based keyword search implementation"""
    
    def __init__(self, 
                 k1: float = 1.5, 
                 b: float = 0.75,
                 preprocessor: TextPreprocessor = None):
        """
        Initialize BM25 search
        
        Args:
            k1: Term frequency saturation parameter
            b: Field length normalization parameter
            preprocessor: Text preprocessor instance
        """
        self.k1 = k1
        self.b = b
        self.preprocessor = preprocessor or TextPreprocessor()
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        self.key_phrases_index = {}
        
    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index document chunks for keyword search"""
        logger.info(f"Indexing {len(chunks)} chunks for keyword search")
        
        self.chunks = chunks
        self.tokenized_corpus = []
        self.key_phrases_index = {}
        
        for i, chunk in enumerate(chunks):
            # Preprocess text
            tokens = self.preprocessor.preprocess_text(chunk.text)
            self.tokenized_corpus.append(tokens)
            
            # Extract and index key phrases
            key_phrases = self.preprocessor.extract_key_phrases(chunk.text)
            for phrase in key_phrases:
                if phrase not in self.key_phrases_index:
                    self.key_phrases_index[phrase] = []
                self.key_phrases_index[phrase].append(i)
        
        # Initialize BM25
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.info("BM25 index created successfully")
        else:
            logger.warning("No valid documents to index")
    
    def search(self, 
               query: str, 
               k: int = Config.KEYWORD_SEARCH_K,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for documents using keyword matching
        
        Returns:
            List of tuples (DocumentChunk, BM25_score)
        """
        if not self.bm25 or not self.chunks:
            logger.warning("No documents indexed for keyword search")
            return []
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_text(query)
        
        if not query_tokens:
            logger.warning("Query produced no valid tokens")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Boost scores for key phrase matches
        phrase_boosted_scores = self._apply_phrase_boost(query, scores)
        
        # Create scored results
        results = []
        for i, score in enumerate(phrase_boosted_scores):
            if score > 0:  # Only include documents with positive scores
                chunk = self.chunks[i]
                
                # Apply metadata filtering if provided
                if filter_dict and not self._matches_filter(chunk, filter_dict):
                    continue
                
                results.append((chunk, float(score)))
        
        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Keyword search returned {min(k, len(results))} results")
        return results[:k]
    
    def _apply_phrase_boost(self, query: str, base_scores: List[float]) -> List[float]:
        """Apply boosting for exact phrase matches"""
        boosted_scores = base_scores.copy()
        query_lower = query.lower()
        
        # Check for exact phrase matches in key phrases
        for phrase, chunk_indices in self.key_phrases_index.items():
            if phrase in query_lower:
                boost_factor = 1.5  # Boost exact phrase matches
                for idx in chunk_indices:
                    if idx < len(boosted_scores):
                        boosted_scores[idx] *= boost_factor
        
        # Check for exact phrase matches in chunk text
        for i, chunk in enumerate(self.chunks):
            chunk_text_lower = chunk.text.lower()
            
            # Simple phrase matching (could be improved with fuzzy matching)
            if query_lower in chunk_text_lower:
                boosted_scores[i] *= 1.3  # Moderate boost for exact text matches
        
        return boosted_scores
    
    def _matches_filter(self, chunk: DocumentChunk, filter_dict: Dict[str, Any]) -> bool:
        """Check if chunk matches the provided filters"""
        for key, value in filter_dict.items():
            chunk_value = chunk.metadata.get(key)
            
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            elif isinstance(value, dict):
                # Handle range queries
                if '$gt' in value and (chunk_value is None or chunk_value <= value['$gt']):
                    return False
                if '$gte' in value and (chunk_value is None or chunk_value < value['$gte']):
                    return False
                if '$lt' in value and (chunk_value is None or chunk_value >= value['$lt']):
                    return False
                if '$lte' in value and (chunk_value is None or chunk_value > value['$lte']):
                    return False
                if '$ne' in value and chunk_value == value['$ne']:
                    return False
            else:
                if chunk_value != value:
                    return False
        
        return True
    
    def get_term_frequencies(self, query: str) -> Dict[str, int]:
        """Get term frequencies for a query across the corpus"""
        if not self.bm25:
            return {}
        
        query_tokens = self.preprocessor.preprocess_text(query)
        term_freqs = {}
        
        for token in query_tokens:
            count = 0
            for doc_tokens in self.tokenized_corpus:
                count += doc_tokens.count(token)
            term_freqs[token] = count
        
        return term_freqs
    
    def save_index(self, filepath: str) -> None:
        """Save the BM25 index to disk"""
        index_data = {
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'key_phrases_index': self.key_phrases_index,
            'k1': self.k1,
            'b': self.b
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            logger.info(f"BM25 index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
    
    def load_index(self, filepath: str) -> bool:
        """Load the BM25 index from disk"""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.chunks = index_data['chunks']
            self.tokenized_corpus = index_data['tokenized_corpus']
            self.key_phrases_index = index_data['key_phrases_index']
            self.k1 = index_data['k1']
            self.b = index_data['b']
            
            # Recreate BM25 instance
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
                logger.info(f"BM25 index loaded from {filepath}")
                return True
            
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
        
        return False

class KeywordSearchManager:
    """High-level manager for keyword search operations"""
    
    def __init__(self, 
                 search_engine: BM25KeywordSearch = None,
                 index_file: str = "./bm25_index.pkl"):
        self.search_engine = search_engine or BM25KeywordSearch()
        self.index_file = index_file
    
    def index_documents(self, chunks: List[DocumentChunk], save_index: bool = True) -> Dict[str, Any]:
        """Index documents and optionally save index to disk"""
        start_time = time.time()
        
        self.search_engine.index_documents(chunks)
        
        if save_index:
            self.search_engine.save_index(self.index_file)
        
        end_time = time.time()
        
        stats = {
            'chunks_indexed': len(chunks),
            'indexing_time': end_time - start_time,
            'chunks_per_second': len(chunks) / (end_time - start_time) if end_time > start_time else 0,
            'index_file': self.index_file if save_index else None
        }
        
        logger.info(f"Keyword indexing completed: {stats}")
        return stats
    
    def load_existing_index(self) -> bool:
        """Load existing index from disk"""
        if Path(self.index_file).exists():
            return self.search_engine.load_index(self.index_file)
        return False
    
    def search_keywords(self, 
                       query: str, 
                       k: int = Config.KEYWORD_SEARCH_K,
                       filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search using keyword matching"""
        return self.search_engine.search(query, k, filters)
    
    def analyze_query_terms(self, query: str) -> Dict[str, Any]:
        """Analyze query terms and their frequencies in the corpus"""
        term_freqs = self.search_engine.get_term_frequencies(query)
        preprocessed_query = self.search_engine.preprocessor.preprocess_text(query)
        
        analysis = {
            'original_query': query,
            'preprocessed_terms': preprocessed_query,
            'term_frequencies': term_freqs,
            'total_terms': len(preprocessed_query),
            'unique_terms': len(set(preprocessed_query))
        }
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    import time
    from document_parser import DocumentParser
    from chunking_strategy import DocumentChunker
    
    logger.info("Testing Keyword Search")
    
    # Initialize components
    parser = DocumentParser()
    chunker = DocumentChunker()
    keyword_manager = KeywordSearchManager()
    
    # Parse and chunk documents
    sample_path = "./Documents"
    if Path(sample_path).exists():
        logger.info("Parsing documents...")
        doc_chunks = parser.parse_directory(sample_path)
        
        logger.info("Chunking documents...")
        text_chunks = chunker.chunk_documents(doc_chunks)
        
        # Try to load existing index
        if not keyword_manager.load_existing_index():
            logger.info("Indexing chunks for keyword search...")
            stats = keyword_manager.index_documents(text_chunks)
            print(f"Keyword indexing stats: {stats}")
        else:
            logger.info("Loaded existing keyword search index")
        
        # Test keyword search
        test_queries = [
            "Master in International Finance",
            "employment opportunities career",
            "course curriculum program",
            "admission requirements application"
        ]
        
        for query in test_queries:
            print(f"\nKeyword search for: '{query}'")
            
            # Analyze query
            analysis = keyword_manager.analyze_query_terms(query)
            print(f"Query analysis: {analysis}")
            
            # Search
            results = keyword_manager.search_keywords(query, k=3)
            
            for i, (chunk, score) in enumerate(results):
                print(f"  Result {i+1} (BM25 score: {score:.3f}):")
                print(f"    Source: {chunk.metadata.get('source', 'unknown')}")
                print(f"    Page: {chunk.metadata.get('page', 'unknown')}")
                print(f"    Text: {chunk.text[:150]}...")
                print()
    else:
        logger.warning(f"Documents directory not found: {sample_path}") 
