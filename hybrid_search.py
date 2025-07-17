"""
Hybrid search implementation combining semantic and keyword search
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

from document_parser import DocumentChunk
from vector_store import VectorStoreManager
from keyword_search import KeywordSearchManager
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with multiple scores"""
    chunk: DocumentChunk
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    rank_semantic: int = 0
    rank_keyword: int = 0
    rank_hybrid: int = 0

class HybridSearchEngine:
    """Advanced hybrid search combining semantic and keyword search"""
    
    def __init__(self,
                 vector_store_manager: VectorStoreManager = None,
                 keyword_manager: KeywordSearchManager = None,
                 alpha: float = Config.HYBRID_ALPHA):
        """
        Initialize hybrid search engine
        
        Args:
            vector_store_manager: Semantic search manager
            keyword_manager: Keyword search manager  
            alpha: Weight for semantic search (0.0 = keyword only, 1.0 = semantic only)
        """
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.keyword_manager = keyword_manager or KeywordSearchManager()
        self.alpha = alpha
        self.fusion_methods = {
            'weighted_sum': self._weighted_sum_fusion,
            'rank_fusion': self._reciprocal_rank_fusion,
            'harmonic_mean': self._harmonic_mean_fusion,
            'geometric_mean': self._geometric_mean_fusion
        }
    
    def search(self,
               query: str,
               k: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               fusion_method: str = 'weighted_sum',
               semantic_k: Optional[int] = None,
               keyword_k: Optional[int] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query: Search query
            k: Number of final results to return
            filters: Metadata filters to apply
            fusion_method: Method to combine results ('weighted_sum', 'rank_fusion', 'harmonic_mean', 'geometric_mean')
            semantic_k: Number of semantic results to retrieve (default: k*2)
            keyword_k: Number of keyword results to retrieve (default: k*2)
            
        Returns:
            List of SearchResult objects sorted by hybrid score
        """
        # Set default retrieval counts
        semantic_k = semantic_k or min(k * 2, Config.SEMANTIC_SEARCH_K * 2)
        keyword_k = keyword_k or min(k * 2, Config.KEYWORD_SEARCH_K * 2)
        
        logger.info(f"Hybrid search for query: '{query}'")
        logger.info(f"Retrieving {semantic_k} semantic + {keyword_k} keyword results")
        
        # Perform both searches
        semantic_results = self.vector_store_manager.search_similar(
            query, k=semantic_k, filters=filters
        )
        
        keyword_results = self.keyword_manager.search_keywords(
            query, k=keyword_k, filters=filters
        )
        
        logger.info(f"Retrieved {len(semantic_results)} semantic and {len(keyword_results)} keyword results")
        
        # Convert to SearchResult objects and combine
        all_results = self._combine_results(semantic_results, keyword_results)
        
        # Apply fusion method
        if fusion_method in self.fusion_methods:
            fused_results = self.fusion_methods[fusion_method](all_results)
        else:
            logger.warning(f"Unknown fusion method: {fusion_method}, using weighted_sum")
            fused_results = self._weighted_sum_fusion(all_results)
        
        # Sort by hybrid score and return top k
        fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Add hybrid rankings
        for i, result in enumerate(fused_results[:k]):
            result.rank_hybrid = i + 1
        
        logger.info(f"Hybrid search returning {min(k, len(fused_results))} results")
        return fused_results[:k]
    
    def _combine_results(self, 
                        semantic_results: List[Tuple[DocumentChunk, float]], 
                        keyword_results: List[Tuple[DocumentChunk, float]]) -> List[SearchResult]:
        """Combine semantic and keyword results into SearchResult objects"""
        
        # Create a mapping from chunk ID to SearchResult
        results_map = {}
        
        # Process semantic results
        for rank, (chunk, score) in enumerate(semantic_results):
            chunk_id = self._get_chunk_id(chunk)
            
            if chunk_id not in results_map:
                results_map[chunk_id] = SearchResult(chunk=chunk)
            
            results_map[chunk_id].semantic_score = score
            results_map[chunk_id].rank_semantic = rank + 1
        
        # Process keyword results
        for rank, (chunk, score) in enumerate(keyword_results):
            chunk_id = self._get_chunk_id(chunk)
            
            if chunk_id not in results_map:
                results_map[chunk_id] = SearchResult(chunk=chunk)
            
            results_map[chunk_id].keyword_score = score
            results_map[chunk_id].rank_keyword = rank + 1
        
        return list(results_map.values())
    
    def _get_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique identifier for a chunk"""
        return f"{chunk.metadata.get('source', '')}_{chunk.page_number}_{chunk.chunk_index}"
    
    def _weighted_sum_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine results using weighted sum of normalized scores"""
        
        # Normalize scores to [0, 1] range
        semantic_scores = [r.semantic_score for r in results if r.semantic_score > 0]
        keyword_scores = [r.keyword_score for r in results if r.keyword_score > 0]
        
        max_semantic = max(semantic_scores) if semantic_scores else 1.0
        max_keyword = max(keyword_scores) if keyword_scores else 1.0
        
        for result in results:
            norm_semantic = result.semantic_score / max_semantic if max_semantic > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            # Weighted combination
            result.hybrid_score = (self.alpha * norm_semantic + 
                                 (1 - self.alpha) * norm_keyword)
        
        return results
    
    def _reciprocal_rank_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)"""
        
        k = 60  # RRF parameter
        
        for result in results:
            rrf_score = 0.0
            
            # Add semantic contribution
            if result.rank_semantic > 0:
                rrf_score += self.alpha / (k + result.rank_semantic)
            
            # Add keyword contribution  
            if result.rank_keyword > 0:
                rrf_score += (1 - self.alpha) / (k + result.rank_keyword)
            
            result.hybrid_score = rrf_score
        
        return results
    
    def _harmonic_mean_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine results using harmonic mean of normalized scores"""
        
        # Normalize scores
        semantic_scores = [r.semantic_score for r in results if r.semantic_score > 0]
        keyword_scores = [r.keyword_score for r in results if r.keyword_score > 0]
        
        max_semantic = max(semantic_scores) if semantic_scores else 1.0
        max_keyword = max(keyword_scores) if keyword_scores else 1.0
        
        for result in results:
            norm_semantic = result.semantic_score / max_semantic if max_semantic > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            # Harmonic mean (handles cases where one score is 0)
            if norm_semantic > 0 and norm_keyword > 0:
                result.hybrid_score = 2 * norm_semantic * norm_keyword / (norm_semantic + norm_keyword)
            elif norm_semantic > 0:
                result.hybrid_score = self.alpha * norm_semantic
            elif norm_keyword > 0:
                result.hybrid_score = (1 - self.alpha) * norm_keyword
            else:
                result.hybrid_score = 0.0
        
        return results
    
    def _geometric_mean_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine results using geometric mean of normalized scores"""
        
        # Normalize scores
        semantic_scores = [r.semantic_score for r in results if r.semantic_score > 0]
        keyword_scores = [r.keyword_score for r in results if r.keyword_score > 0]
        
        max_semantic = max(semantic_scores) if semantic_scores else 1.0
        max_keyword = max(keyword_scores) if keyword_scores else 1.0
        
        for result in results:
            norm_semantic = result.semantic_score / max_semantic if max_semantic > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            # Geometric mean with alpha weighting
            if norm_semantic > 0 and norm_keyword > 0:
                result.hybrid_score = (norm_semantic ** self.alpha) * (norm_keyword ** (1 - self.alpha))
            elif norm_semantic > 0:
                result.hybrid_score = self.alpha * norm_semantic
            elif norm_keyword > 0:
                result.hybrid_score = (1 - self.alpha) * norm_keyword
            else:
                result.hybrid_score = 0.0
        
        return results
    
    def compare_search_methods(self, 
                             query: str, 
                             k: int = 5,
                             filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[SearchResult]]:
        """Compare different search methods for analysis"""
        
        # Individual searches
        semantic_results = self.vector_store_manager.search_similar(query, k=k, filters=filters)
        keyword_results = self.keyword_manager.search_keywords(query, k=k, filters=filters)
        
        # Convert to SearchResult format
        semantic_only = [SearchResult(chunk=chunk, semantic_score=score, hybrid_score=score) 
                        for chunk, score in semantic_results]
        
        keyword_only = [SearchResult(chunk=chunk, keyword_score=score, hybrid_score=score) 
                       for chunk, score in keyword_results]
        
        # Hybrid searches with different fusion methods
        hybrid_results = {}
        for method in self.fusion_methods.keys():
            hybrid_results[f"hybrid_{method}"] = self.search(
                query, k=k, filters=filters, fusion_method=method
            )
        
        return {
            'semantic_only': semantic_only,
            'keyword_only': keyword_only,
            **hybrid_results
        }
    
    def analyze_search_coverage(self, 
                               query: str, 
                               k: int = 10) -> Dict[str, Any]:
        """Analyze overlap and coverage between search methods"""
        
        # Get larger result sets for analysis
        semantic_results = self.vector_store_manager.search_similar(query, k=k*2)
        keyword_results = self.keyword_manager.search_keywords(query, k=k*2)
        
        # Extract chunk IDs
        semantic_ids = {self._get_chunk_id(chunk) for chunk, _ in semantic_results}
        keyword_ids = {self._get_chunk_id(chunk) for chunk, _ in keyword_results}
        
        # Calculate overlap metrics
        intersection = semantic_ids & keyword_ids
        union = semantic_ids | keyword_ids
        
        analysis = {
            'query': query,
            'semantic_results': len(semantic_ids),
            'keyword_results': len(keyword_ids),
            'overlap_count': len(intersection),
            'union_count': len(union),
            'overlap_percentage': len(intersection) / len(union) * 100 if union else 0,
            'semantic_unique': len(semantic_ids - keyword_ids),
            'keyword_unique': len(keyword_ids - semantic_ids),
            'jaccard_similarity': len(intersection) / len(union) if union else 0
        }
        
        return analysis

class HybridSearchManager:
    """High-level manager for hybrid search operations"""
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager = None,
                 keyword_manager: KeywordSearchManager = None,
                 alpha: float = Config.HYBRID_ALPHA):
        
        self.hybrid_engine = HybridSearchEngine(
            vector_store_manager=vector_store_manager,
            keyword_manager=keyword_manager,
            alpha=alpha
        )
    
    def index_documents(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Index documents for both semantic and keyword search"""
        logger.info("Indexing documents for hybrid search")
        
        # Index for semantic search
        semantic_stats = self.hybrid_engine.vector_store_manager.index_documents(chunks)
        
        # Index for keyword search
        keyword_stats = self.hybrid_engine.keyword_manager.index_documents(chunks)
        
        combined_stats = {
            'total_chunks': len(chunks),
            'semantic_indexing': semantic_stats,
            'keyword_indexing': keyword_stats,
            'hybrid_ready': True
        }
        
        logger.info("Hybrid search indexing completed")
        return combined_stats
    
    def search(self, 
               query: str, 
               k: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               method: str = 'weighted_sum') -> List[SearchResult]:
        """Perform hybrid search"""
        return self.hybrid_engine.search(
            query=query, 
            k=k, 
            filters=filters, 
            fusion_method=method
        )
    
    def experimental_search(self, 
                          query: str, 
                          k: int = 5) -> Dict[str, Any]:
        """Run experimental comparison of different search methods"""
        
        results = self.hybrid_engine.compare_search_methods(query, k=k)
        coverage = self.hybrid_engine.analyze_search_coverage(query, k=k)
        
        return {
            'query': query,
            'results_by_method': results,
            'coverage_analysis': coverage
        }
    
    def tune_alpha(self, 
                   queries: List[str], 
                   alpha_values: List[float] = None) -> Dict[str, Any]:
        """Experimental alpha tuning for optimal hybrid performance"""
        
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        results = {}
        original_alpha = self.hybrid_engine.alpha
        
        for alpha in alpha_values:
            self.hybrid_engine.alpha = alpha
            alpha_results = []
            
            for query in queries:
                coverage = self.hybrid_engine.analyze_search_coverage(query)
                alpha_results.append(coverage)
            
            # Calculate average metrics
            avg_overlap = np.mean([r['overlap_percentage'] for r in alpha_results])
            avg_jaccard = np.mean([r['jaccard_similarity'] for r in alpha_results])
            
            results[alpha] = {
                'average_overlap_percentage': avg_overlap,
                'average_jaccard_similarity': avg_jaccard,
                'query_results': alpha_results
            }
        
        # Restore original alpha
        self.hybrid_engine.alpha = original_alpha
        
        return {
            'alpha_analysis': results,
            'recommended_alpha': max(results.keys(), 
                                   key=lambda a: results[a]['average_jaccard_similarity'])
        }

# Example usage and testing
if __name__ == "__main__":
    import time
    from document_parser import DocumentParser
    from chunking_strategy import DocumentChunker
    
    logger.info("Testing Hybrid Search")
    
    # Initialize components
    parser = DocumentParser()
    chunker = DocumentChunker()
    hybrid_manager = HybridSearchManager()
    
    # Parse, chunk, and index documents
    sample_path = "./Documents"
    from pathlib import Path
    
    if Path(sample_path).exists():
        logger.info("Setting up hybrid search system...")
        
        # Parse and chunk
        doc_chunks = parser.parse_directory(sample_path)
        text_chunks = chunker.chunk_documents(doc_chunks)
        
        # Index for hybrid search
        indexing_stats = hybrid_manager.index_documents(text_chunks)
        print(f"Hybrid indexing stats: {indexing_stats}")
        
        # Test hybrid search
        test_queries = [
            "Master in International Finance program details",
            "career opportunities after graduation",
            "admission requirements and application process",
            "course curriculum and program structure"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"HYBRID SEARCH: '{query}'")
            print('='*60)
            
            # Standard hybrid search
            results = hybrid_manager.search(query, k=3, method='weighted_sum')
            
            print("\nTop 3 Hybrid Results:")
            for i, result in enumerate(results):
                print(f"\n  Result {i+1}:")
                print(f"    Hybrid Score: {result.hybrid_score:.3f}")
                print(f"    Semantic Score: {result.semantic_score:.3f}")
                print(f"    Keyword Score: {result.keyword_score:.3f}")
                print(f"    Source: {result.chunk.metadata.get('source', 'unknown')}")
                print(f"    Page: {result.chunk.metadata.get('page', 'unknown')}")
                print(f"    Text: {result.chunk.text[:200]}...")
            
            # Experimental comparison
            experimental = hybrid_manager.experimental_search(query, k=2)
            coverage = experimental['coverage_analysis']
            
            print(f"\n  Coverage Analysis:")
            print(f"    Semantic results: {coverage['semantic_results']}")
            print(f"    Keyword results: {coverage['keyword_results']}")
            print(f"    Overlap: {coverage['overlap_count']} ({coverage['overlap_percentage']:.1f}%)")
            print(f"    Jaccard similarity: {coverage['jaccard_similarity']:.3f}")
    
    else:
        logger.warning(f"Documents directory not found: {sample_path}") 