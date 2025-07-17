"""
Chunk optimization system for analyzing and optimizing chunk sizes
"""
import logging
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from document_parser import DocumentChunk
from chunking_strategy import DocumentChunker, SemanticChunker
from hybrid_search import HybridSearchManager, SearchResult
from vector_store import VectorStoreManager
from keyword_search import KeywordSearchManager
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetrics:
    """Metrics for evaluating chunk quality"""
    chunk_id: str
    size_chars: int
    size_words: int
    semantic_density: float  # Information density
    keyword_density: float   # Keyword richness
    retrieval_frequency: int # How often retrieved
    avg_relevance_score: float # Average relevance when retrieved
    document_type: str
    section: str

@dataclass
class OptimizationResult:
    """Result of chunk optimization analysis"""
    current_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    performance_improvement: float
    confidence_score: float
    analysis_details: Dict[str, Any]

class ChunkAnalyzer:
    """Analyzer for chunk characteristics and performance"""
    
    def __init__(self):
        self.chunk_metrics = {}
        self.search_history = []
        self.performance_data = defaultdict(list)
    
    def analyze_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze chunk characteristics"""
        logger.info(f"Analyzing {len(chunks)} chunks...")
        
        metrics = []
        size_stats = []
        doc_type_dist = Counter()
        section_dist = Counter()
        
        for chunk in chunks:
            # Basic size metrics
            char_count = len(chunk.text)
            word_count = len(chunk.text.split())
            size_stats.append(char_count)
            
            # Document type and section distribution
            doc_type = chunk.metadata.get('document_type', 'unknown')
            section = chunk.metadata.get('section', 'unknown')
            doc_type_dist[doc_type] += 1
            section_dist[section] += 1
            
            # Calculate information density (simple heuristic)
            semantic_density = self._calculate_semantic_density(chunk.text)
            keyword_density = self._calculate_keyword_density(chunk.text)
            
            chunk_id = f"{chunk.metadata.get('source', '')}_{chunk.page_number}_{chunk.chunk_index}"
            
            metric = ChunkMetrics(
                chunk_id=chunk_id,
                size_chars=char_count,
                size_words=word_count,
                semantic_density=semantic_density,
                keyword_density=keyword_density,
                retrieval_frequency=0,  # Will be updated during search analysis
                avg_relevance_score=0.0,
                document_type=doc_type,
                section=section
            )
            
            metrics.append(metric)
            self.chunk_metrics[chunk_id] = metric
        
        # Calculate distribution statistics
        analysis = {
            'total_chunks': len(chunks),
            'size_statistics': {
                'mean': statistics.mean(size_stats),
                'median': statistics.median(size_stats),
                'std_dev': statistics.stdev(size_stats) if len(size_stats) > 1 else 0,
                'min': min(size_stats),
                'max': max(size_stats),
                'percentiles': {
                    '25th': np.percentile(size_stats, 25),
                    '75th': np.percentile(size_stats, 75),
                    '90th': np.percentile(size_stats, 90)
                }
            },
            'document_type_distribution': dict(doc_type_dist),
            'section_distribution': dict(section_dist),
            'semantic_density': {
                'mean': statistics.mean([m.semantic_density for m in metrics]),
                'std_dev': statistics.stdev([m.semantic_density for m in metrics]) if len(metrics) > 1 else 0
            },
            'keyword_density': {
                'mean': statistics.mean([m.keyword_density for m in metrics]),
                'std_dev': statistics.stdev([m.keyword_density for m in metrics]) if len(metrics) > 1 else 0
            }
        }
        
        return analysis
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic information density (simplified heuristic)"""
        if not text:
            return 0.0
        
        # Simple heuristics for information density
        sentences = text.split('.')
        avg_sentence_length = len(text) / len(sentences) if sentences else 0
        
        # Count information-bearing words (simple heuristic)
        words = text.lower().split()
        info_words = [w for w in words if len(w) > 3 and w.isalpha()]
        unique_words = len(set(info_words))
        total_words = len(words)
        
        # Density score combining factors
        uniqueness_ratio = unique_words / total_words if total_words > 0 else 0
        length_factor = min(avg_sentence_length / 50, 1.0)  # Normalize to reasonable range
        
        return (uniqueness_ratio + length_factor) / 2
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate keyword richness"""
        if not text:
            return 0.0
        
        # Academic/business keywords that indicate important content
        important_keywords = {
            'program', 'course', 'curriculum', 'degree', 'master', 'international',
            'finance', 'career', 'employment', 'admission', 'requirement', 'application',
            'skill', 'experience', 'opportunity', 'development', 'management', 'business',
            'financial', 'analysis', 'research', 'student', 'graduate', 'professional'
        }
        
        words = text.lower().split()
        keyword_count = sum(1 for word in words if any(kw in word for kw in important_keywords))
        
        return keyword_count / len(words) if words else 0.0
    
    def record_search_result(self, query: str, results: List[SearchResult], search_type: str):
        """Record search results for analysis"""
        search_record = {
            'query': query,
            'search_type': search_type,
            'timestamp': time.time(),
            'results': []
        }
        
        for rank, result in enumerate(results):
            chunk_id = f"{result.chunk.metadata.get('source', '')}_{result.chunk.page_number}_{result.chunk.chunk_index}"
            
            result_record = {
                'chunk_id': chunk_id,
                'rank': rank + 1,
                'score': result.hybrid_score,
                'semantic_score': result.semantic_score,
                'keyword_score': result.keyword_score
            }
            
            search_record['results'].append(result_record)
            
            # Update chunk metrics
            if chunk_id in self.chunk_metrics:
                metric = self.chunk_metrics[chunk_id]
                metric.retrieval_frequency += 1
                
                # Update average relevance score
                current_avg = metric.avg_relevance_score
                new_score = result.hybrid_score
                metric.avg_relevance_score = (current_avg * (metric.retrieval_frequency - 1) + new_score) / metric.retrieval_frequency
        
        self.search_history.append(search_record)
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze retrieval performance by chunk characteristics"""
        if not self.search_history:
            return {'error': 'No search history available'}
        
        # Analyze performance by chunk size
        size_performance = defaultdict(list)
        density_performance = defaultdict(list)
        type_performance = defaultdict(list)
        
        for chunk_id, metric in self.chunk_metrics.items():
            if metric.retrieval_frequency > 0:
                # Group by size ranges
                if metric.size_chars < 500:
                    size_group = 'small'
                elif metric.size_chars < 1000:
                    size_group = 'medium'
                elif metric.size_chars < 1500:
                    size_group = 'large'
                else:
                    size_group = 'very_large'
                
                size_performance[size_group].append(metric.avg_relevance_score)
                
                # Group by density
                if metric.semantic_density < 0.3:
                    density_group = 'low_density'
                elif metric.semantic_density < 0.6:
                    density_group = 'medium_density'
                else:
                    density_group = 'high_density'
                
                density_performance[density_group].append(metric.avg_relevance_score)
                
                # Group by document type
                type_performance[metric.document_type].append(metric.avg_relevance_score)
        
        # Calculate average performance for each group
        size_avg_performance = {
            size: statistics.mean(scores) for size, scores in size_performance.items()
        }
        
        density_avg_performance = {
            density: statistics.mean(scores) for density, scores in density_performance.items()
        }
        
        type_avg_performance = {
            doc_type: statistics.mean(scores) for doc_type, scores in type_performance.items()
        }
        
        return {
            'total_searches': len(self.search_history),
            'chunks_retrieved': len([m for m in self.chunk_metrics.values() if m.retrieval_frequency > 0]),
            'size_performance': size_avg_performance,
            'density_performance': density_avg_performance,
            'type_performance': type_avg_performance,
            'most_retrieved_chunks': sorted(
                self.chunk_metrics.values(),
                key=lambda x: x.retrieval_frequency,
                reverse=True
            )[:10]
        }

class ChunkOptimizer:
    """Main optimizer for chunk size and strategy"""
    
    def __init__(self, 
                 test_queries: List[str] = None,
                 evaluation_k: int = 5):
        """
        Initialize chunk optimizer
        
        Args:
            test_queries: Queries to use for evaluation
            evaluation_k: Number of results to evaluate per query
        """
        self.test_queries = test_queries or self._get_default_test_queries()
        self.evaluation_k = evaluation_k
        self.analyzer = ChunkAnalyzer()
        
    def _get_default_test_queries(self) -> List[str]:
        """Get default test queries for evaluation"""
        return [
            "Master in International Finance program overview",
            "career opportunities after graduation",
            "admission requirements and application process",
            "course curriculum and program structure",
            "employment statistics and job placement",
            "international finance skills and competencies",
            "program duration and schedule",
            "faculty and teaching methodology",
            "financial aid and scholarships",
            "alumni network and career support"
        ]
    
    def optimize_chunk_configuration(self, 
                                   document_chunks: List[DocumentChunk]) -> OptimizationResult:
        """
        Optimize chunk configuration based on performance analysis
        
        Args:
            document_chunks: Original document chunks (pages)
            
        Returns:
            OptimizationResult with recommendations
        """
        logger.info("Starting chunk optimization analysis...")
        
        current_config = {
            'chunk_size': Config.CHUNK_SIZE,
            'chunk_overlap': Config.CHUNK_OVERLAP,
            'min_chunk_size': Config.MIN_CHUNK_SIZE,
            'max_chunk_size': Config.MAX_CHUNK_SIZE
        }
        
        # Test different configurations
        test_configs = self._generate_test_configurations(current_config)
        
        config_performance = {}
        
        for config_name, config in test_configs.items():
            logger.info(f"Testing configuration: {config_name}")
            performance = self._evaluate_configuration(document_chunks, config)
            config_performance[config_name] = performance
        
        # Find best configuration
        best_config_name = max(config_performance.keys(), 
                              key=lambda k: config_performance[k]['avg_performance'])
        
        best_config = test_configs[best_config_name]
        best_performance = config_performance[best_config_name]
        current_performance = config_performance.get('current', best_performance)
        
        # Calculate improvement
        performance_improvement = (
            best_performance['avg_performance'] - current_performance['avg_performance']
        ) / current_performance['avg_performance'] * 100
        
        # Calculate confidence based on consistency
        performance_scores = [p['avg_performance'] for p in config_performance.values()]
        performance_std = statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0
        confidence_score = max(0, 1 - (performance_std / statistics.mean(performance_scores)))
        
        result = OptimizationResult(
            current_config=current_config,
            recommended_config=best_config,
            performance_improvement=performance_improvement,
            confidence_score=confidence_score,
            analysis_details={
                'tested_configurations': config_performance,
                'best_config_name': best_config_name,
                'analyzer_data': self.analyzer.get_performance_analysis()
            }
        )
        
        logger.info(f"Optimization complete. Recommended improvement: {performance_improvement:.2f}%")
        return result
    
    def _generate_test_configurations(self, base_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate test configurations to evaluate"""
        configs = {
            'current': base_config.copy(),
        }
        
        # Test different chunk sizes
        for size in [600, 800, 1000, 1200, 1500]:
            if size != base_config['chunk_size']:
                configs[f'size_{size}'] = {
                    **base_config,
                    'chunk_size': size,
                    'max_chunk_size': max(size * 2, base_config['max_chunk_size'])
                }
        
        # Test different overlap settings
        for overlap in [100, 150, 200, 250, 300]:
            if overlap != base_config['chunk_overlap']:
                configs[f'overlap_{overlap}'] = {
                    **base_config,
                    'chunk_overlap': overlap
                }
        
        # Test combined optimizations
        configs['optimized_small'] = {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'min_chunk_size': 100,
            'max_chunk_size': 1600
        }
        
        configs['optimized_large'] = {
            'chunk_size': 1200,
            'chunk_overlap': 250,
            'min_chunk_size': 150,
            'max_chunk_size': 2400
        }
        
        return configs
    
    def _evaluate_configuration(self, 
                               document_chunks: List[DocumentChunk], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific chunking configuration"""
        
        # Create chunker with test configuration
        chunker = DocumentChunker(
            chunking_strategy=SemanticChunker(
                chunk_size=config['chunk_size'],
                overlap=config['chunk_overlap'],
                min_chunk_size=config['min_chunk_size'],
                max_chunk_size=config['max_chunk_size']
            )
        )
        
        # Chunk documents
        text_chunks = chunker.chunk_documents(document_chunks)
        
        # Quick indexing for evaluation (simplified)
        vector_manager = VectorStoreManager()
        keyword_manager = KeywordSearchManager()
        hybrid_manager = HybridSearchManager(vector_manager, keyword_manager)
        
        try:
            # Index chunks
            hybrid_manager.index_documents(text_chunks)
            
            # Evaluate with test queries
            query_performances = []
            
            for query in self.test_queries:
                try:
                    results = hybrid_manager.search(query, k=self.evaluation_k)
                    
                    # Calculate performance metrics
                    if results:
                        avg_score = statistics.mean([r.hybrid_score for r in results])
                        score_variance = statistics.variance([r.hybrid_score for r in results]) if len(results) > 1 else 0
                        
                        # Record results for analysis
                        self.analyzer.record_search_result(query, results, 'hybrid')
                        
                        query_performances.append({
                            'avg_score': avg_score,
                            'score_variance': score_variance,
                            'result_count': len(results)
                        })
                    
                except Exception as e:
                    logger.warning(f"Error evaluating query '{query}': {e}")
            
            # Calculate overall performance
            if query_performances:
                avg_performance = statistics.mean([qp['avg_score'] for qp in query_performances])
                performance_consistency = 1.0 / (1.0 + statistics.mean([qp['score_variance'] for qp in query_performances]))
            else:
                avg_performance = 0.0
                performance_consistency = 0.0
            
        except Exception as e:
            logger.error(f"Error during configuration evaluation: {e}")
            avg_performance = 0.0
            performance_consistency = 0.0
        
        # Analyze chunk distribution
        chunk_analysis = chunker.analyze_chunk_distribution(text_chunks)
        
        return {
            'avg_performance': avg_performance,
            'performance_consistency': performance_consistency,
            'chunk_distribution': chunk_analysis,
            'total_chunks': len(text_chunks),
            'config': config
        }
    
    def visualize_optimization_results(self, 
                                     optimization_result: OptimizationResult, 
                                     save_path: Optional[str] = None):
        """Create visualizations of optimization results"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Performance comparison
            configs = optimization_result.analysis_details['tested_configurations']
            config_names = list(configs.keys())
            performances = [configs[name]['avg_performance'] for name in config_names]
            
            ax1.bar(config_names, performances)
            ax1.set_title('Performance by Configuration')
            ax1.set_ylabel('Average Performance Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Chunk size distribution
            current_dist = configs['current']['chunk_distribution']
            if 'size_distribution' in current_dist:
                size_dist = current_dist['size_distribution']
                ax2.pie(size_dist.values(), labels=size_dist.keys(), autopct='%1.1f%%')
                ax2.set_title('Current Chunk Size Distribution')
            
            # Performance vs chunk count
            chunk_counts = [configs[name]['total_chunks'] for name in config_names]
            ax3.scatter(chunk_counts, performances)
            ax3.set_xlabel('Total Chunks')
            ax3.set_ylabel('Average Performance')
            ax3.set_title('Performance vs Chunk Count')
            
            # Configuration comparison
            current_config = optimization_result.current_config
            recommended_config = optimization_result.recommended_config
            
            metrics = ['chunk_size', 'chunk_overlap', 'min_chunk_size', 'max_chunk_size']
            current_values = [current_config[m] for m in metrics]
            recommended_values = [recommended_config[m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, current_values, width, label='Current', alpha=0.7)
            ax4.bar(x + width/2, recommended_values, width, label='Recommended', alpha=0.7)
            ax4.set_xlabel('Configuration Parameters')
            ax4.set_ylabel('Values')
            ax4.set_title('Current vs Recommended Configuration')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics, rotation=45)
            ax4.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

# Example usage and testing
if __name__ == "__main__":
    from document_parser import DocumentParser
    
    logger.info("Testing Chunk Optimizer")
    
    # Parse documents
    parser = DocumentParser()
    sample_path = "./Documents"
    
    from pathlib import Path
    if Path(sample_path).exists():
        doc_chunks = parser.parse_directory(sample_path)
        
        if doc_chunks:
            # Initialize optimizer
            optimizer = ChunkOptimizer()
            
            # Run optimization
            print("Running chunk optimization analysis...")
            result = optimizer.optimize_chunk_configuration(doc_chunks)
            
            print(f"\nOptimization Results:")
            print(f"Current configuration: {result.current_config}")
            print(f"Recommended configuration: {result.recommended_config}")
            print(f"Performance improvement: {result.performance_improvement:.2f}%")
            print(f"Confidence score: {result.confidence_score:.3f}")
            
            # Show analysis details
            if 'analyzer_data' in result.analysis_details:
                analyzer_data = result.analysis_details['analyzer_data']
                if 'size_performance' in analyzer_data:
                    print(f"\nPerformance by chunk size: {analyzer_data['size_performance']}")
                if 'density_performance' in analyzer_data:
                    print(f"Performance by information density: {analyzer_data['density_performance']}")
            
            # Create visualization
            try:
                optimizer.visualize_optimization_results(result, "chunk_optimization_results.png")
            except Exception as e:
                print(f"Could not create visualization: {e}")
        
        else:
            print("No documents found to analyze")
    
    else:
        print(f"Documents directory not found: {sample_path}") 