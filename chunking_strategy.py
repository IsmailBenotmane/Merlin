"""
Advanced chunking strategy for optimal text segmentation in RAG systems
"""
import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from document_parser import DocumentChunk
from config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary with confidence score"""
    position: int
    confidence: float
    boundary_type: str  # 'sentence', 'paragraph', 'semantic'

class SemanticChunker:
    """Advanced chunking strategy using semantic similarity and structure"""
    
    def __init__(self, 
                 chunk_size: int = Config.CHUNK_SIZE,
                 overlap: int = Config.CHUNK_OVERLAP,
                 min_chunk_size: int = Config.MIN_CHUNK_SIZE,
                 max_chunk_size: int = Config.MAX_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def chunk_document(self, document_chunk: DocumentChunk) -> List[DocumentChunk]:
        """
        Chunk a document using semantic boundaries and optimal sizing
        """
        text = document_chunk.text
        
        # If text is already small enough, return as-is
        if len(text) <= self.chunk_size:
            chunk = DocumentChunk(
                text=text,
                metadata={
                    **document_chunk.metadata,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'chunk_type': 'complete_page'
                },
                page_number=document_chunk.page_number,
                chunk_index=0
            )
            return [chunk]
        
        # Apply semantic chunking strategy
        chunks = self._semantic_chunk_text(text, document_chunk.metadata, document_chunk.page_number)
        
        # Post-process chunks to ensure optimal sizes
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        
        return optimized_chunks
    
    def _semantic_chunk_text(self, text: str, base_metadata: Dict[str, Any], page_number: Optional[int]) -> List[DocumentChunk]:
        """Apply semantic chunking to text"""
        
        # Step 1: Identify potential boundaries
        boundaries = self._identify_boundaries(text)
        
        # Step 2: Score boundaries using semantic similarity
        scored_boundaries = self._score_boundaries(text, boundaries)
        
        # Step 3: Select optimal chunk boundaries
        chunk_boundaries = self._select_chunk_boundaries(text, scored_boundaries)
        
        # Step 4: Create chunks
        chunks = self._create_chunks_from_boundaries(text, chunk_boundaries, base_metadata, page_number)
        
        return chunks
    
    def _identify_boundaries(self, text: str) -> List[ChunkBoundary]:
        """Identify potential chunk boundaries"""
        boundaries = []
        
        # Sentence boundaries
        sentences = nltk.sent_tokenize(text)
        current_pos = 0
        for sentence in sentences:
            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)
            
            boundaries.append(ChunkBoundary(
                position=sentence_end,
                confidence=0.6,
                boundary_type='sentence'
            ))
            current_pos = sentence_end
        
        # Paragraph boundaries (double newlines)
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(ChunkBoundary(
                position=match.end(),
                confidence=0.8,
                boundary_type='paragraph'
            ))
        
        # Section boundaries (headers, bullet points, etc.)
        section_patterns = [
            r'\n\d+\.\s+',  # Numbered lists
            r'\n[A-Z][^.]*:',  # Headers ending with colon
            r'\n\*\s+',  # Bullet points
            r'\n-\s+',   # Dash bullet points
            r'\n#{1,6}\s+',  # Markdown headers
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(ChunkBoundary(
                    position=match.start(),
                    confidence=0.9,
                    boundary_type='section'
                ))
        
        # Sort boundaries by position
        return sorted(boundaries, key=lambda x: x.position)
    
    def _score_boundaries(self, text: str, boundaries: List[ChunkBoundary]) -> List[ChunkBoundary]:
        """Score boundaries using semantic similarity"""
        if len(boundaries) < 2:
            return boundaries
        
        # Create text segments around each boundary
        segments = []
        for i, boundary in enumerate(boundaries):
            start = max(0, boundary.position - 100)
            end = min(len(text), boundary.position + 100)
            segments.append(text[start:end])
        
        try:
            # Calculate TF-IDF vectors for segments
            if len(segments) > 1:
                tfidf_matrix = self.vectorizer.fit_transform(segments)
                
                # Calculate similarity scores between adjacent segments
                for i in range(len(boundaries) - 1):
                    similarity = cosine_similarity(
                        tfidf_matrix[i:i+1], 
                        tfidf_matrix[i+1:i+2]
                    )[0][0]
                    
                    # Lower similarity indicates better boundary
                    semantic_score = 1.0 - similarity
                    
                    # Combine with existing confidence
                    boundaries[i].confidence = (boundaries[i].confidence + semantic_score) / 2
        
        except Exception as e:
            logger.warning(f"Error in semantic scoring: {e}")
        
        return boundaries
    
    def _select_chunk_boundaries(self, text: str, boundaries: List[ChunkBoundary]) -> List[int]:
        """Select optimal chunk boundaries based on size and confidence"""
        selected_boundaries = [0]  # Always start at the beginning
        current_pos = 0
        
        while current_pos < len(text):
            # Find boundaries within the optimal chunk size range
            target_end = current_pos + self.chunk_size
            min_end = current_pos + self.min_chunk_size
            max_end = current_pos + self.max_chunk_size
            
            # Filter boundaries within acceptable range
            candidate_boundaries = [
                b for b in boundaries 
                if min_end <= b.position <= max_end and b.position > current_pos
            ]
            
            if not candidate_boundaries:
                # No good boundaries found, force a boundary at max size
                next_boundary = min(max_end, len(text))
            else:
                # Select boundary with highest confidence closest to target
                best_boundary = max(candidate_boundaries, 
                                  key=lambda b: b.confidence - abs(b.position - target_end) / self.chunk_size)
                next_boundary = best_boundary.position
            
            selected_boundaries.append(next_boundary)
            current_pos = next_boundary
            
            if next_boundary >= len(text):
                break
        
        # Ensure we end at the text end
        if selected_boundaries[-1] < len(text):
            selected_boundaries.append(len(text))
        
        return selected_boundaries
    
    def _create_chunks_from_boundaries(self, text: str, boundaries: List[int], 
                                     base_metadata: Dict[str, Any], 
                                     page_number: Optional[int]) -> List[DocumentChunk]:
        """Create document chunks from selected boundaries"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Add overlap with previous chunk (except for first chunk)
            if i > 0:
                overlap_start = max(0, start - self.overlap)
                chunk_text = text[overlap_start:end]
            else:
                chunk_text = text[start:end]
            
            # Skip very small chunks
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk_metadata = {
                **base_metadata,
                'chunk_index': i,
                'total_chunks': len(boundaries) - 1,
                'chunk_start': start,
                'chunk_end': end,
                'chunk_size': len(chunk_text),
                'chunk_type': 'semantic'
            }
            
            chunk = DocumentChunk(
                text=chunk_text.strip(),
                metadata=chunk_metadata,
                page_number=page_number,
                chunk_index=i
            )
            chunks.append(chunk)
        
        return chunks
    
    def _optimize_chunk_sizes(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks to optimize sizes"""
        optimized_chunks = []
        
        for chunk in chunks:
            if len(chunk.text) > self.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            elif len(chunk.text) < self.min_chunk_size and optimized_chunks:
                # Merge small chunks with previous chunk
                prev_chunk = optimized_chunks[-1]
                merged_text = prev_chunk.text + "\n\n" + chunk.text
                
                if len(merged_text) <= self.max_chunk_size:
                    # Update previous chunk
                    prev_chunk.text = merged_text
                    prev_chunk.metadata['chunk_size'] = len(merged_text)
                    prev_chunk.metadata['chunk_end'] = chunk.metadata['chunk_end']
                else:
                    optimized_chunks.append(chunk)
            else:
                optimized_chunks.append(chunk)
        
        # Update chunk indices and total counts
        for i, chunk in enumerate(optimized_chunks):
            chunk.chunk_index = i
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(optimized_chunks)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Split a chunk that's too large"""
        text = chunk.text
        sub_chunks = []
        
        # Simple splitting by sentences for large chunks
        sentences = nltk.sent_tokenize(text)
        current_text = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_text + sentence) > self.chunk_size and current_text:
                # Create sub-chunk
                sub_chunk_metadata = {
                    **chunk.metadata,
                    'chunk_size': len(current_text),
                    'parent_chunk': chunk.chunk_index,
                    'chunk_type': 'split_semantic'
                }
                
                sub_chunk = DocumentChunk(
                    text=current_text.strip(),
                    metadata=sub_chunk_metadata,
                    page_number=chunk.page_number,
                    chunk_index=len(sub_chunks)
                )
                sub_chunks.append(sub_chunk)
                current_text = sentence
            else:
                current_text += " " + sentence if current_text else sentence
        
        # Add remaining text
        if current_text.strip():
            sub_chunk_metadata = {
                **chunk.metadata,
                'chunk_size': len(current_text),
                'parent_chunk': chunk.chunk_index,
                'chunk_type': 'split_semantic'
            }
            
            sub_chunk = DocumentChunk(
                text=current_text.strip(),
                metadata=sub_chunk_metadata,
                page_number=chunk.page_number,
                chunk_index=len(sub_chunks)
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]

class DocumentChunker:
    """Main document chunker that processes parsed documents"""
    
    def __init__(self, chunking_strategy: SemanticChunker = None):
        self.chunker = chunking_strategy or SemanticChunker()
    
    def chunk_documents(self, document_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Chunk a list of document chunks (typically pages)"""
        all_chunks = []
        
        for doc_chunk in document_chunks:
            try:
                chunks = self.chunker.chunk_document(doc_chunk)
                all_chunks.extend(chunks)
                logger.info(f"Chunked page {doc_chunk.page_number}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error chunking page {doc_chunk.page_number}: {e}")
                # Fallback: add original chunk
                all_chunks.append(doc_chunk)
        
        return all_chunks
    
    def analyze_chunk_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Analyze the distribution of chunk sizes for optimization"""
        sizes = [len(chunk.text) for chunk in chunks]
        
        analysis = {
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(sizes),
            'median_chunk_size': np.median(sizes),
            'min_chunk_size': np.min(sizes),
            'max_chunk_size': np.max(sizes),
            'std_chunk_size': np.std(sizes),
            'size_distribution': {
                'small_chunks': len([s for s in sizes if s < Config.MIN_CHUNK_SIZE]),
                'optimal_chunks': len([s for s in sizes if Config.MIN_CHUNK_SIZE <= s <= Config.CHUNK_SIZE]),
                'large_chunks': len([s for s in sizes if s > Config.CHUNK_SIZE]),
                'oversized_chunks': len([s for s in sizes if s > Config.MAX_CHUNK_SIZE])
            }
        }
        
        return analysis

# Example usage
if __name__ == "__main__":
    import os
    from document_parser import DocumentParser
    
    # Test chunking strategy
    parser = DocumentParser()
    chunker = DocumentChunker()
    
    # Parse and chunk documents
    sample_path = "./Documents"
    if os.path.exists(sample_path):
        doc_chunks = parser.parse_directory(sample_path)
        text_chunks = chunker.chunk_documents(doc_chunks)
        
        print(f"Created {len(text_chunks)} chunks from {len(doc_chunks)} pages")
        
        # Analyze chunk distribution
        analysis = chunker.analyze_chunk_distribution(text_chunks)
        print(f"Chunk analysis: {analysis}")
        
        # Show sample chunks
        for i, chunk in enumerate(text_chunks[:3]):
            print(f"\nChunk {i}:")
            print(f"Size: {len(chunk.text)} characters")
            print(f"Page: {chunk.page_number}")
            print(f"Text preview: {chunk.text[:200]}...")
            print(f"Metadata: {chunk.metadata}") 
