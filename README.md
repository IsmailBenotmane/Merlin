# Advanced Conversational RAG System for MiF Documents

This system is a Retrieval-Augmented Generation (RAG) platform for Master in International Finance (MiF) documents. It combines document parsing, semantic chunking, hybrid search, and metadata filtering with a conversational AI layer powered by OpenAI GPT models. Users can ask natural questions, and the system synthesizes answers from multiple documents, always citing sources.

---

## System Overview

- Conversational AI layer: Uses OpenAI GPT models (e.g., GPT-4o-mini) to synthesize and reason over retrieved document chunks, providing answers with citations.
- Natural language interface: Users interact using plain English questions; no special commands are required.
- Source citation: Every answer cites the original document(s) and page(s).
- Fallback: If the OpenAI API is unavailable, the system displays the most relevant raw document chunks.
- Scalable: Designed to handle hundreds or thousands of documents efficiently, with further optimizations available for large-scale deployments.

---

# Advanced RAG System for MiF Documents

A Retrieval-Augmented Generation (RAG) system for Master in International Finance (MiF) documents, featuring document parsing, semantic chunking, hybrid search, and metadata filtering.

## Features

### Core Capabilities
- Conversational answer synthesis using OpenAI GPT models for multi-document answers with citations
- PDF parsing using PyMuPDF for text extraction with structure preservation
- Semantic chunking with size boundaries based on content structure and similarity
- Hybrid search combining semantic and keyword search with configurable fusion methods
- Metadata filtering with query builders and preset filters
- Chunk optimization for retrieval performance
- Interactive query interface for testing and exploration

### Advanced Components
- Vector store using ChromaDB with text-embedding-3-small embeddings
- Keyword search using BM25 with phrase boosting and preprocessing
- Multiple fusion methods including weighted sum, reciprocal rank fusion, harmonic/geometric mean
- Filter presets for academic documents, career information, presentations, and more
- Performance analytics with chunk analysis and optimization recommendations

## 📋 System Requirements

### Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **Document Processing**: `pymupdf`, `pypdf`, `python-docx`, `nltk`
- **Embeddings & Search**: `chromadb`, `sentence-transformers`, `openai`, `rank-bm25`
- **Analysis**: `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- **Utilities**: `langchain`, `langchain-community`, `tqdm`

### Optional Setup
Create a `.env` file for OpenAI API key (for text-embedding-3-small):
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

Run the system and enter questions in natural language:

```bash
python rag_system.py
```

Example questions:
- What are the admission requirements?
- How long is the MIF program?
- What career opportunities are available?
- What courses are offered in the curriculum?
- What is the employment rate after graduation?

How it works:
- The system retrieves the most relevant information from all indexed MiF documents.
- The conversational AI layer synthesizes an answer, citing all sources (document name and page number).
- If the OpenAI API is unavailable, the system shows the most relevant raw document chunks.

Advanced options:
- `help` — Show all available options
- `status` — View system information
- `history` — See recent questions
- `semantic:<question>` — Use semantic search only
- `keyword:<question>` — Use keyword search only
- `filter:<preset> <question>` — Search specific document types

---

## Performance and Scaling

- The system is designed to handle up to thousands of documents with minimal latency.
- For very large document sets, further optimizations such as distributed vector stores, caching, and sharding are possible.
- Chunking is optimized for efficient retrieval and answer synthesis.

---

## 🔧 System Architecture

### Core Components

```
📁 RAG System
├── 📄 document_parser.py      # Advanced PDF parsing with PyMuPDF
├── 📄 chunking_strategy.py    # Semantic chunking with optimization
├── 📄 vector_store.py         # ChromaDB + embeddings management
├── 📄 keyword_search.py       # BM25 search with preprocessing
├── 📄 hybrid_search.py        # Multiple fusion methods
├── 📄 metadata_filters.py     # Advanced filtering system
├── 📄 chunk_optimizer.py      # Performance analysis & optimization
├── 📄 rag_system.py          # Main system integration
└── 📄 config.py              # Configuration management
```

### Document Processing Pipeline

```
PDF Documents → Parse (PyMuPDF) → Semantic Chunking → Dual Indexing
                     ↓                    ↓              ↓
               Structure Preserved → Optimal Chunks → Vector Store + BM25
                                         ↓
                                   Metadata Extraction
                                         ↓
                                   Ready for Search
```

## 🔍 Search Methods

### 1. Semantic Search
Uses text-embedding-3-small (or local fallback) for meaning-based retrieval:
```python
results = rag.search("career opportunities", search_type='semantic', k=5)
```

### 2. Keyword Search
BM25-based search with phrase boosting and preprocessing:
```python
results = rag.search("international finance program", search_type='keyword', k=5)
```

### 3. Hybrid Search (Recommended)
Combines semantic and keyword search with multiple fusion methods:
```python
# Weighted sum fusion (default)
results = rag.search("admission requirements", search_type='hybrid', k=5)

# Reciprocal rank fusion
results = rag.search("course curriculum", 
                    search_type='hybrid', 
                    fusion_method='rank_fusion', k=5)
```

## 🎯 Metadata Filtering

### Simple Filters
```python
# Filter by document type
filters = {"document_type": "brochure"}
results = rag.search("program overview", filters=filters)

# Filter by page range
filters = {"page": {"$lte": 10}}
results = rag.search("introduction", filters=filters)
```

### Advanced Filter Builder
```python
from metadata_filters import MetadataFilterBuilder

# Complex filter
filter_builder = (MetadataFilterBuilder()
    .equals("document_type", "program_info")
    .range_filter("page", 1, 10)
    .contains("section", "curriculum", case_sensitive=False))

results = rag.search("course details", filters=filter_builder)
```

### Preset Filters
```python
# Academic documents
results = rag.search_with_preset_filter("program structure", "academic")

# Career-related content
results = rag.search_with_preset_filter("job opportunities", "career")

# Presentation slides
results = rag.search_with_preset_filter("overview", "presentation")
```

## 📊 Performance Optimization

### Chunk Analysis
```python
from chunk_optimizer import ChunkOptimizer

# Initialize optimizer
optimizer = ChunkOptimizer()

# Run optimization analysis
doc_chunks = rag.parser.parse_directory("./Documents")
result = optimizer.optimize_chunk_configuration(doc_chunks)

print(f"Recommended improvement: {result.performance_improvement:.2f}%")
print(f"Recommended config: {result.recommended_config}")
```

### Performance Comparison
```python
# Compare search methods
comparison = rag.compare_search_methods("Master in International Finance")

for method, results in comparison.items():
    print(f"{method}: {len(results)} results")
    if results:
        print(f"  Top score: {results[0].hybrid_score:.3f}")
```

## 🎨 Interactive Interface

The system provides a user-friendly question-answering interface. Simply ask your questions naturally:

**Example Natural Questions:**
```
❓ Your question: What are the admission requirements?
❓ Your question: How long is the program?
❓ Your question: What career opportunities are available?
❓ Your question: What courses are offered?
❓ Your question: What is the employment rate?
❓ Your question: help                     # Show advanced options
❓ Your question: quit                     # Exit system
```

**Advanced Options** (for power users):
```
❓ Your question: semantic:program structure            # Semantic search only
❓ Your question: keyword:international finance         # Keyword search only
❓ Your question: filter:academic courses offered       # Search with document filter
❓ Your question: analyze:curriculum structure          # Compare search methods
❓ Your question: status                               # Show system status
❓ Your question: history                              # Show query history
```

The system automatically uses hybrid search (combining semantic and keyword approaches) to provide the best results from all MIF documents.

## 📈 Configuration Options

### Chunking Configuration
```python
# In config.py
CHUNK_SIZE = 1000          # Target chunk size in characters
CHUNK_OVERLAP = 200        # Overlap between chunks
MIN_CHUNK_SIZE = 100       # Minimum chunk size
MAX_CHUNK_SIZE = 2000      # Maximum chunk size
```

### Search Configuration
```python
SEMANTIC_SEARCH_K = 5      # Number of semantic results
KEYWORD_SEARCH_K = 5       # Number of keyword results  
HYBRID_ALPHA = 0.7         # Weight for semantic search (0.0-1.0)
```

### Embedding Configuration
```python
EMBEDDING_MODEL = "text-embedding-3-small"    # OpenAI model
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Local fallback
```

## 🔧 Customization

### Adding New Document Types
```python
# In document_parser.py, update _classify_document_type method
def _classify_document_type(self, pdf_path: str, metadata: Dict) -> str:
    filename = Path(pdf_path).stem.lower()
    
    if 'syllabus' in filename:
        return 'syllabus'
    elif 'handbook' in filename:
        return 'handbook'
    # ... existing classifications
```

### Custom Filter Presets
```python
# In metadata_filters.py, add to FilterPresets class
@staticmethod
def custom_filter() -> MetadataFilterBuilder:
    return (MetadataFilterBuilder()
        .equals("document_type", "custom_type")
        .greater_than("page", 5))
```

### Custom Fusion Methods
```python
# In hybrid_search.py, add to HybridSearchEngine class
def _custom_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
    # Custom fusion logic
    for result in results:
        result.hybrid_score = custom_combination_logic(
            result.semantic_score, 
            result.keyword_score
        )
    return results
```

## 📊 System Monitoring

### Performance Metrics
```python
# Get system status
status = rag.get_system_status()
print(f"Total chunks: {status['total_chunks']}")
print(f"Vector store stats: {status['vector_store_stats']}")

# Analyze query performance
analysis = rag.analyze_query("program curriculum")
print(f"Result overlap: {analysis['result_overlap']}")
```

### Chunk Distribution Analysis
```python
from chunking_strategy import DocumentChunker

chunker = DocumentChunker()
chunks = chunker.chunk_documents(doc_chunks)
analysis = chunker.analyze_chunk_distribution(chunks)

print(f"Average chunk size: {analysis['avg_chunk_size']}")
print(f"Size distribution: {analysis['size_distribution']}")
```

## 🚨 Troubleshooting

### Common Issues

1. **No OpenAI API Key**: System automatically falls back to local embeddings
2. **Memory Issues**: Reduce batch sizes in `vector_store.py`
3. **Slow Indexing**: Check if NLTK data is downloaded properly
4. **Poor Search Results**: Try chunk optimization or adjust hybrid alpha

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for all components
```

### Performance Tuning
```python
# Optimize for speed vs. quality
Config.CHUNK_SIZE = 800           # Smaller chunks = faster indexing
Config.SEMANTIC_SEARCH_K = 3      # Fewer results = faster search
Config.HYBRID_ALPHA = 0.5         # Balanced semantic/keyword weight
```

## 📄 Document Types Supported

The system currently handles these MiF document types:
- **Program Information** (`program_info`): Student guides, program overviews
- **Course Lists** (`course_list`): Curriculum details, course descriptions  
- **Brochures** (`brochure`): Marketing materials, program highlights
- **Employment Data** (`employment_data`): Career statistics, job placement info
- **Presentations** (`presentation`): Slide decks, presentation materials

## 🔮 Future Enhancements

- **Multi-language Support**: Extend to French/other languages
- **Advanced Analytics**: More sophisticated performance metrics
- **Query Expansion**: Automatic query enhancement
- **Caching Layer**: Improve response times for repeated queries
- **API Interface**: REST API for external integrations

## 📝 License

This project is designed for educational and research purposes in the context of Master in International Finance programs.

## 🤝 Contributing

To extend the system:
1. Follow the existing architecture patterns
2. Add comprehensive logging
3. Include unit tests for new components
4. Update configuration options as needed
5. Document new features in this README

---

**Created for Master in International Finance (MiF) Document Analysis** 