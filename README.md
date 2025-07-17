# Merlin Retrieval-Augmented Generation System

Merlin is a retrieval-augmented generation (RAG) framework focused on Master in International Finance (MiF) documents. It parses PDFs, produces searchable text chunks and provides both semantic and keyword search with optional conversational answers via the OpenAI API.

## Repository Overview

- **document_parser.py** – extracts text from PDFs and tags each chunk with metadata.
- **chunking_strategy.py** – splits pages into semantically meaningful segments.
- **vector_store.py** – manages the ChromaDB vector store and generates embeddings.
- **keyword_search.py** – performs BM25 keyword search with text preprocessing.
- **hybrid_search.py** – combines semantic and keyword scores using several fusion methods.
- **metadata_filters.py** – offers a builder pattern for constructing metadata filters.
- **chunk_optimizer.py** – analyses chunk statistics and suggests configuration tweaks.
- **rag_system.py** – wires all components together and exposes an interactive CLI.
- **config.py** – central configuration of chunk sizes, search parameters and models.

## Installation

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in the environment or in a `.env` file to enable OpenAI embeddings and conversational mode. Without a key the system falls back to local embeddings and displays raw search results.

## Running the System

Place your PDF documents in `./Documents` and launch:

```bash
python rag_system.py
```

The CLI indexes the documents and accepts natural language questions. Use `help`, `status` or `history` for commands. Prefix a query with `semantic:` or `keyword:` to restrict the search type, or use `filter:<preset>` to apply predefined metadata filters.

## Configuration

All tuning parameters reside in `config.py`:

- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE`
- `SEMANTIC_SEARCH_K`, `KEYWORD_SEARCH_K`, `HYBRID_ALPHA`
- `EMBEDDING_MODEL`, `LOCAL_EMBEDDING_MODEL`
- `CHROMA_PERSIST_DIRECTORY` for vector store persistence

## Programmatic Example

```python
from rag_system import RAGSystem

rag = RAGSystem(documents_path="./Documents")
setup_stats = rag.setup_system()

results = rag.search("admission requirements", search_type="hybrid", k=5)
for r in results:
    print(r.chunk.metadata.get("source"), r.chunk.metadata.get("page"))
```

## Document Types

The parser recognises program information, course lists, brochures, employment data and presentation slides. Additional types can be added by extending `document_parser.py`.

## License

This project is for educational and research use within Master in International Finance programs.
