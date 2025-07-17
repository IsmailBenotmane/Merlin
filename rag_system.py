"""
Complete RAG System integrating all components
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from openai import OpenAI

from document_parser import DocumentParser, DocumentChunk
from chunking_strategy import DocumentChunker
from vector_store import VectorStoreManager
from keyword_search import KeywordSearchManager
from hybrid_search import HybridSearchManager, SearchResult
from metadata_filters import MetadataFilterBuilder, FilterPresets, ChromaDBFilterAdapter
from config import Config

logger = logging.getLogger(__name__)

class ConversationalRAG:
    """Conversational layer that generates comprehensive responses using OpenAI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.client = None
        
        if (self.api_key
                and self.api_key.strip()
                and self.api_key != "sk-your-api-key"
                and not self.api_key.startswith('your-')):
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized for conversational responses")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.info("No valid OpenAI API key provided. Conversational responses will use fallback mode.")
                
    def generate_response(self, question: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        Generate a conversational response based on search results
        
        Args:
            question: User's question
            search_results: Retrieved chunks from RAG system
            
        Returns:
            Dict with 'response', 'sources', 'has_openai_response'
        """
        if not self.client or not search_results:
            return {
                'response': None,
                'sources': self._format_sources(search_results),
                'has_openai_response': False
            }
        
        try:
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Create the prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(question, context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4.1",  # Use the cheaper but capable model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3,  # Lower temperature for more factual responses
                stream=False
            )
            
            ai_response = response.choices[0].message.content
            sources = self._format_sources(search_results)
            
            return {
                'response': ai_response,
                'sources': sources,
                'has_openai_response': True
            }
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return {
                'response': None,
                'sources': self._format_sources(search_results),
                'has_openai_response': False,
                'error': str(e)
            }
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context string from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
            chunk = result.chunk
            source = chunk.metadata.get('source', 'Unknown document')
            page = chunk.metadata.get('page', '?')
            doc_type = chunk.metadata.get('document_type', 'Document')
            
            # Clean up source name
            source_clean = source.replace('(1).pdf', '').replace('_', ' ')
            
            context_part = f"""[Source {i}: {source_clean}, Page {page}, Type: {doc_type}]
{chunk.text.strip()}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for OpenAI"""
        return """You are an expert assistant helping prospective students learn about the Master in International Finance (MIF) program. 

Your role is to:
1. Provide comprehensive, accurate answers based on the provided document excerpts
2. Synthesize information from multiple sources when relevant
3. Be conversational and helpful in your tone
4. Always cite your sources using the format [Source X]
5. If information is incomplete or unclear, acknowledge this
6. Focus on being practical and actionable in your responses

Guidelines:
- Use the exact information from the provided sources
- Don't make assumptions or add information not in the sources
- If sources conflict, mention both perspectives
- Keep responses well-structured and easy to read
- Use bullet points or numbered lists when helpful
- End with an offer to answer follow-up questions"""

    def _create_user_prompt(self, question: str, context: str) -> str:
        """Create the user prompt with question and context"""
        return f"""Based on the following excerpts from MIF program documents, please answer this question:

QUESTION: {question}

RELEVANT INFORMATION:
{context}

Please provide a comprehensive and helpful response, citing your sources appropriately."""

    def _format_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format sources for display"""
        sources = []
        
        for i, result in enumerate(search_results, 1):
            chunk = result.chunk
            source = chunk.metadata.get('source', 'Unknown document')
            page = chunk.metadata.get('page', '?')
            doc_type = chunk.metadata.get('document_type', 'Document')
            section = chunk.metadata.get('section', '')
            
            # Clean up source name
            source_display = source.replace('(1).pdf', '').replace('_', ' ').title()
            
            sources.append({
                'number': i,
                'source': source_display,
                'page': page,
                'type': doc_type,
                'section': section,
                'relevance': result.hybrid_score,
                'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        
        return sources

class RAGSystem:
    """Complete RAG system integrating all components"""
    
    def __init__(self, 
                 documents_path: str = Config.DOCUMENTS_PATH,
                 reset_indices: bool = False):
        """
        Initialize the RAG system
        
        Args:
            documents_path: Path to documents directory
            reset_indices: Whether to reset existing indices
        """
        self.documents_path = Path(documents_path)
        self.reset_indices = reset_indices
        
        # Initialize components
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.vector_manager = VectorStoreManager()
        self.keyword_manager = KeywordSearchManager()
        self.hybrid_manager = HybridSearchManager(
            vector_store_manager=self.vector_manager,
            keyword_manager=self.keyword_manager
        )
        
        # Initialize conversational AI layer
        self.conversational_rag = ConversationalRAG()
        
        # System state
        self.is_indexed = False
        self.total_chunks = 0
        self.indexing_stats = {}
        
        logger.info(f"RAG System initialized with documents path: {documents_path}")
    
    def setup_system(self) -> Dict[str, Any]:
        """Set up the complete RAG system"""
        logger.info("Setting up RAG system...")
        
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.documents_path}")
        
        setup_stats = {
            'start_time': time.time(),
            'documents_found': 0,
            'parsing_stats': {},
            'chunking_stats': {},
            'indexing_stats': {}
        }
        
        # Step 1: Parse documents
        logger.info("Step 1: Parsing documents...")
        doc_chunks = self.parser.parse_directory(str(self.documents_path))
        setup_stats['documents_found'] = len(set(chunk.metadata.get('source', '') for chunk in doc_chunks))
        setup_stats['pages_parsed'] = len(doc_chunks)
        
        if not doc_chunks:
            raise ValueError("No documents found to parse")
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        text_chunks = self.chunker.chunk_documents(doc_chunks)
        self.total_chunks = len(text_chunks)
        
        chunk_analysis = self.chunker.analyze_chunk_distribution(text_chunks)
        setup_stats['chunking_stats'] = chunk_analysis
        
        # Step 3: Index documents
        logger.info("Step 3: Indexing documents for search...")
        indexing_stats = self.hybrid_manager.index_documents(text_chunks)
        setup_stats['indexing_stats'] = indexing_stats
        self.indexing_stats = indexing_stats
        
        # Mark as indexed
        self.is_indexed = True
        
        setup_stats['end_time'] = time.time()
        setup_stats['total_time'] = setup_stats['end_time'] - setup_stats['start_time']
        
        logger.info(f"RAG system setup complete! Indexed {self.total_chunks} chunks")
        return setup_stats
    
    def search(self, 
               query: str,
               search_type: str = 'hybrid',
               k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               fusion_method: str = 'weighted_sum') -> List[SearchResult]:
        """
        Search the document collection
        
        Args:
            query: Search query
            search_type: 'semantic', 'keyword', or 'hybrid'
            k: Number of results to return
            filters: Metadata filters (can be dict or FilterGroup)
            fusion_method: For hybrid search ('weighted_sum', 'rank_fusion', etc.)
            
        Returns:
            List of SearchResult objects
        """
        if not self.is_indexed:
            raise ValueError("System not indexed. Call setup_system() first.")
        
        logger.info(f"Searching for: '{query}' (type: {search_type}, k: {k})")
        
        # Convert filters if needed
        if isinstance(filters, dict):
            # Simple dict filters - convert to ChromaDB format for vector search
            chroma_filters = filters
        elif hasattr(filters, 'build'):
            # MetadataFilterBuilder - convert to ChromaDB format
            filter_group = filters.build()
            chroma_filters = ChromaDBFilterAdapter.to_chroma_where(filter_group)
        else:
            chroma_filters = filters
        
        start_time = time.time()
        
        if search_type == 'semantic':
            # Semantic search only
            semantic_results = self.vector_manager.search_similar(
                query, k=k, filters=chroma_filters
            )
            results = [SearchResult(chunk=chunk, semantic_score=score, hybrid_score=score) 
                      for chunk, score in semantic_results]
            
        elif search_type == 'keyword':
            # Keyword search only
            keyword_results = self.keyword_manager.search_keywords(
                query, k=k, filters=chroma_filters
            )
            results = [SearchResult(chunk=chunk, keyword_score=score, hybrid_score=score) 
                      for chunk, score in keyword_results]
            
        elif search_type == 'hybrid':
            # Hybrid search
            results = self.hybrid_manager.search(
                query, k=k, filters=chroma_filters, method=fusion_method
            )
            
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Use 'semantic', 'keyword', or 'hybrid'")
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s, returned {len(results)} results")
        
        return results
    
    def search_with_preset_filter(self, 
                                 query: str, 
                                 filter_preset: str,
                                 search_type: str = 'hybrid',
                                 k: int = 5) -> List[SearchResult]:
        """
        Search with predefined filter presets
        
        Args:
            query: Search query
            filter_preset: Name of filter preset (see FilterPresets class)
            search_type: Type of search to perform
            k: Number of results
        """
        # Get filter from presets
        preset_map = {
            'academic': FilterPresets.academic_documents(),
            'career': FilterPresets.career_documents(),
            'presentation': FilterPresets.presentation_slides(),
            'recent_pages': FilterPresets.recent_pages(5),
            'large_chunks': FilterPresets.large_chunks(800)
        }
        
        if filter_preset not in preset_map:
            raise ValueError(f"Unknown filter preset: {filter_preset}. Available: {list(preset_map.keys())}")
        
        filter_builder = preset_map[filter_preset]
        filter_group = filter_builder.build()
        chroma_filters = ChromaDBFilterAdapter.to_chroma_where(filter_group)
        
        return self.search(query, search_type=search_type, k=k, filters=chroma_filters)
    
    def compare_search_methods(self, 
                              query: str, 
                              k: int = 3) -> Dict[str, List[SearchResult]]:
        """Compare different search methods for the same query"""
        comparison = {}
        
        for search_type in ['semantic', 'keyword', 'hybrid']:
            try:
                results = self.search(query, search_type=search_type, k=k)
                comparison[search_type] = results
            except Exception as e:
                logger.error(f"Error in {search_type} search: {e}")
                comparison[search_type] = []
        
        return comparison
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query across different search methods"""
        if not self.is_indexed:
            raise ValueError("System not indexed. Call setup_system() first.")
        
        analysis = {
            'query': query,
            'timestamp': time.time(),
            'query_length': len(query),
            'word_count': len(query.split())
        }
        
        # Keyword analysis
        if hasattr(self.keyword_manager, 'analyze_query_terms'):
            try:
                keyword_analysis = self.keyword_manager.analyze_query_terms(query)
                analysis['keyword_analysis'] = keyword_analysis
            except Exception as e:
                logger.warning(f"Keyword analysis failed: {e}")
        
        # Search comparison
        try:
            comparison = self.compare_search_methods(query, k=3)
            analysis['search_comparison'] = comparison
            
            # Calculate overlap
            semantic_ids = {self._get_result_id(r) for r in comparison.get('semantic', [])}
            keyword_ids = {self._get_result_id(r) for r in comparison.get('keyword', [])}
            hybrid_ids = {self._get_result_id(r) for r in comparison.get('hybrid', [])}
            
            analysis['result_overlap'] = {
                'semantic_keyword': len(semantic_ids & keyword_ids),
                'semantic_hybrid': len(semantic_ids & hybrid_ids),
                'keyword_hybrid': len(keyword_ids & hybrid_ids),
                'total_unique': len(semantic_ids | keyword_ids | hybrid_ids)
            }
            
        except Exception as e:
            logger.error(f"Search analysis failed: {e}")
        
        return analysis
    
    def _get_result_id(self, result: SearchResult) -> str:
        """Generate ID for a search result"""
        chunk = result.chunk
        return f"{chunk.metadata.get('source', '')}_{chunk.page_number}_{chunk.chunk_index}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        status = {
            'is_indexed': self.is_indexed,
            'total_chunks': self.total_chunks,
            'documents_path': str(self.documents_path),
            'config': {
                'chunk_size': Config.CHUNK_SIZE,
                'chunk_overlap': Config.CHUNK_OVERLAP,
                'embedding_model': Config.EMBEDDING_MODEL,
                'hybrid_alpha': Config.HYBRID_ALPHA
            }
        }
        
        if self.is_indexed and self.indexing_stats:
            status['indexing_stats'] = self.indexing_stats
            
            # Get vector store stats
            try:
                vector_stats = self.vector_manager.vector_store.get_collection_stats()
                status['vector_store_stats'] = vector_stats
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {e}")
        
        return status
    
    def save_system_state(self, filepath: str) -> None:
        """Save system configuration and state"""
        state = {
            'documents_path': str(self.documents_path),
            'is_indexed': self.is_indexed,
            'total_chunks': self.total_chunks,
            'indexing_stats': self.indexing_stats,
            'config': {
                'chunk_size': Config.CHUNK_SIZE,
                'chunk_overlap': Config.CHUNK_OVERLAP,
                'embedding_model': Config.EMBEDDING_MODEL,
                'hybrid_alpha': Config.HYBRID_ALPHA
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"System state saved to {filepath}")

class RAGQueryInterface:
    """Simple command-line interface for the RAG system"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self.conversational_rag = rag_system.conversational_rag
        self.session_history = []
    
    def interactive_query(self):
        """Start interactive query session"""
        print("\n" + "="*60)
        print("🎓 MIF RAG System - Ask me anything about the MIF program!")
        print("="*60)
        print(f"📚 I have access to information from 5 MIF documents")
        print(f"💡 Ready to answer your questions! ({self.rag.total_chunks} sections indexed)")
        print("\n🔮 Examples of what you can ask:")
        print("   • What are the admission requirements for MIF?")
        print("   • How long is the program and what does it cost?")
        print("   • What career opportunities are available after graduation?")
        print("   • What courses are included in the curriculum?")
        print("   • What is the employment rate and salary expectations?")
        print("   • Who are the faculty members teaching in the program?")
        print("\n📝 Just type your question naturally! (or 'help' for advanced options, 'quit' to exit)")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if not user_input:
                    print("💬 Please ask me something about the MIF program!")
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using the MIF RAG System! Goodbye!")
                    break
                
                # Check for special commands
                if user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'history':
                    self._show_history()
                # Check for advanced commands (hidden from main interface)
                elif any(user_input.lower().startswith(cmd) for cmd in ['semantic:', 'keyword:', 'filter:', 'analyze:']):
                    self._process_advanced_command(user_input)
                else:
                    # Default: treat as natural question
                    self._answer_question(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Thank you for using the MIF RAG System! Goodbye!")
                break
            except Exception as e:
                print(f"❌ I encountered an error: {e}")
                print("💡 Please try rephrasing your question or type 'help' for assistance")
                logger.error(f"Query interface error: {e}")
    
    def _answer_question(self, question: str):
        """Answer a natural question using conversational AI with retrieved information"""
        print(f"\n🔍 Searching for information about: '{question}'")
        
        start_time = time.time()
        # Use hybrid search to get relevant chunks
        results = self.rag.search(question, search_type='hybrid', k=5)
        search_time = time.time() - start_time
        
        # Log the query
        self.session_history.append({
            'query': question,
            'search_type': 'conversational',
            'results_count': len(results),
            'search_time': search_time,
            'timestamp': time.time()
        })
        
        if not results:
            print("❌ I couldn't find specific information about that topic in the MIF documents.")
            print("💡 Try rephrasing your question or asking about:")
            print("   • Admission requirements • Program structure • Career outcomes")
            print("   • Course curriculum • Faculty • Application process")
            return
        
        print(f"✅ Found relevant information! ({search_time:.2f}s)")
        
        # Generate conversational response using OpenAI
        print("\n🤖 Generating comprehensive response...")
        response_start = time.time()
        ai_result = self.conversational_rag.generate_response(question, results)
        response_time = time.time() - response_start
        
        if ai_result['has_openai_response'] and ai_result['response']:
            # Display AI-generated response
            print(f"\n💬 **Response** (generated in {response_time:.2f}s):")
            print("="*60)
            print(ai_result['response'])
            print("="*60)
            
            # Show sources used
            print(f"\n📚 **Sources** ({len(ai_result['sources'])} documents):")
            for source in ai_result['sources'][:3]:  # Show top 3 sources
                print(f"   📄 Source {source['number']}: {source['source']} (Page {source['page']})")
                print(f"      🎯 Relevance: {source['relevance']:.1%}")
                if source['section'] and source['section'] != 'unknown':
                    print(f"      📍 Section: {source['section']}")
                print()
            
        else:
            # Fallback to showing raw chunks if OpenAI fails
            error_msg = ai_result.get('error', 'OpenAI API not available')
            print(f"\n⚠️  AI response unavailable ({error_msg}). Showing source information:")
            
            # Format results in a user-friendly way  
            for i, result in enumerate(results[:3], 1):
                chunk = result.chunk
                source = chunk.metadata.get('source', 'Unknown document')
                page = chunk.metadata.get('page', '?')
                section = chunk.metadata.get('section', '')
                
                # Make source name more readable
                source_display = source.replace('(1).pdf', '').replace('_', ' ').title()
                
                print(f"📄 **Source {i}**: {source_display}")
                if section and section != 'unknown':
                    print(f"📍 **Section**: {section}")
                print(f"📖 **Page**: {page}")
                print(f"🎯 **Relevance**: {result.hybrid_score:.1%}")
                print(f"💬 **Content**:")
                
                # Format content nicely
                content = chunk.text.strip()
                if len(content) > 400:
                    content = content[:400] + "..."
                
                # Indent content for better readability
                for line in content.split('\n'):
                    if line.strip():
                        print(f"   {line.strip()}")
                
                print()  # Add space between results
        
        # Add helpful follow-up suggestions
        print("💡 **Need more details?** You can ask follow-up questions like:")
        if "admission" in question.lower():
            print("   • 'What documents do I need for application?'")
            print("   • 'What are the language requirements?'")
        elif "career" in question.lower() or "job" in question.lower():
            print("   • 'What is the average salary after graduation?'")
            print("   • 'Which companies hire MIF graduates?'")
        elif "course" in question.lower() or "curriculum" in question.lower():
            print("   • 'What are the core subjects?'")
            print("   • 'Are there any elective courses?'")
        else:
            print("   • 'Tell me more about [specific topic]'")
            print("   • 'What are the requirements for [specific aspect]?'")
    
    def _process_advanced_command(self, command: str):
        """Process advanced commands for power users"""
        if command.startswith('semantic:'):
            query = command[9:].strip()
            self._execute_search(query, 'semantic')
        elif command.startswith('keyword:'):
            query = command[8:].strip()
            self._execute_search(query, 'keyword')
        elif command.startswith('filter:'):
            parts = command[7:].strip().split(' ', 1)
            if len(parts) == 2:
                filter_preset, query = parts
                self._execute_filtered_search(query, filter_preset)
            else:
                print("❌ Usage: filter:<preset> <question>")
                print("Available presets: academic, career, presentation")
        elif command.startswith('analyze:'):
            query = command[8:].strip()
            self._analyze_query(query)
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("📚 **MIF RAG System Help**")
        print("="*50)
        print("🎯 **Main Usage**: Just ask your question naturally!")
        print("   Example: 'What are the admission requirements?'")
        print()
        print("🔧 **Commands**:")
        print("   help     - Show this help")
        print("   status   - Show system information")
        print("   history  - Show your recent questions")
        print("   quit     - Exit the system")
        print()
        print("⚡ **Advanced (for power users)**:")
        print("   semantic:<question>  - Use semantic search only")
        print("   keyword:<question>   - Use keyword search only")
        print("   filter:<preset> <question> - Search specific document types")
        print("     • academic, career, presentation presets available")
        print("   analyze:<question>   - Compare different search methods")
        print()
        print("💡 **Tips**:")
        print("   • Ask specific questions for better results")
        print("   • Use natural language - no special commands needed")
        print("   • Follow up with related questions to dive deeper")
        print("="*50)
    
    def _execute_search(self, query: str, search_type: str = 'hybrid'):
        """Execute a search and display results"""
        print(f"\nSearching ({search_type}): '{query}'")
        print("-" * 50)
        
        start_time = time.time()
        results = self.rag.search(query, search_type=search_type, k=5)
        search_time = time.time() - start_time
        
        self.session_history.append({
            'query': query,
            'search_type': search_type,
            'results_count': len(results),
            'search_time': search_time,
            'timestamp': time.time()
        })
        
        if not results:
            print("No results found.")
            return
        
        print(f"Found {len(results)} results in {search_time:.3f}s\n")
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            print(f"Result {i}:")
            print(f"  Hybrid Score: {result.hybrid_score:.3f}")
            if result.semantic_score > 0:
                print(f"  Semantic Score: {result.semantic_score:.3f}")
            if result.keyword_score > 0:
                print(f"  Keyword Score: {result.keyword_score:.3f}")
            print(f"  Source: {chunk.metadata.get('source', 'unknown')}")
            print(f"  Page: {chunk.metadata.get('page', '?')}")
            print(f"  Type: {chunk.metadata.get('document_type', 'unknown')}")
            print(f"  Section: {chunk.metadata.get('section', 'unknown')}")
            print(f"  Text: {chunk.text[:300]}...")
            print()
    
    def _execute_filtered_search(self, query: str, filter_preset: str):
        """Execute a search with filter preset"""
        print(f"\nFiltered Search ({filter_preset}): '{query}'")
        print("-" * 50)
        
        try:
            results = self.rag.search_with_preset_filter(query, filter_preset, k=5)
            
            if not results:
                print(f"No results found with filter '{filter_preset}'.")
                return
            
            print(f"Found {len(results)} results with filter '{filter_preset}'\n")
            
            for i, result in enumerate(results, 1):
                chunk = result.chunk
                print(f"Result {i}:")
                print(f"  Score: {result.hybrid_score:.3f}")
                print(f"  Source: {chunk.metadata.get('source', 'unknown')}")
                print(f"  Page: {chunk.metadata.get('page', '?')}")
                print(f"  Type: {chunk.metadata.get('document_type', 'unknown')}")
                print(f"  Text: {chunk.text[:200]}...")
                print()
                
        except ValueError as e:
            print(f"Filter error: {e}")
    
    def _analyze_query(self, query: str):
        """Analyze a query across different methods"""
        print(f"\nAnalyzing: '{query}'")
        print("-" * 50)
        
        analysis = self.rag.analyze_query(query)
        
        print(f"Query length: {analysis['query_length']} characters")
        print(f"Word count: {analysis['word_count']} words")
        
        if 'keyword_analysis' in analysis:
            ka = analysis['keyword_analysis']
            print(f"Preprocessed terms: {ka.get('preprocessed_terms', [])}")
            print(f"Term frequencies: {ka.get('term_frequencies', {})}")
        
        if 'result_overlap' in analysis:
            overlap = analysis['result_overlap']
            print(f"Result overlap - Semantic∩Keyword: {overlap['semantic_keyword']}")
            print(f"Total unique results: {overlap['total_unique']}")
        
        if 'search_comparison' in analysis:
            comparison = analysis['search_comparison']
            for method, results in comparison.items():
                print(f"\n{method.capitalize()} search: {len(results)} results")
                if results:
                    top_result = results[0]
                    print(f"  Top result: {top_result.chunk.metadata.get('source', 'unknown')} "
                          f"(score: {top_result.hybrid_score:.3f})")
    
    def _show_status(self):
        """Show system status"""
        status = self.rag.get_system_status()
        
        print("\n" + "="*40)
        print("📊 **System Status**")
        print("="*40)
        print(f"🟢 **Status**: {'Ready' if status['is_indexed'] else 'Not Ready'}")
        print(f"📚 **Documents**: {status['total_chunks']} sections indexed")
        print(f"📁 **Source**: {status['documents_path']}")
        print(f"🤖 **AI Model**: {status['config']['embedding_model']}")
        print(f"📏 **Chunk Size**: {status['config']['chunk_size']} characters")
        print(f"⚖️ **Search Balance**: {status['config']['hybrid_alpha']:.1%} semantic")
        
        if 'vector_store_stats' in status:
            vs_stats = status['vector_store_stats']
            doc_types = vs_stats.get('document_types', {})
            if doc_types:
                print(f"📋 **Document Types**:")
                for doc_type, count in doc_types.items():
                    print(f"   • {doc_type.title()}: {count}")
        print("="*40)

    def _show_history(self):
        """Show query history"""
        if not self.session_history:
            print("\n📝 No questions asked yet in this session.")
            print("💡 Try asking something like 'What are the admission requirements?'")
            return
        
        print("\n" + "="*50)
        print("📝 **Your Recent Questions**")
        print("="*50)
        
        for i, entry in enumerate(self.session_history[-5:], 1):  # Show last 5
            results_emoji = "✅" if entry['results_count'] > 0 else "❌"
            print(f"{i}. {results_emoji} '{entry['query']}'")
            print(f"   └─ {entry['results_count']} results in {entry['search_time']:.2f}s")
            print()
        
        if len(self.session_history) > 5:
            print(f"... and {len(self.session_history) - 5} more questions")
        print("="*50)

# Example usage and main function
def main():
    """Main function to demonstrate the RAG system"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    rag = RAGSystem(documents_path="./Documents")
    
    try:
        # Setup the system
        print("Setting up RAG system...")
        setup_stats = rag.setup_system()
        
        print("\nSetup completed!")
        print(f"Documents found: {setup_stats['documents_found']}")
        print(f"Pages parsed: {setup_stats['pages_parsed']}")
        print(f"Total chunks: {setup_stats['chunking_stats']['total_chunks']}")
        print(f"Setup time: {setup_stats['total_time']:.2f}s")
        
        # Start interactive interface
        interface = RAGQueryInterface(rag)
        interface.interactive_query()
        
    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        logger.error(f"RAG system error: {e}")

if __name__ == "__main__":
    main() 