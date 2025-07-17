"""
Advanced document parser for RAG system with optimal text extraction
"""
import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with associated metadata"""
    text: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None

class AdvancedPDFParser:
    """Advanced PDF parser using PyMuPDF for optimal text extraction"""
    
    def __init__(self):
        self.text_cleaning_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'\n\s*\n', '\n\n'),  # Multiple newlines to double newline
            (r'[^\S\n]+', ' '),  # Non-newline whitespace to single space
        ]
    
    def extract_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata from PDF"""
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        # Enhanced metadata extraction
        doc_info = {
            'source': os.path.basename(pdf_path),
            'full_path': pdf_path,
            'title': metadata.get('title', '') or self._infer_title_from_filename(pdf_path),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'created': metadata.get('creationDate', ''),
            'modified': metadata.get('modDate', ''),
            'pages': doc.page_count,
            'document_type': self._classify_document_type(pdf_path, metadata),
        }
        
        doc.close()
        return doc_info
    
    def _infer_title_from_filename(self, pdf_path: str) -> str:
        """Infer document title from filename"""
        filename = Path(pdf_path).stem
        # Remove common suffixes like (1), _v1, etc.
        filename = re.sub(r'\s*\([^)]*\)\s*$', '', filename)
        filename = re.sub(r'_v?\d+$', '', filename)
        return filename.replace('_', ' ').replace('-', ' ').strip()
    
    def _classify_document_type(self, pdf_path: str, metadata: Dict) -> str:
        """Classify document type based on filename and metadata"""
        filename = Path(pdf_path).stem.lower()
        title = metadata.get('title', '').lower()
        
        if any(keyword in filename or keyword in title for keyword in ['slide', 'presentation', 'pres']):
            return 'presentation'
        elif any(keyword in filename or keyword in title for keyword in ['brochure', 'catalog']):
            return 'brochure'
        elif any(keyword in filename or keyword in title for keyword in ['employment', 'job', 'career']):
            return 'employment_data'
        elif any(keyword in filename or keyword in title for keyword in ['course', 'cours', 'curriculum']):
            return 'course_list'
        elif any(keyword in filename or keyword in title for keyword in ['student', 'master', 'program']):
            return 'program_info'
        else:
            return 'document'
    
    def extract_text_with_structure(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract text with preserved structure and enhanced metadata"""
        doc = fitz.open(pdf_path)
        chunks = []
        doc_metadata = self.extract_document_metadata(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Extract text blocks with position information
            text_blocks = page.get_text("dict")
            page_text = ""
            
            # Process blocks in reading order
            blocks = sorted(text_blocks["blocks"], key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
            
            for block in blocks:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            text = span["text"]
                            # Preserve formatting indicators
                            if span["flags"] & 2**4:  # Bold
                                text = f"**{text}**"
                            line_text += text
                        block_text += line_text + "\n"
                    page_text += block_text + "\n"
            
            # Clean and process the extracted text
            cleaned_text = self._clean_text(page_text)
            
            if cleaned_text.strip():
                # Create chunk with metadata
                chunk_metadata = {
                    **doc_metadata,
                    'page': page_num + 1,
                    'section': self._identify_section(cleaned_text),
                }
                
                chunk = DocumentChunk(
                    text=cleaned_text,
                    metadata=chunk_metadata,
                    page_number=page_num + 1
                )
                chunks.append(chunk)
        
        doc.close()
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Apply text cleaning patterns
        for pattern, replacement in self.text_cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
    
    def _identify_section(self, text: str) -> str:
        """Identify document section based on content"""
        text_lower = text.lower()
        
        # Common section indicators for academic/program documents
        if any(keyword in text_lower for keyword in ['table of contents', 'contents', 'sommaire']):
            return 'table_of_contents'
        elif any(keyword in text_lower for keyword in ['introduction', 'overview', 'aperçu']):
            return 'introduction'
        elif any(keyword in text_lower for keyword in ['curriculum', 'courses', 'cours', 'program']):
            return 'curriculum'
        elif any(keyword in text_lower for keyword in ['admission', 'requirements', 'prerequis']):
            return 'admission'
        elif any(keyword in text_lower for keyword in ['career', 'employment', 'job', 'emploi']):
            return 'career'
        elif any(keyword in text_lower for keyword in ['faculty', 'staff', 'professor', 'enseignant']):
            return 'faculty'
        elif any(keyword in text_lower for keyword in ['schedule', 'calendar', 'dates', 'planning']):
            return 'schedule'
        else:
            return 'content'

class DocumentParser:
    """Main document parser that handles multiple file types"""
    
    def __init__(self):
        self.pdf_parser = AdvancedPDFParser()
        self.supported_extensions = {'.pdf': self._parse_pdf}
    
    def parse_document(self, file_path: str) -> List[DocumentChunk]:
        """Parse a document and return structured chunks"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        logger.info(f"Parsing document: {file_path}")
        return self.supported_extensions[file_extension](file_path)
    
    def _parse_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Parse PDF document"""
        return self.pdf_parser.extract_text_with_structure(file_path)
    
    def parse_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Parse all supported documents in a directory"""
        all_chunks = []
        directory = Path(directory_path)
        
        for file_path in directory.glob("*.pdf"):
            try:
                chunks = self.parse_document(str(file_path))
                all_chunks.extend(chunks)
                logger.info(f"Successfully parsed {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
        
        return all_chunks

# Example usage and testing
if __name__ == "__main__":
    parser = DocumentParser()
    
    # Test with a single document
    sample_path = "./Documents"
    if os.path.exists(sample_path):
        chunks = parser.parse_directory(sample_path)
        print(f"Parsed {len(chunks)} chunks from {sample_path}")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            print(f"\nSample chunk metadata: {sample_chunk.metadata}")
            print(f"Sample text (first 200 chars): {sample_chunk.text[:200]}...") 
