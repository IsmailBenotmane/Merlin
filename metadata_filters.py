"""
Advanced metadata filtering system for RAG document retrieval
"""
import re
from typing import List, Dict, Any, Optional, Union, Set, Callable
from dataclasses import dataclass
from datetime import datetime, date
import logging
from enum import Enum

from document_parser import DocumentChunk
from config import Config

logger = logging.getLogger(__name__)

class FilterOperator(Enum):
    """Supported filter operators"""
    EQUALS = "eq"
    NOT_EQUALS = "ne" 
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    EXISTS = "exists"
    RANGE = "range"

@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True

@dataclass
class FilterGroup:
    """Represents a group of filter conditions with logical operator"""
    conditions: List[Union[FilterCondition, 'FilterGroup']]
    logical_operator: str = "AND"  # "AND" or "OR"

class MetadataFilterBuilder:
    """Builder for constructing complex metadata filters"""
    
    def __init__(self):
        self.conditions = []
        self.current_group = None
    
    def where(self, field: str, operator: Union[FilterOperator, str], value: Any, 
              case_sensitive: bool = True) -> 'MetadataFilterBuilder':
        """Add a filter condition"""
        if isinstance(operator, str):
            try:
                operator = FilterOperator(operator)
            except ValueError:
                raise ValueError(f"Invalid operator: {operator}")
        
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value,
            case_sensitive=case_sensitive
        )
        
        if self.current_group is not None:
            self.current_group.conditions.append(condition)
        else:
            self.conditions.append(condition)
        
        return self
    
    def equals(self, field: str, value: Any) -> 'MetadataFilterBuilder':
        """Convenience method for equality filter"""
        return self.where(field, FilterOperator.EQUALS, value)
    
    def not_equals(self, field: str, value: Any) -> 'MetadataFilterBuilder':
        """Convenience method for not equals filter"""
        return self.where(field, FilterOperator.NOT_EQUALS, value)
    
    def contains(self, field: str, value: str, case_sensitive: bool = False) -> 'MetadataFilterBuilder':
        """Convenience method for contains filter"""
        return self.where(field, FilterOperator.CONTAINS, value, case_sensitive)
    
    def in_list(self, field: str, values: List[Any]) -> 'MetadataFilterBuilder':
        """Convenience method for in filter"""
        return self.where(field, FilterOperator.IN, values)
    
    def range_filter(self, field: str, min_val: Any, max_val: Any) -> 'MetadataFilterBuilder':
        """Convenience method for range filter"""
        return self.where(field, FilterOperator.RANGE, (min_val, max_val))
    
    def greater_than(self, field: str, value: Any) -> 'MetadataFilterBuilder':
        """Convenience method for greater than filter"""
        return self.where(field, FilterOperator.GREATER_THAN, value)
    
    def less_than(self, field: str, value: Any) -> 'MetadataFilterBuilder':
        """Convenience method for less than filter"""
        return self.where(field, FilterOperator.LESS_THAN, value)
    
    def exists(self, field: str) -> 'MetadataFilterBuilder':
        """Convenience method for field existence filter"""
        return self.where(field, FilterOperator.EXISTS, True)
    
    def regex_match(self, field: str, pattern: str, case_sensitive: bool = True) -> 'MetadataFilterBuilder':
        """Convenience method for regex filter"""
        return self.where(field, FilterOperator.REGEX, pattern, case_sensitive)
    
    def group(self, logical_operator: str = "AND") -> 'MetadataFilterBuilder':
        """Start a new filter group"""
        if logical_operator not in ["AND", "OR"]:
            raise ValueError("Logical operator must be 'AND' or 'OR'")
        
        self.current_group = FilterGroup(conditions=[], logical_operator=logical_operator)
        return self
    
    def end_group(self) -> 'MetadataFilterBuilder':
        """End the current filter group"""
        if self.current_group is not None:
            self.conditions.append(self.current_group)
            self.current_group = None
        return self
    
    def build(self) -> FilterGroup:
        """Build the final filter group"""
        if self.current_group is not None:
            self.end_group()
        
        return FilterGroup(conditions=self.conditions, logical_operator="AND")

class MetadataFilterEngine:
    """Engine for applying metadata filters to document chunks"""
    
    def __init__(self):
        self.operators = {
            FilterOperator.EQUALS: self._equals,
            FilterOperator.NOT_EQUALS: self._not_equals,
            FilterOperator.GREATER_THAN: self._greater_than,
            FilterOperator.GREATER_THAN_EQUAL: self._greater_than_equal,
            FilterOperator.LESS_THAN: self._less_than,
            FilterOperator.LESS_THAN_EQUAL: self._less_than_equal,
            FilterOperator.IN: self._in,
            FilterOperator.NOT_IN: self._not_in,
            FilterOperator.CONTAINS: self._contains,
            FilterOperator.NOT_CONTAINS: self._not_contains,
            FilterOperator.STARTS_WITH: self._starts_with,
            FilterOperator.ENDS_WITH: self._ends_with,
            FilterOperator.REGEX: self._regex,
            FilterOperator.EXISTS: self._exists,
            FilterOperator.RANGE: self._range,
        }
    
    def apply_filters(self, chunks: List[DocumentChunk], 
                     filter_group: FilterGroup) -> List[DocumentChunk]:
        """Apply filters to a list of document chunks"""
        return [chunk for chunk in chunks if self._evaluate_group(chunk, filter_group)]
    
    def _evaluate_group(self, chunk: DocumentChunk, group: FilterGroup) -> bool:
        """Evaluate a filter group against a document chunk"""
        results = []
        
        for condition_or_group in group.conditions:
            if isinstance(condition_or_group, FilterCondition):
                result = self._evaluate_condition(chunk, condition_or_group)
            elif isinstance(condition_or_group, FilterGroup):
                result = self._evaluate_group(chunk, condition_or_group)
            else:
                logger.warning(f"Unknown condition type: {type(condition_or_group)}")
                result = False
            
            results.append(result)
        
        # Apply logical operator
        if group.logical_operator == "AND":
            return all(results) if results else True
        elif group.logical_operator == "OR":
            return any(results) if results else False
        else:
            logger.warning(f"Unknown logical operator: {group.logical_operator}")
            return True
    
    def _evaluate_condition(self, chunk: DocumentChunk, condition: FilterCondition) -> bool:
        """Evaluate a single filter condition"""
        field_value = chunk.metadata.get(condition.field)
        
        # Handle non-existent fields
        if field_value is None and condition.operator != FilterOperator.EXISTS:
            return False
        
        operator_func = self.operators.get(condition.operator)
        if operator_func is None:
            logger.warning(f"Unsupported operator: {condition.operator}")
            return False
        
        try:
            return operator_func(field_value, condition.value, condition.case_sensitive)
        except Exception as e:
            logger.warning(f"Error evaluating condition {condition.field} {condition.operator} {condition.value}: {e}")
            return False
    
    def _equals(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Equality comparison"""
        if isinstance(field_value, str) and isinstance(condition_value, str) and not case_sensitive:
            return field_value.lower() == condition_value.lower()
        return field_value == condition_value
    
    def _not_equals(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Not equals comparison"""
        return not self._equals(field_value, condition_value, case_sensitive)
    
    def _greater_than(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Greater than comparison"""
        return field_value > condition_value
    
    def _greater_than_equal(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Greater than or equal comparison"""
        return field_value >= condition_value
    
    def _less_than(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Less than comparison"""
        return field_value < condition_value
    
    def _less_than_equal(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """Less than or equal comparison"""
        return field_value <= condition_value
    
    def _in(self, field_value: Any, condition_value: List[Any], case_sensitive: bool) -> bool:
        """In list comparison"""
        if isinstance(field_value, str) and not case_sensitive:
            return field_value.lower() in [str(v).lower() for v in condition_value]
        return field_value in condition_value
    
    def _not_in(self, field_value: Any, condition_value: List[Any], case_sensitive: bool) -> bool:
        """Not in list comparison"""
        return not self._in(field_value, condition_value, case_sensitive)
    
    def _contains(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """Contains substring comparison"""
        if not isinstance(field_value, str):
            field_value = str(field_value)
        
        if not case_sensitive:
            return condition_value.lower() in field_value.lower()
        return condition_value in field_value
    
    def _not_contains(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """Not contains substring comparison"""
        return not self._contains(field_value, condition_value, case_sensitive)
    
    def _starts_with(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """Starts with comparison"""
        if not isinstance(field_value, str):
            field_value = str(field_value)
        
        if not case_sensitive:
            return field_value.lower().startswith(condition_value.lower())
        return field_value.startswith(condition_value)
    
    def _ends_with(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """Ends with comparison"""
        if not isinstance(field_value, str):
            field_value = str(field_value)
        
        if not case_sensitive:
            return field_value.lower().endswith(condition_value.lower())
        return field_value.endswith(condition_value)
    
    def _regex(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """Regular expression matching"""
        if not isinstance(field_value, str):
            field_value = str(field_value)
        
        flags = 0 if case_sensitive else re.IGNORECASE
        return bool(re.search(condition_value, field_value, flags))
    
    def _exists(self, field_value: Any, condition_value: bool, case_sensitive: bool) -> bool:
        """Field existence check"""
        exists = field_value is not None
        return exists if condition_value else not exists
    
    def _range(self, field_value: Any, condition_value: tuple, case_sensitive: bool) -> bool:
        """Range comparison"""
        min_val, max_val = condition_value
        return min_val <= field_value <= max_val

class ChromaDBFilterAdapter:
    """Adapter to convert filter groups to ChromaDB where clauses"""
    
    @staticmethod
    def to_chroma_where(filter_group: FilterGroup) -> Dict[str, Any]:
        """Convert filter group to ChromaDB where clause"""
        if not filter_group.conditions:
            return {}
        
        # Simple cases - single condition
        if len(filter_group.conditions) == 1 and isinstance(filter_group.conditions[0], FilterCondition):
            return ChromaDBFilterAdapter._condition_to_chroma(filter_group.conditions[0])
        
        # Complex cases with multiple conditions
        chroma_conditions = []
        for condition_or_group in filter_group.conditions:
            if isinstance(condition_or_group, FilterCondition):
                chroma_condition = ChromaDBFilterAdapter._condition_to_chroma(condition_or_group)
                if chroma_condition:
                    chroma_conditions.append(chroma_condition)
            elif isinstance(condition_or_group, FilterGroup):
                nested_condition = ChromaDBFilterAdapter.to_chroma_where(condition_or_group)
                if nested_condition:
                    chroma_conditions.append(nested_condition)
        
        if not chroma_conditions:
            return {}
        
        if len(chroma_conditions) == 1:
            return chroma_conditions[0]
        
        # Combine conditions with logical operator
        if filter_group.logical_operator == "AND":
            return {"$and": chroma_conditions}
        elif filter_group.logical_operator == "OR":
            return {"$or": chroma_conditions}
        
        return chroma_conditions[0]  # Fallback
    
    @staticmethod
    def _condition_to_chroma(condition: FilterCondition) -> Dict[str, Any]:
        """Convert a single condition to ChromaDB format"""
        field = condition.field
        value = condition.value
        
        # ChromaDB operator mapping
        operator_map = {
            FilterOperator.EQUALS: lambda v: {field: v},
            FilterOperator.NOT_EQUALS: lambda v: {field: {"$ne": v}},
            FilterOperator.GREATER_THAN: lambda v: {field: {"$gt": v}},
            FilterOperator.GREATER_THAN_EQUAL: lambda v: {field: {"$gte": v}},
            FilterOperator.LESS_THAN: lambda v: {field: {"$lt": v}},
            FilterOperator.LESS_THAN_EQUAL: lambda v: {field: {"$lte": v}},
            FilterOperator.IN: lambda v: {field: {"$in": v}},
            FilterOperator.NOT_IN: lambda v: {field: {"$nin": v}},
            FilterOperator.RANGE: lambda v: {field: {"$gte": v[0], "$lte": v[1]}},
        }
        
        converter = operator_map.get(condition.operator)
        if converter:
            return converter(value)
        
        logger.warning(f"ChromaDB does not support operator: {condition.operator}")
        return {}

class FilterPresets:
    """Common filter presets for document types"""
    
    @staticmethod
    def by_document_type(doc_type: str) -> MetadataFilterBuilder:
        """Filter by document type"""
        return MetadataFilterBuilder().equals("document_type", doc_type)
    
    @staticmethod
    def by_source(source: str) -> MetadataFilterBuilder:
        """Filter by source document"""
        return MetadataFilterBuilder().equals("source", source)
    
    @staticmethod
    def by_page_range(min_page: int, max_page: int) -> MetadataFilterBuilder:
        """Filter by page range"""
        return MetadataFilterBuilder().range_filter("page", min_page, max_page)
    
    @staticmethod
    def by_section(section: str) -> MetadataFilterBuilder:
        """Filter by document section"""
        return MetadataFilterBuilder().equals("section", section)
    
    @staticmethod
    def academic_documents() -> MetadataFilterBuilder:
        """Filter for academic/program documents"""
        return (MetadataFilterBuilder()
                .in_list("document_type", ["program_info", "course_list", "brochure"])
                .group("OR")
                .contains("section", "curriculum", case_sensitive=False)
                .contains("section", "program", case_sensitive=False)
                .end_group())
    
    @staticmethod
    def career_documents() -> MetadataFilterBuilder:
        """Filter for career-related documents"""
        return (MetadataFilterBuilder()
                .group("OR")
                .equals("document_type", "employment_data")
                .contains("section", "career", case_sensitive=False)
                .contains("section", "employment", case_sensitive=False)
                .end_group())
    
    @staticmethod
    def presentation_slides() -> MetadataFilterBuilder:
        """Filter for presentation documents"""
        return MetadataFilterBuilder().equals("document_type", "presentation")
    
    @staticmethod
    def recent_pages(max_page: int = 10) -> MetadataFilterBuilder:
        """Filter for early pages (typically contain overview info)"""
        return MetadataFilterBuilder().less_than_equal("page", max_page)
    
    @staticmethod
    def large_chunks(min_size: int = 1000) -> MetadataFilterBuilder:
        """Filter for large chunks (typically more detailed content)"""
        return MetadataFilterBuilder().greater_than_equal("chunk_text_length", min_size)

# Example usage and testing
if __name__ == "__main__":
    from document_parser import DocumentParser
    from chunking_strategy import DocumentChunker
    
    logger.info("Testing Metadata Filters")
    
    # Parse and chunk documents for testing
    parser = DocumentParser()
    chunker = DocumentChunker()
    
    sample_path = "./Documents"
    from pathlib import Path
    
    if Path(sample_path).exists():
        doc_chunks = parser.parse_directory(sample_path)
        text_chunks = chunker.chunk_documents(doc_chunks)
        
        print(f"Total chunks for filtering: {len(text_chunks)}")
        
        # Initialize filter engine
        filter_engine = MetadataFilterEngine()
        
        # Test various filters
        test_filters = [
            ("Academic documents", FilterPresets.academic_documents().build()),
            ("Career documents", FilterPresets.career_documents().build()),
            ("Presentation slides", FilterPresets.presentation_slides().build()),
            ("Recent pages (1-5)", FilterPresets.recent_pages(5).build()),
            ("Large chunks (>800 chars)", FilterPresets.large_chunks(800).build()),
            ("Program info from brochure", 
             MetadataFilterBuilder()
             .equals("document_type", "brochure")
             .contains("section", "program", case_sensitive=False)
             .build()
            ),
            ("Complex filter example",
             MetadataFilterBuilder()
             .group("OR")
             .equals("document_type", "employment_data")
             .group("AND")
             .contains("source", "brochure", case_sensitive=False)
             .less_than_equal("page", 10)
             .end_group()
             .end_group()
             .build()
            )
        ]
        
        for filter_name, filter_group in test_filters:
            filtered_chunks = filter_engine.apply_filters(text_chunks, filter_group)
            print(f"\n{filter_name}: {len(filtered_chunks)} chunks")
            
            if filtered_chunks:
                # Show sample results
                sample = filtered_chunks[0]
                print(f"  Sample: {sample.metadata.get('source', 'unknown')} "
                      f"(page {sample.metadata.get('page', '?')}, "
                      f"type: {sample.metadata.get('document_type', 'unknown')})")
                print(f"  Text preview: {sample.text[:100]}...")
            
            # Test ChromaDB conversion
            chroma_where = ChromaDBFilterAdapter.to_chroma_where(filter_group)
            print(f"  ChromaDB where clause: {chroma_where}")
    
    else:
        logger.warning(f"Documents directory not found: {sample_path}") 
