"""
Span Selection Package for RAG Pipeline

This package provides span selection functionality using either LLM or BGE-based approaches.
"""

from .span_selection_factory import SpanSelectionFactory, SpanSelector
from .span_selection_base import BaseSpanSelector

# Import span selectors only when needed to avoid nltk dependency
def get_llm_span_selector():
    """Get LLMSpanSelector class, importing it only when needed."""
    from .span_selection_llm import LLMSpanSelector
    return LLMSpanSelector

def get_bge_span_selector():
    """Get BGESpanSelector class, importing it only when needed."""
    from .span_selection_bge import BGESpanSelector
    return BGESpanSelector

__all__ = [
    'SpanSelectionFactory',
    'BaseSpanSelector', 
    'LLMSpanSelector',
    'BGESpanSelector',
    'SpanSelector',
    'get_llm_span_selector',
    'get_bge_span_selector'
]
