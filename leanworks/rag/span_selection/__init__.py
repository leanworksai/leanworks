"""
Span Selection Package for RAG Pipeline

This package provides span selection functionality using either LLM or BGE-based approaches.
"""

from .span_selection_factory import SpanSelectionFactory, SpanSelector
from .span_selection_base import BaseSpanSelector
from .span_selection_llm import LLMSpanSelector
from .span_selection_bge import BGESpanSelector

__all__ = [
    'SpanSelectionFactory',
    'BaseSpanSelector', 
    'LLMSpanSelector',
    'BGESpanSelector',
    'SpanSelector'
]
