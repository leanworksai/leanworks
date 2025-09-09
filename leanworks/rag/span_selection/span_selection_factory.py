"""
Span Selection Factory Module for RAG Pipeline

This module provides a factory for creating different types of span selectors based on configuration.
"""

from typing import Optional, List, Dict, Any
import logging
from .span_selection_base import BaseSpanSelector
from leanworks.setting import (
    SPAN_SELECTION_TYPE, BGE_MODEL_NAME, BGE_DEVICE, BGE_MAX_WORKERS, BGE_CACHE_SIZE,
    BGE_MAX_LENGTH, BGE_BATCH_SIZE, BGE_INTRA_OP_THREADS, BGE_INTER_OP_THREADS
)

logger = logging.getLogger(__name__)

class SpanSelectionFactory:
    """
    Factory class for creating different types of span selectors based on configuration.
    """
    
    @staticmethod
    def create_span_selector(
        span_selection_type: str = SPAN_SELECTION_TYPE,
        reranker: Optional[object] = None,
        **kwargs
    ) -> BaseSpanSelector:
        """
        Create a span selector instance based on the specified type.
        
        Args:
            span_selection_type: Type of span selector to create ("llm" or "bge")
            reranker: Reranker instance (LLM or BGE reranker depending on type)
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseSpanSelector instance
            
        Raises:
            ValueError: If span selection type is unsupported or required parameters are missing
        """
        span_selection_type = span_selection_type.lower()
        
        if span_selection_type == "llm":
            if reranker is None:
                raise ValueError("reranker is required for LLM-based span selector")
            
            top_spans_per_doc = kwargs.get("top_spans_per_doc", 6)
            context_window = kwargs.get("context_window", 1)
            min_span_length = kwargs.get("min_span_length", 10)
            max_span_length = kwargs.get("max_span_length", 500)
            use_sliding_windows = kwargs.get("use_sliding_windows", True)
            window_size = kwargs.get("window_size", 96)
            window_stride = kwargs.get("window_stride", 48)
            max_span_candidates = kwargs.get("max_span_candidates", 60)
            max_final_spans = kwargs.get("max_final_spans", 18)
            use_bm25_prefilter = kwargs.get("use_bm25_prefilter", True)
            bm25_k1 = kwargs.get("bm25_k1", 1.2)
            bm25_b = kwargs.get("bm25_b", 0.75)
            
            logger.info("Creating LLMSpanSelector")
            from .span_selection_llm import LLMSpanSelector
            return LLMSpanSelector(
                top_spans_per_doc=top_spans_per_doc,
                context_window=context_window,
                min_span_length=min_span_length,
                max_span_length=max_span_length,
                llm_reranker=reranker,
                use_sliding_windows=use_sliding_windows,
                window_size=window_size,
                window_stride=window_stride,
                max_span_candidates=max_span_candidates,
                max_final_spans=max_final_spans,
                use_bm25_prefilter=use_bm25_prefilter,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b
            )
            
        elif span_selection_type == "bge":
            if reranker is None:
                raise ValueError("reranker is required for BGE-based span selector")
            
            top_spans_per_doc = kwargs.get("top_spans_per_doc", 6)
            context_window = kwargs.get("context_window", 1)
            min_span_length = kwargs.get("min_span_length", 10)
            max_span_length = kwargs.get("max_span_length", 500)
            use_sliding_windows = kwargs.get("use_sliding_windows", True)
            window_size = kwargs.get("window_size", 96)
            window_stride = kwargs.get("window_stride", 48)
            max_span_candidates = kwargs.get("max_span_candidates", 60)
            max_final_spans = kwargs.get("max_final_spans", 18)
            use_bm25_prefilter = kwargs.get("use_bm25_prefilter", True)
            bm25_k1 = kwargs.get("bm25_k1", 1.2)
            bm25_b = kwargs.get("bm25_b", 0.75)
            
            logger.info("Creating BGESpanSelector")
            from .span_selection_bge import BGESpanSelector
            return BGESpanSelector(
                top_spans_per_doc=top_spans_per_doc,
                context_window=context_window,
                min_span_length=min_span_length,
                max_span_length=max_span_length,
                bge_reranker=reranker,
                use_sliding_windows=use_sliding_windows,
                window_size=window_size,
                window_stride=window_stride,
                max_span_candidates=max_span_candidates,
                max_final_spans=max_final_spans,
                use_bm25_prefilter=use_bm25_prefilter,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b
            )
        else:
            raise ValueError(f"Unsupported span selection type: {span_selection_type}")
    
    @staticmethod
    def get_available_types() -> list:
        """
        Get list of available span selection types.
        
        Returns:
            List of supported span selection type strings
        """
        return ["llm", "bge"]


class SpanSelector:
    """
    Unified span selection module that selects the most relevant spans from documents
    using either LLM or BGE semantic scoring with sliding windows or sentence-based candidates.
    """
    
    def __init__(
        self,
        top_spans_per_doc: int = 6,
        context_window: int = 1,
        min_span_length: int = 10,
        max_span_length: int = 500,
        reranker: Optional[object] = None,
        use_sliding_windows: bool = True,
        window_size: int = 96,
        window_stride: int = 48,
        max_span_candidates: int = 60,
        max_final_spans: int = 18,
        use_bm25_prefilter: bool = True,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        span_selection_type: Optional[str] = None
    ):
        """
        Initialize the SpanSelector using the factory pattern.
        
        Args:
            top_spans_per_doc: Number of top spans to select per document (default 6)
            context_window: Number of neighbor sentences to include (±1)
            min_span_length: Minimum span length to consider
            max_span_length: Maximum span length to consider
            reranker: Reranker instance (LLM or BGE) for semantic scoring
            use_sliding_windows: Whether to use sliding windows vs sentences
            window_size: Size of sliding windows in tokens (default 96)
            window_stride: Stride of sliding windows in tokens (default 48)
            max_span_candidates: Maximum span candidates to score (default 60)
            max_final_spans: Maximum final spans to return (default 18)
            use_bm25_prefilter: Whether to use BM25 pre-filtering before semantic scoring
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            span_selection_type: Type of span selector ("llm" or "bge"), defaults to setting
        """
        # Create the appropriate span selector using the factory
        self.span_selector = SpanSelectionFactory.create_span_selector(
            span_selection_type=span_selection_type,
            reranker=reranker,
            top_spans_per_doc=top_spans_per_doc,
            context_window=context_window,
            min_span_length=min_span_length,
            max_span_length=max_span_length,
            use_sliding_windows=use_sliding_windows,
            window_size=window_size,
            window_stride=window_stride,
            max_span_candidates=max_span_candidates,
            max_final_spans=max_final_spans,
            use_bm25_prefilter=use_bm25_prefilter,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b
        )
        
        logger.info(f"SpanSelector initialized with factory-created {type(self.span_selector).__name__}")
    
    def select_spans(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Select relevant spans from a list of documents using the configured span selector.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            
        Returns:
            List of documents with selected spans in metadata
        """
        return self.span_selector.select_spans(query, documents)
    
    def get_selection_stats(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about span selection results.
        
        Args:
            documents: Processed documents
            
        Returns:
            Dictionary with selection statistics
        """
        return self.span_selector.get_selection_stats(documents)
