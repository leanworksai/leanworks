"""
Base Span Selection Module for RAG Pipeline

This module provides the base class for span selection implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseSpanSelector(ABC):
    """
    Abstract base class for all span selector implementations.
    Provides a common interface for different span selection strategies.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base span selector.
        
        Args:
            **kwargs: Configuration parameters specific to each span selector implementation
        """
        self.logger = logger
        
    @abstractmethod
    def select_spans(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Select relevant spans from a list of documents.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            
        Returns:
            List of documents with selected spans in metadata
        """
        pass
    
    def get_selection_stats(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about span selection results.
        
        Args:
            documents: Processed documents
            
        Returns:
            Dictionary with selection statistics
        """
        stats = {
            "total_documents": len(documents),
            "documents_with_spans": 0,
            "total_selected_spans": 0,
            "avg_spans_per_doc": 0.0,
            "total_original_sentences": 0,
            "selection_ratio": 0.0,
            "selection_method": "unknown",
            "avg_span_score": 0.0,
            "span_scores_available": False
        }
        
        total_span_scores = 0.0
        span_scores_count = 0
        
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata.get("span_selection_applied"):
                stats["documents_with_spans"] += 1
                selected_spans = doc.metadata.get("selected_spans", [])
                stats["total_selected_spans"] += len(selected_spans)
                stats["total_original_sentences"] += doc.metadata.get("total_sentences", 0)
                
                # Track selection method
                method = doc.metadata.get("span_selection_method", "unknown")
                if method != "unknown":
                    stats["selection_method"] = method
                
                # Track span scores if available
                span_scores = doc.metadata.get("span_scores", [])
                if span_scores:
                    stats["span_scores_available"] = True
                    total_span_scores += sum(span_scores)
                    span_scores_count += len(span_scores)
        
        if stats["documents_with_spans"] > 0:
            stats["avg_spans_per_doc"] = stats["total_selected_spans"] / stats["documents_with_spans"]
        
        if stats["total_original_sentences"] > 0:
            stats["selection_ratio"] = stats["total_selected_spans"] / stats["total_original_sentences"]
        
        if span_scores_count > 0:
            stats["avg_span_score"] = total_span_scores / span_scores_count
        
        return stats

    def _compute_rrf_scores(self, score_lists: List[List[float]], k: int) -> List[float]:
        """
        Compute Reciprocal Rank Fusion scores for multiple ranked lists.

        Args:
            score_lists: List of score arrays (higher is better)
            k: RRF constant

        Returns:
            List of fused scores aligned to original indices
        """
        if not score_lists:
            return []
        lengths = {len(scores) for scores in score_lists if scores is not None}
        if len(lengths) != 1:
            return score_lists[0] if score_lists else []
        n = lengths.pop()
        if n == 0:
            return []

        ranks_list = []
        for scores in score_lists:
            ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
            ranks = [0] * n
            for rank, idx in enumerate(ranked_indices, start=1):
                ranks[idx] = rank
            ranks_list.append(ranks)

        rrf_scores = [0.0] * n
        for ranks in ranks_list:
            for i, rank in enumerate(ranks):
                rrf_scores[i] += 1.0 / (k + rank)
        return rrf_scores
