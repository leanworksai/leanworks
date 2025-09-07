from abc import ABC, abstractmethod
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """
    Abstract base class for all reranker implementations.
    Provides a common interface for different reranking strategies.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base reranker.
        
        Args:
            **kwargs: Configuration parameters specific to each reranker implementation
        """
        self.logger = logger
        
    @abstractmethod
    def rerank(self, query: str, documents: List[Any], **kwargs) -> List[Any]:
        """
        Synchronous reranking method.
        
        Args:
            query: The user query
            documents: List of document objects to rerank
            **kwargs: Additional parameters for reranking
            
        Returns:
            List of reranked documents
        """
        pass
    
    @abstractmethod
    async def rerank_async(self, query: str, documents: List[Any], **kwargs) -> List[Any]:
        """
        Asynchronous reranking method.
        
        Args:
            query: The user query
            documents: List of document objects to rerank
            **kwargs: Additional parameters for reranking
            
        Returns:
            List of reranked documents
        """
        pass
    
    def _extract_query_focused_text(self, doc, query: str) -> str:
        """
        Extract text focused around sections most relevant to the query.
        Common implementation that can be overridden by subclasses.
        
        Args:
            doc: Document object with metadata
            query: The user query
            
        Returns:
            Query-relevant text from document
        """
        # Extract text directly from metadata
        text = doc.metadata.get("chunk_text", "")
        # If text is short enough, just return it
        if len(text) <= 1000:
            return text
        
        # Split text into 500-character buckets
        bucket_size = 500
        buckets = []
        for i in range(0, len(text), bucket_size):
            buckets.append(text[i:i+bucket_size])
        
        # Calculate simple relevance score for each bucket
        query_terms = set(query.lower().split())

        scored_buckets = []
        
        for i, bucket in enumerate(buckets):
            # Simple term overlap scoring
            bucket_lower = bucket.lower()
            overlap = sum(1 for term in query_terms if term in bucket_lower)
            
            # Boost exact phrase matches
            if query.lower() in bucket_lower:
                overlap += 2
                
            scored_buckets.append((i, overlap))
        
        # Get indices of top matching buckets
        top_matches = sorted(scored_buckets, key=lambda x: x[1], reverse=True)[:3]
        
        # If we didn't find relevant matches, fall back to first portion
        if not top_matches or all(score == 0 for _, score in top_matches):
            return text[:1000]
        
        # Extract windows around matches to maintain context
        selected_text_parts = []
        for idx, _ in top_matches:
            # Add 250 characters from previous bucket if it exists
            if idx > 0:
                prev_bucket = buckets[idx-1]
                selected_text_parts.append("..."+prev_bucket[-250:])

            # Add the matched bucket
            selected_text_parts.append(buckets[idx])
            
            # Add 250 characters from next bucket if it exists
            if idx < len(buckets) - 1:
                next_bucket = buckets[idx+1]
                selected_text_parts.append(next_bucket[:250]+"...")
        
        # Rebuild text with selected parts
        result = " ".join(selected_text_parts)
        return result
