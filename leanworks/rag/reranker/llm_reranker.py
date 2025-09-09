from typing import List, Any
import json
import logging
import datetime
import math
import time
import asyncio
import threading
from leanworks.setting import *
from leanworks.rag.reranker.base_reranker import BaseReranker
import random
# Set up logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker module that scores query-document pairs directly.
    This provides more accurate semantic matching than vector similarity alone,
    especially for queries with minimal context.
    """
    def __init__(self, model_client, cache_size: int = 1000, max_concurrent_requests: int = 3):
        """
        Initialize the reranker with a model client.
        
        Args:
            model_client: The model client to use for scoring (e.g., OpenAI client)
            cache_size: Size of the LRU cache for document scoring
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        super().__init__()
        self.model_client = model_client
        self._score_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_size = cache_size
        self._max_batch_size = 5
        self._max_retries = 3
        self._base_delay = 1.0  # Base delay for exponential backoff
        self._max_concurrent_requests = max_concurrent_requests
        logger.info("CrossEncoderReranker initialized")
    
    @property
    def rate_limit_semaphore(self):
        """Get or create rate limit semaphore for current event loop."""
        if not hasattr(self, '_current_semaphore'):
            self._current_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        return self._current_semaphore
    
    
    def rerank(self, query: str, documents: List[Any], **kwargs) -> List[Any]:
        """Synchronous wrapper that properly handles async context."""
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # If we're already in async context, raise an error
            raise RuntimeError("Cannot call sync rerank from async context. Use rerank_async instead.")
        except RuntimeError as e:
            # If the error message contains our text, re-raise it
            if "Cannot call sync rerank from async context" in str(e):
                raise
            # Otherwise, no event loop running, safe to use asyncio.run()
            return asyncio.run(self.rerank_async(query, documents, **kwargs))

    async def rerank_async(
            self, query: str, 
            documents: List[Any], 
            top_k: int = RERANK_TOP_K, 
            min_score_threshold: float = MIN_SCORE_THRESHOLD, 
            recency_weight: float = RECENCY_WEIGHT,
            recency_coefficient: float = RECENCY_COEFFICIENT
            ) -> List[Any]:
        """
        Asynchronously rerank documents based on their relevance to the query and recency.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            top_k: Number of top documents to return after reranking
            min_score_threshold: Minimum score threshold for documents to keep
            recency_weight: Weight given to recency in the final score (0-1)
            recency_coefficient: Coefficient for recency calculation
        Returns:
            List of reranked documents
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Async reranking {len(documents)} documents for query: '{query}'")
        
        # Extract document texts using query-focused extraction
        doc_texts = [self._extract_query_focused_text(doc, query) for doc in documents]
        
        # Score documents using the model
        try:
            scores = await self._score_documents_async(query, doc_texts)
            
            # Get current time for recency calculation (UTC)
            current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
            
            # Add scores to documents and calculate combined score with recency
            for i, doc in enumerate(documents):
                semantic_score = scores[i]
                
                # Calculate recency score (normalized between 0-1)
                recency_score = 0.0
                if hasattr(doc, "metadata") and "timestamp" in doc.metadata:
                    try:
                        doc_timestamp = float(doc.metadata["timestamp"])
                        # Calculate recency score - more recent = higher score
                        # Use exponential decay based on days difference
                        days_diff = (current_time - doc_timestamp) / (60 * 60 * 24)  # Convert to days
                        recency_score = max(0.0, min(1.0, math.exp(-recency_coefficient * days_diff)))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid timestamp format in document metadata")
                
                # Combine semantic score with recency score
                combined_score = (1 - recency_weight) * semantic_score + recency_weight * recency_score
                
                setattr(doc, "semantic_score", semantic_score)
                setattr(doc, "recency_score", recency_score)
                setattr(doc, "rerank_score", combined_score)
            
            # Sort by combined score (descending)
            reranked_docs = sorted(documents, key=lambda x: getattr(x, "rerank_score", 0), reverse=True)
            
            # Filter by minimum score if specified
            if min_score_threshold > 0:
                reranked_docs = [doc for doc in reranked_docs if getattr(doc, "rerank_score", 0) >= min_score_threshold]
            
            # Return top-k documents
            result = reranked_docs[:top_k]
            logger.info(f"Successfully async reranked documents with recency boost. Returning top {len(result)} results")
            return result
            
        except Exception as e:
            logger.error(f"Error during async reranking: {str(e)}")
            # Return original documents in case of error, sorted by timestamp if available
            try:
                # Sort by timestamp (most recent first) as fallback
                return sorted(documents, 
                              key=lambda x: x.metadata.get("timestamp", 0) if hasattr(x, "metadata") else 0, 
                              reverse=True)[:top_k]
            except Exception:
                # If sorting fails, return unsorted
                return documents[:top_k]

    async def _score_documents_async(self, query: str, documents: List[str]) -> List[float]:
        """
        Score documents based on their relevance to the query using the model.
        Uses async processing and rate limiting for better performance.
        
        Args:
            query: The user query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        # Check cache for entire query-documents combination
        cache_key = self._create_cache_key(query, documents)
        with self._cache_lock:
            if cache_key in self._score_cache:
                return self._score_cache[cache_key]
        
        scores = [0.0] * len(documents)
        
        # For very small document sets, just process directly
        if len(documents) <= self._max_batch_size:
            batch_scores = await self._batch_score_documents_async(query, documents)
            scores = batch_scores
        else:
            # Process documents in parallel batches with rate limiting
            semaphore = asyncio.Semaphore(3)  # Limit concurrent batches
            
            async def process_batch(start_idx: int, batch_docs: List[str]) -> tuple:
                async with semaphore:
                    batch_scores = await self._batch_score_documents_async(query, batch_docs)
                    return start_idx, batch_scores
            
            # Create tasks for each batch
            tasks = []
            for i in range(0, len(documents), self._max_batch_size):
                batch_docs = documents[i:i+self._max_batch_size]
                task = process_batch(i, batch_docs)
                tasks.append(task)
            
            # Wait for all batches to complete
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch scoring: {str(result)}")
                        continue
                    
                    start_idx, batch_scores = result
                    # Put scores in the right positions
                    for j, score in enumerate(batch_scores):
                        if start_idx + j < len(scores):
                            scores[start_idx + j] = score
                            
            except Exception as e:
                logger.error(f"Error in parallel batch processing: {str(e)}")
                # Fill with default scores
                scores = [0.5] * len(documents)
        
        # Update cache
        with self._cache_lock:
            # Simple cache management - if too many items, clear half the cache
            if len(self._score_cache) >= self._cache_size:
                keys_to_remove = list(self._score_cache.keys())[:self._cache_size // 2]
                for k in keys_to_remove:
                    self._score_cache.pop(k, None)
            
            self._score_cache[cache_key] = scores
        
        return scores

    async def _batch_score_documents_async(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a batch of documents using the model with rate limiting and retry logic.
        
        Args:
            query: The user query
            documents: List of document texts
            
        Returns:
            List of relevance scores for the batch
        """
        # Handle empty batch
        if not documents:
            return []
        
        # Helper to sanitize and limit document length
        def sanitize_document(doc: str, max_chars: int = 1000) -> str:
            # Truncate overly long documents
            if len(doc) > max_chars:
                doc = doc[:max_chars] + "..."
            # Remove problematic characters that might affect formatting
            doc = doc.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Collapse multiple spaces
            import re
            doc = re.sub(r'\s+', ' ', doc).strip()
            return doc
        
        # Helper to create span-exact cache key
        def _doc_cache_key(query: str, doc: str) -> str:
            import hashlib
            h = hashlib.md5(doc.encode()).hexdigest()
            return f"{query}::{h}"
        
        # Check individual document caches
        cached_scores = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            # Use span-exact cache key
            doc_key = _doc_cache_key(query, doc)
            score = None
            
            with self._cache_lock:
                if doc_key in self._score_cache:
                    score = self._score_cache[doc_key]
            
            if score is not None:
                cached_scores.append((i, score))
            else:
                uncached_docs.append(doc)
                uncached_indices.append(i)
        
        # If all documents are cached, return scores directly
        if not uncached_docs:
            # Sort by original index and return scores
            return [score for _, score in sorted(cached_scores, key=lambda x: x[0])]
        
        # Prepare prompt for scoring only uncached documents
        prompt = "Rate the relevance of each document to the query on a scale of 0 to 10, where 10 is extremely relevant and 0 is completely irrelevant.\n\n"
        prompt += f"Query: {query}\n\n"
        
        for i, doc in enumerate(uncached_docs):
            prompt += f"Document {i+1}: {sanitize_document(doc)}\n\n"
        
        prompt += f"\nYou must return EXACTLY {len(uncached_docs)} scores, one for each document, in the format: Score1, Score2, Score3, ...\n"
        prompt += "Example response format: 8, 7, 9, 5\n"
        prompt += "DO NOT include any other text or explanation in your response."
        
        # Get model response with rate limiting and retry logic
        for attempt in range(self._max_retries):
            try:
                # Use semaphore to limit concurrent requests
                async with self.rate_limit_semaphore:
                    logger.info(f"Requesting scores for {len(uncached_docs)} documents (attempt {attempt + 1})")
                    
                    # Make the API call
                    response = self.model_client.chat.completions.create(
                        model=RERANK_MODEL,
                        max_tokens=256,
                        temperature=0.0,  # Use deterministic output
                        messages=[
                            {"role": "system", "content": "You are a document ranking assistant. Your task is to rate how relevant each document is to the given query on a scale of 0-10. You must respond ONLY with comma-separated numerical scores (e.g., '8, 7, 9, 6'). Do not include any other text, explanations, or formatting in your response."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Parse scores from response
                    answer = response.choices[0].message.content
                    logger.debug(f"Raw reranking response: {answer}")
                    
                    # Parse and process scores
                    scores = await self._parse_scores_async(answer, len(uncached_docs))
                    
                    # Update cache with new scores
                    with self._cache_lock:
                        for i, doc in enumerate(uncached_docs):
                            doc_key = _doc_cache_key(query, doc)
                            self._score_cache[doc_key] = scores[i]
                    
                    # Combine cached and new scores
                    all_scores = [0.0] * len(documents)
                    
                    # Fill in cached scores
                    for idx, score in cached_scores:
                        all_scores[idx] = score
                        
                    # Fill in newly calculated scores
                    for i, orig_idx in enumerate(uncached_indices):
                        all_scores[orig_idx] = scores[i]
                        
                    return all_scores
                    
            except Exception as e:
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    wait_time = self._calculate_backoff_delay_async(attempt)
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}. Waiting {wait_time:.2f} seconds before retry")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error calling model for reranking (attempt {attempt + 1}): {str(e)}")
                    if attempt == self._max_retries - 1:
                        # Return default scores on final attempt
                        return [0.5] * len(documents)
                    
                    # Wait before retrying non-rate-limit errors
                    await asyncio.sleep(self._base_delay * (attempt + 1))
        
        # If all retries failed, return default scores
        logger.error("All retry attempts failed, returning default scores")
        return [0.5] * len(documents)

    def _calculate_backoff_delay_async(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter for async operations.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        
        # Exponential backoff: base_delay * (2^attempt)
        delay = self._base_delay * (2 ** attempt)
        
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.5)
        
        # Cap the maximum delay at 60 seconds
        return min(delay + jitter, 60.0)

    async def _parse_scores_async(self, response_text: str, expected_count: int) -> List[float]:
        """
        Parse scores from model response text (async version).
        
        Args:
            response_text: The raw response from the model
            expected_count: Expected number of scores
            
        Returns:
            List of normalized scores (0-1 range)
        """
        try:
            scores = []
            
            # Method 1: Try comma-separated values first (most common format)
            if "," in response_text:
                for score_str in response_text.split(","):
                    # Clean and extract numeric values
                    cleaned = ''.join([c for c in score_str if c.isdigit() or c == '.'])
                    if cleaned:
                        try:
                            scores.append(float(cleaned))
                        except ValueError:
                            pass
            
            # Method 2: If that didn't work, look for numbers with common prefixes/patterns
            if len(scores) != expected_count:
                import re
                # Look for patterns like "1: 8" or "Document 1: 8" or just numbers
                score_matches = re.findall(r'(?:Document\s*)?(?:\d+\s*[:.-])?\s*(\d+(?:\.\d+)?)', response_text)
                if score_matches:
                    scores = [float(match) for match in score_matches]
            
            # Method 3: If all else fails, extract any numbers in the text
            if len(scores) != expected_count:
                scores = [float(num) for num in response_text.split() if num.replace(".", "").isdigit()]
            
            # Verify we have the expected number of scores
            if len(scores) > expected_count:
                # Too many scores, trim to expected number
                logger.warning(f"Found {len(scores)} scores, trimming to {expected_count}")
                scores = scores[:expected_count]
            
            # Normalize to ensure we have correct number of scores
            if len(scores) < expected_count:
                logger.warning(f"Failed to parse correct number of scores from: {response_text}. Found {len(scores)}, expected {expected_count}")
                # Use found scores and fill rest with default
                scores = scores + [5.0] * (expected_count - len(scores))
            
            # Ensure scores are in the valid range
            scores = [max(0.0, min(10.0, score)) for score in scores]
            
            # Convert scores to 0-1 range
            return [score / 10.0 for score in scores]
            
        except Exception as e:
            logger.warning(f"Failed to parse scores from: {response_text}. Error: {str(e)}")
            return [0.5] * expected_count  # Default score

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if the error is a rate limit error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if it's a rate limit error, False otherwise
        """
        error_str = str(error).lower()
        return (
            "rate limit" in error_str or
            "429" in error_str or
            "too many requests" in error_str or
            "quota" in error_str
        )

    def _create_cache_key(self, query: str, documents: List[str]) -> str:
        """Create a unique cache key based on query and document content."""
        import hashlib
        
        # Create a hash of the documents content
        doc_content = "::".join(doc[:100] for doc in documents)  # First 100 chars of each doc
        content_hash = hashlib.md5(f"{query}::{doc_content}".encode()).hexdigest()
        
        return f"{query}::{len(documents)}::{content_hash}"
