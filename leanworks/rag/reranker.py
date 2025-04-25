from typing import List, Any
import json
import logging
import datetime
import math
import concurrent.futures
import threading
from leanworks.rag.setting import MIN_SCORE_THRESHOLD, RECENCY_WEIGHT, RERANK_MODEL, RERANK_TOP_K

# Set up logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Cross-encoder reranker module that scores query-document pairs directly.
    This provides more accurate semantic matching than vector similarity alone,
    especially for queries with minimal context.
    """
    def __init__(self, model_client, cache_size: int = 1000):
        """
        Initialize the reranker with a model client.
        
        Args:
            model_client: The model client to use for scoring (e.g., OpenAI client)
            cache_size: Size of the LRU cache for document scoring
        """
        self.model_client = model_client
        self._score_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_size = cache_size
        self._max_batch_size = 5
        logger.info("CrossEncoderReranker initialized")
    
    def _extract_text(self, doc) -> str:
        """
        Extract text from document without complex parsing.
        
        Args:
            doc: Document object with metadata
            
        Returns:
            Extracted text from document
        """
        if not hasattr(doc, "metadata"):
            return ""
        
        metadata = doc.metadata
        text = json.loads(metadata.get("_node_content", "")).get("text", "")
        return text
    
    def _extract_query_focused_text(self, doc, query: str) -> str:
        """
        Extract text focused around sections most relevant to the query.
        
        Args:
            doc: Document object with metadata
            query: The user query
            window_size: Number of characters to include around matches
            
        Returns:
            Query-relevant text from document
        """
        # Extract base text using existing method
        text = self._extract_text(doc)
        # If text is short enough, just return it
        if len(text) <= 1000:
            return text
        
        # Split text into 1000-character buckets
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
    
    def rerank(self, query: str, documents: List[Any], top_k: int = RERANK_TOP_K, 
               min_score_threshold: float = MIN_SCORE_THRESHOLD, recency_weight: float = RECENCY_WEIGHT) -> List[Any]:
        """
        Rerank documents based on their relevance to the query and recency.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            top_k: Number of top documents to return after reranking
            min_score_threshold: Minimum score threshold for documents to keep
            recency_weight: Weight given to recency in the final score (0-1)
            
        Returns:
            List of reranked documents
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
        
        # Extract document texts using query-focused extraction
        doc_texts = [self._extract_query_focused_text(doc, query) for doc in documents]
        
        # Score documents using the model
        try:
            scores = self._score_documents(query, doc_texts)
            
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
                        recency_score = max(0.0, min(1.0, math.exp(-0.3 * days_diff)))
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
            logger.info(f"Successfully reranked documents with recency boost. Returning top {len(result)} results")
            return result
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Return original documents in case of error, sorted by timestamp if available
            try:
                # Sort by timestamp (most recent first)
                return sorted(documents, 
                              key=lambda x: x.metadata.get("timestamp", 0) if hasattr(x, "metadata") else 0, 
                              reverse=True)[:top_k]
            except Exception:
                # If sorting fails, return unsorted
                return documents[:top_k]
    
    def _score_documents(self, query: str, documents: List[str]) -> List[float]:
        """
        Score documents based on their relevance to the query using the model.
        Uses parallel processing and caching for better performance.
        
        Args:
            query: The user query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        # Check cache for entire query-documents combination
        cache_key = f"{query}::{len(documents)}"
        with self._cache_lock:
            if cache_key in self._score_cache:
                return self._score_cache[cache_key]
        
        scores = [0.0] * len(documents)
        
        # For very small document sets, just process directly (avoid thread overhead)
        if len(documents) <= self._max_batch_size:
            batch_scores = self._batch_score_documents(query, documents)
            scores = batch_scores
        else:
            # Process documents in parallel batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(0, len(documents), self._max_batch_size):
                    batch_docs = documents[i:i+self._max_batch_size]
                    
                    # Create a future for each batch
                    future = executor.submit(self._batch_score_documents, query, batch_docs)
                    futures.append((future, i))
                
                # Collect results as they complete
                for future, start_idx in futures:
                    try:
                        batch_scores = future.result()
                        # Put scores in the right positions
                        for j, score in enumerate(batch_scores):
                            if start_idx + j < len(scores):
                                scores[start_idx + j] = score
                    except Exception as e:
                        logger.error(f"Error in batch scoring: {str(e)}")
                        # Use default scores for this batch
                        for j in range(len(documents[start_idx:start_idx+self._max_batch_size])):
                            if start_idx + j < len(scores):
                                scores[start_idx + j] = 0.5
        
        # Update cache
        with self._cache_lock:
            # Simple cache management - if too many items, clear half the cache
            if len(self._score_cache) >= self._cache_size:
                keys_to_remove = list(self._score_cache.keys())[:self._cache_size // 2]
                for k in keys_to_remove:
                    self._score_cache.pop(k, None)
            
            self._score_cache[cache_key] = scores
        
        return scores
    
    def _batch_score_documents(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a batch of documents using the model.
        
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
        
        # Check individual document caches
        cached_scores = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            # Use first 100 chars as part of cache key
            doc_key = f"{query}::{sanitize_document(doc[:100])[:100]}"
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
        
        # Get model response for uncached documents
        try:      
            logger.info(f"Requesting scores for {len(uncached_docs)} documents")
            response = self.model_client.chat.completions.create(
                model=RERANK_MODEL,
                max_tokens=1024,
                temperature=0.0,  # Use deterministic output
                messages=[
                    {"role": "system", "content": "You are a document ranking assistant. Your task is to rate how relevant each document is to the given query on a scale of 0-10. You must respond ONLY with comma-separated numerical scores (e.g., '8, 7, 9, 6'). Do not include any other text, explanations, or formatting in your response."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse scores from response
            answer = response.choices[0].message.content
            logger.debug(f"Raw reranking response: {answer}")
            
            # Parse scores from the response with improved robustness
            try:
                # More robust parsing approach
                expected_docs = len(uncached_docs)
                scores = []
                
                # Method 1: Try comma-separated values first (most common format)
                if "," in answer:
                    scores = []
                    for score_str in answer.split(","):
                        # Clean and extract numeric values
                        cleaned = ''.join([c for c in score_str if c.isdigit() or c == '.'])
                        if cleaned:
                            try:
                                scores.append(float(cleaned))
                            except ValueError:
                                pass
                
                # Method 2: If that didn't work, look for numbers with common prefixes/patterns
                if len(scores) != expected_docs:
                    import re
                    # Look for patterns like "1: 8" or "Document 1: 8" or just numbers
                    score_matches = re.findall(r'(?:Document\s*)?(?:\d+\s*[:.-])?\s*(\d+(?:\.\d+)?)', answer)
                    if score_matches:
                        scores = [float(match) for match in score_matches]
                
                # Method 3: If all else fails, extract any numbers in the text
                if len(scores) != expected_docs:
                    scores = [float(num) for num in answer.split() if num.replace(".", "").isdigit()]
                
                # Verify we have the expected number of scores
                if len(scores) > expected_docs:
                    # Too many scores, trim to expected number
                    logger.warning(f"Found {len(scores)} scores, trimming to {expected_docs}")
                    scores = scores[:expected_docs]
                
                # Normalize to ensure we have correct number of scores
                if len(scores) < expected_docs:
                    logger.warning(f"Failed to parse correct number of scores from: {answer}. Found {len(scores)}, expected {expected_docs}")
                    # Use found scores and fill rest with default
                    scores = scores + [5.0] * (expected_docs - len(scores))
                
                # Ensure scores are in the valid range
                scores = [max(0.0, min(10.0, score)) for score in scores]
                
            except Exception as e:
                logger.warning(f"Failed to parse scores from: {answer}. Error: {str(e)}")
                scores = [5.0] * len(uncached_docs)  # Default score
            
            # Convert scores to 0-1 range
            normalized_scores = [score / 10.0 for score in scores]
            
            # Update cache with new scores
            with self._cache_lock:
                for i, doc in enumerate(uncached_docs):
                    doc_key = f"{query}::{sanitize_document(doc[:100])[:100]}"
                    self._score_cache[doc_key] = normalized_scores[i]
            
            # Combine cached and new scores
            all_scores = [0.0] * len(documents)
            
            # Fill in cached scores
            for idx, score in cached_scores:
                all_scores[idx] = score
                
            # Fill in newly calculated scores
            for i, orig_idx in enumerate(uncached_indices):
                all_scores[orig_idx] = normalized_scores[i]
                
            return all_scores
            
        except Exception as e:
            logger.error(f"Error calling model for reranking: {str(e)}")
            # Return default scores
            return [0.5] * len(documents)