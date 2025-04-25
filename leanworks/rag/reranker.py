from typing import List, Dict, Any, Optional
import json
import logging
import datetime
import math
import concurrent.futures
from functools import lru_cache
import threading
import re
from leanworks.rag.setting import OTHER_MODEL, MIN_SCORE_THRESHOLD, RECENCY_WEIGHT, RERANK_MODEL

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
    
    def rerank(self, query: str, documents: List[Any], top_k: int = 8, 
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
                
                # Print document text and score for debugging
                print(f"Document {i} text: {doc_texts[i]}")
                print(f"Document {i} semantic score: {semantic_score}")
                
                # Calculate recency score (normalized between 0-1)
                recency_score = 0.0
                if hasattr(doc, "metadata") and "timestamp" in doc.metadata:
                    try:
                        doc_timestamp = float(doc.metadata["timestamp"])
                        # Calculate recency score - more recent = higher score
                        # Use exponential decay based on days difference
                        days_diff = (current_time - doc_timestamp) / (60 * 60 * 24)  # Convert to days
                        recency_score = max(0.0, min(1.0, math.exp(-0.1 * days_diff)))
                        print(f"Document {i} recency score: {recency_score}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid timestamp format in document metadata")
                
                # Combine semantic score with recency score
                combined_score = (1 - recency_weight) * semantic_score + recency_weight * recency_score
                print(f"Document {i} combined score: {combined_score}")
                
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
        
        # Check individual document caches
        cached_scores = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            # Use first 100 chars as part of cache key
            doc_key = f"{query}::{doc[:100]}"
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
        prompt = "Rate the relevance of each document to the query on a scale of 0 to 10, where 10 is extremely relevant and 0 is completely irrelevant. Return only the numerical scores separated by commas, without any explanation.\n\n"
        prompt += f"Query: {query}\n\n"
        
        for i, doc in enumerate(uncached_docs):
            prompt += f"Document {i+1}: {doc}\n\n"
        
        # Get model response for uncached documents
        try:      
            response = self.model_client.chat.completions.create(
                model=RERANK_MODEL,
                max_tokens=1024,
                temperature=0.0,  # Use deterministic output
                messages=[
                    {"role": "system", "content": "You are a document ranking assistant. Your task is to rate how relevant each document is to the given query on a scale of 0-10."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse scores from response
            answer = response.choices[0].message.content
            logger.debug(f"Raw reranking response: {answer}")
            
            # Parse scores from the response
            try:
                # Try to parse comma-separated scores
                scores = [float(score.strip()) for score in answer.split(",")]
                
                # Ensure we have the right number of scores
                if len(scores) != len(uncached_docs):
                    # Fallback to parsing numbers from the text
                    scores = [float(num) for num in answer.split() if num.replace(".", "").isdigit()]
                    
                # Normalize to ensure we have correct number of scores
                if len(scores) != len(uncached_docs):
                    logger.warning(f"Failed to parse correct number of scores from: {answer}")
                    scores = [5.0] * len(uncached_docs)  # Default score
            except Exception as e:
                logger.warning(f"Failed to parse scores from: {answer}. Error: {str(e)}")
                scores = [5.0] * len(uncached_docs)  # Default score
            
            # Convert scores to 0-1 range
            normalized_scores = [score / 10.0 for score in scores]
            
            # Update cache with new scores
            with self._cache_lock:
                for i, doc in enumerate(uncached_docs):
                    doc_key = f"{query}::{doc[:100]}"
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

# Alternative reranking strategies can be implemented here
class HybridReranker(CrossEncoderReranker):
    """
    Hybrid reranker that combines semantic, lexical matching scores, and recency.
    This can be more effective for technical or specialized queries.
    """
    def __init__(self, model_client, lexical_weight: float = 0.2, recency_weight: float = RECENCY_WEIGHT, cache_size: int = 1000):
        """
        Initialize the hybrid reranker.
        
        Args:
            model_client: The model client to use for scoring
            lexical_weight: Weight for lexical matching (0-1)
            recency_weight: Weight for recency (0-1)
            cache_size: Size of score cache
        """
        super().__init__(model_client, cache_size=cache_size)
        self.lexical_weight = lexical_weight
        self.recency_weight = recency_weight
        
    def rerank(self, query: str, documents: List[Any], top_k: int = 8, 
               min_score_threshold: float = MIN_SCORE_THRESHOLD) -> List[Any]:
        """
        Rerank using semantic, lexical matching, and recency.
        Optimized implementation that calculates lexical scores first and
        only runs expensive semantic ranking on promising documents.
        
        Args:
            query: The user query
            documents: List of document objects
            top_k: Number of top documents to return
            min_score_threshold: Minimum score threshold
            
        Returns:
            List of reranked documents
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
            
        logger.info(f"Hybrid reranking {len(documents)} documents for query: '{query}'")
        
        # Get current time for recency calculation
        current_time = datetime.datetime.now().timestamp()
        
        # Calculate cheap lexical and recency scores first (fast pre-filtering)
        query_terms = set(query.lower().split())
        if not query_terms:
            # If no meaningful query terms, use parent implementation
            return super().rerank(query, documents, top_k, min_score_threshold)
        
        # Calculate lexical and recency scores for all documents
        doc_scores = []
        for doc in documents:
            # Extract text using query-focused extraction for better lexical matching
            text = self._extract_query_focused_text(doc, query)
            
            # Calculate lexical score
            lexical_score = 0.0
            if text:
                doc_terms = set(text.lower().split())
                matches = query_terms.intersection(doc_terms)
                lexical_score = len(matches) / max(1, len(query_terms))
                
                # Boost exact phrase matches
                if query.lower() in text.lower():
                    lexical_score = min(1.0, lexical_score * 1.5)
            
            # Calculate recency score
            recency_score = 0.0
            if hasattr(doc, "metadata") and "timestamp" in doc.metadata:
                try:
                    doc_timestamp = float(doc.metadata["timestamp"])
                    days_diff = (current_time - doc_timestamp) / (60 * 60 * 24)
                    recency_score = max(0.0, min(1.0, math.exp(-0.1 * days_diff)))
                except (ValueError, TypeError):
                    pass
            
            # Calculate preliminary score without semantic component
            preliminary_score = (self.lexical_weight * lexical_score + 
                                self.recency_weight * recency_score)
            
            # Store doc with its preliminary score
            doc_scores.append((doc, preliminary_score, lexical_score, recency_score))
        
        # Sort by preliminary score to prioritize promising documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Debug: Print number of documents before semantic processing
        logger.info(f"Documents before semantic filtering: {len(doc_scores)}")
        print(f"Documents before semantic filtering: {len(doc_scores)}")

        # For small document sets, process all; for larger sets, focus on promising ones
        semantic_candidate_count = min(len(documents), max(top_k * 3, 20))

        # Get promising documents for semantic scoring
        semantic_candidates = [item[0] for item in doc_scores[:semantic_candidate_count]]

        # Get semantic scores only for promising documents
        semantic_reranked = super().rerank(
            query, 
            semantic_candidates,
            top_k=len(semantic_candidates),  # Get scores for all candidates
            min_score_threshold=0,  # No filtering yet
            recency_weight=0  # We'll handle recency ourselves
        )

        # Debug: Print number of documents after semantic reranking
        logger.info(f"Documents after semantic reranking: {len(semantic_reranked)}")
        print(f"Documents after semantic reranking: {len(semantic_reranked)}")

        # Create a map from document back to its lexical and recency scores
        score_map = {doc: (ls, rs) for doc, _, ls, rs in doc_scores}

        # Calculate final combined scores
        for doc in semantic_reranked:
            semantic_score = getattr(doc, "semantic_score", 0.5)
            lexical_score, recency_score = score_map.get(doc, (0.0, 0.0))
            
            # Recalculate semantic weight based on other weights
            semantic_weight = 1.0 - (self.lexical_weight + self.recency_weight)
            
            # Calculate combined score
            combined_score = (semantic_weight * semantic_score + 
                             self.lexical_weight * lexical_score + 
                             self.recency_weight * recency_score)
            
            setattr(doc, "lexical_score", lexical_score)
            setattr(doc, "recency_score", recency_score)
            setattr(doc, "rerank_score", combined_score)

        # Sort by combined score
        reranked_docs = sorted(semantic_reranked, key=lambda x: getattr(x, "rerank_score", 0), reverse=True)

        # Debug: Print rerank scores before threshold filtering
        print("Rerank scores before filtering:")
        for i, doc in enumerate(reranked_docs[:min(10, len(reranked_docs))]):
            print(f"  Doc {i}: score={getattr(doc, 'rerank_score', 0):.3f}, semantic={getattr(doc, 'semantic_score', 0):.3f}, lexical={getattr(doc, 'lexical_score', 0):.3f}, recency={getattr(doc, 'recency_score', 0):.3f}")

        # Filter by threshold
        if min_score_threshold > 0:
            before_count = len(reranked_docs)
            reranked_docs = [doc for doc in reranked_docs if getattr(doc, "rerank_score", 0) >= min_score_threshold]
            logger.info(f"Filtered by threshold {min_score_threshold}: {before_count} -> {len(reranked_docs)} docs")
            print(f"Filtered by threshold {min_score_threshold}: {before_count} -> {len(reranked_docs)} docs")

        # If we didn't find enough documents with semantic scoring, add some from lexical matching
        if len(reranked_docs) < top_k and len(doc_scores) > semantic_candidate_count:
            # Get additional documents that weren't semantically scored
            additional_docs = [item[0] for item in doc_scores[semantic_candidate_count:]]
            
            # Set preliminary scores as final scores for these documents
            for doc, prelim_score, lexical_score, recency_score in doc_scores[semantic_candidate_count:]:
                setattr(doc, "semantic_score", 0.4)  # Below-average semantic score
                setattr(doc, "lexical_score", lexical_score)
                setattr(doc, "recency_score", recency_score)
                setattr(doc, "rerank_score", prelim_score)
            
            # Combine and sort all documents
            all_docs = reranked_docs + additional_docs
            reranked_docs = sorted(all_docs, key=lambda x: getattr(x, "rerank_score", 0), reverse=True)
            
            # Debug: Print count after adding lexical matches
            print(f"After adding lexical matches: {len(reranked_docs)} docs")
            
            # Apply threshold again
            if min_score_threshold > 0:
                before_count = len(reranked_docs)
                reranked_docs = [doc for doc in reranked_docs if getattr(doc, "rerank_score", 0) >= min_score_threshold]
                logger.info(f"Filtered again by threshold {min_score_threshold}: {before_count} -> {len(reranked_docs)} docs")
                print(f"Filtered again by threshold {min_score_threshold}: {before_count} -> {len(reranked_docs)} docs")

        logger.info(f"Hybrid reranking complete. Returning top {min(top_k, len(reranked_docs))} results")
        print("reranked_docs", reranked_docs)
        return reranked_docs[:top_k]