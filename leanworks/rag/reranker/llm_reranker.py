from typing import List, Any
import json
import logging
import datetime
import math
import time
import asyncio
import os
from leanworks.setting import *
from leanworks.rag.reranker.base_reranker import BaseReranker
from leanworks.rag.reranker.rate_limiter import (
    DualRateLimiter, 
    _approx_token_count, 
    _estimate_prompt_tokens, 
    _adaptive_batch_size, 
    _jitter_backoff
)
import random
# Set up logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker module that scores query-document pairs directly.
    This provides more accurate semantic matching than vector similarity alone,
    especially for queries with minimal context.
    """
    def __init__(self, model_client, max_concurrent_requests: int = 10):
        """
        Initialize the reranker with a model client.
        
        Args:
            model_client: The model client to use for scoring (e.g., OpenAI client)
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        super().__init__()
        self.model_client = model_client
        self._max_retries = 3
        self._base_delay = 1.0  # Base delay for exponential backoff
        self._max_concurrent_requests = max_concurrent_requests
        
        # Rate limiting configuration - Claude Haiku 3 limits
        self._rpm = int(os.getenv("RERANK_RPM", "1000"))     # Claude Haiku 3: 1,000 RPM
        self._tpm = int(os.getenv("RERANK_TPM", "100000"))   # Claude Haiku 3: 100,000 input TPM
        self._limiter = DualRateLimiter(rpm=self._rpm, tpm=self._tpm, burst_requests=self._rpm, burst_tokens=self._tpm)
        self._hard_cap_batch = int(os.getenv("RERANK_BATCH_CAP", "30"))   # optimal for Claude Haiku 3
        self._target_latency = float(os.getenv("RERANK_TARGET_LATENCY_S", "2.0"))
        self._inflight = asyncio.Semaphore(self._max_concurrent_requests)  # one global semaphore only
        self._doc_token_cap = int(os.getenv("RERANK_DOC_TOKEN_CAP", "280"))  # per-doc cap
        self._prompt_overhead = 80  # system+formatting budget, keep it tiny
        
        logger.info(f"CrossEncoderReranker initialized with RPM={self._rpm}, TPM={self._tpm}, batch_cap={self._hard_cap_batch}")
    
    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        # quick char-based fall-back; replace with tiktoken-aware truncation if possible
        approx_chars = max_tokens * 4
        if len(text) <= approx_chars:
            return " ".join(text.split())
        return " ".join(text[:approx_chars].split()) + " …"
    
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
        
        logger.info(f"Async reranking {len(documents)} documents for query (length: {len(query)} chars)")
        
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
        Uses token-aware adaptive batching and rate limiting for optimal performance.
        
        Args:
            query: The user query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        # Token-cap the documents to keep batches predictable
        capped_docs = [self._truncate_by_tokens(doc, self._doc_token_cap) for doc in documents]
        
        # Process documents with adaptive batching
        remaining = capped_docs[:]  # list of strings (already query-focused)
        scores = [0.0] * len(remaining)
        
        idx_map = list(range(len(remaining)))  # track original positions
        while remaining:
            avg_doc_tokens = int(sum(_approx_token_count(d) for d in remaining) / len(remaining))
            bsz = _adaptive_batch_size(
                docs_remaining=len(remaining),
                avg_doc_tokens=avg_doc_tokens,
                limiter=self._limiter,
                hard_cap=self._hard_cap_batch,
                overhead_tokens=self._prompt_overhead,
                target_latency_s=self._target_latency,
            )
            batch_docs = remaining[:bsz]
            batch_idx = idx_map[:bsz]
            del remaining[:bsz], idx_map[:bsz]
            
            est_tokens = _estimate_prompt_tokens(query, batch_docs, overhead_tokens=self._prompt_overhead)
            
            # unified concurrency + token-aware gating
            async with self._inflight:
                await self._limiter.acquire(est_tokens)
                batch_scores = await self._batch_score_documents_async(query, batch_docs)
            
            for i, s in zip(batch_idx, batch_scores):
                scores[i] = s
        
        return scores

    async def _batch_score_documents_async(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a batch of documents using the model with structured output and header awareness.
        
        Args:
            query: The user query
            documents: List of document texts (already token-capped)
            
        Returns:
            List of relevance scores for the batch
        """
        # Handle empty batch
        if not documents:
            return []
        
        # Get model response with retry logic
        for attempt in range(self._max_retries):
            try:
                logger.info(f"Requesting scores for {len(documents)} documents (attempt {attempt + 1})")
                
                # Use a more robust approach with better prompting
                prompt = f"Rate the relevance of each document to the query on a scale of 0 to 10, where 10 is extremely relevant and 0 is completely irrelevant.\n\n"
                prompt += f"Query: {query}\n\n"
                
                for i, doc in enumerate(documents):
                    prompt += f"Document {i+1}: {doc}\n\n"
                
                prompt += f"\nYou must return EXACTLY {len(documents)} scores, one for each document, in the format: Score1, Score2, Score3, ...\n"
                if len(documents) <= 4:
                    example_scores = ", ".join([str(i+5) for i in range(len(documents))])
                    prompt += f"Example response format for {len(documents)} documents: {example_scores}\n"
                else:
                    prompt += "Example response format: 8, 7, 9, 5, 6, 3, 8, 4\n"
                prompt += f"CRITICAL: Your response must contain exactly {len(documents)} comma-separated numbers. No more, no less.\n"
                prompt += "DO NOT include any other text or explanation in your response."
                
                # Check if this is an Anthropic client (has messages attribute) or OpenAI client
                if hasattr(self.model_client, 'messages'):
                    # Anthropic client
                    response = self.model_client.messages.create(
                        model=RERANK_MODEL,
                        max_tokens=256,
                        temperature=0.0,  # Use deterministic output
                        system="You are a document ranking assistant. Your task is to rate how relevant each document is to the given query on a scale of 0-10. You must respond ONLY with comma-separated numerical scores (e.g., '8, 7, 9, 6'). Do not include any other text, explanations, or formatting in your response.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.content[0].text
                else:
                    # OpenAI client (fallback)
                    response = self.model_client.chat.completions.create(
                        model=RERANK_MODEL,
                        max_tokens=256,
                        temperature=0.0,  # Use deterministic output
                        messages=[
                            {"role": "system", "content": "You are a document ranking assistant. Your task is to rate how relevant each document is to the given query on a scale of 0-10. You must respond ONLY with comma-separated numerical scores (e.g., '8, 7, 9, 6'). Do not include any other text, explanations, or formatting in your response."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content
                
                # Parse scores from response
                logger.debug(f"Raw reranking response received (length: {len(answer) if answer else 0} chars)")
                
                # Quick validation: count commas to estimate number of scores
                if answer and "," in answer:
                    comma_count = answer.count(",")
                    expected_commas = len(documents) - 1  # n scores = n-1 commas
                    if comma_count != expected_commas:
                        logger.debug(f"Response has {comma_count} commas, expected {expected_commas} for {len(documents)} scores")
                
                # Parse and process scores using the robust parser
                scores = await self._parse_scores_async(answer, len(documents))
                
                # (Optional) If your SDK exposes headers, feed them:
                try:
                    self._limiter.observe_headers(getattr(response, "response", {}).get("headers", {}))
                except Exception:
                    pass
                
                return scores
                    
            except Exception as e:
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    wait_time = _jitter_backoff(self._base_delay, 60.0, attempt)
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
            
            # Method 0: Try JSON parsing first (in case model returns JSON despite instructions)
            try:
                import re
                # Look for JSON-like structure in the response
                json_match = re.search(r'\{[^}]*"scores"[^}]*\[([^\]]+)\][^}]*\}', response_text, re.DOTALL)
                if json_match:
                    scores_str = json_match.group(1)
                    # Parse the scores array content
                    score_values = re.findall(r'(\d+(?:\.\d+)?)', scores_str)
                    if score_values:
                        scores = [float(val) for val in score_values]
            except Exception:
                pass
            
            # Method 1: Try comma-separated values (most common format)
            if len(scores) != expected_count and "," in response_text:
                scores = []
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
                import re
                all_numbers = re.findall(r'\d+(?:\.\d+)?', response_text)
                if all_numbers:
                    scores = [float(num) for num in all_numbers]
            
            # Verify we have the expected number of scores
            if len(scores) > expected_count:
                # Too many scores, trim to expected number
                logger.warning(f"Found {len(scores)} scores, trimming to {expected_count}")
                scores = scores[:expected_count]
            
            # Normalize to ensure we have correct number of scores
            if len(scores) < expected_count:
                # Check if we're only missing one score and the response might be truncated
                if len(scores) == expected_count - 1:
                    logger.info(f"Model returned {len(scores)} scores instead of {expected_count}. This is likely due to the model missing one document or response truncation. Adding default score.")
                else:
                    logger.warning(f"Failed to parse correct number of scores from response (length: {len(response_text)} chars). Found {len(scores)}, expected {expected_count}")
                # Use found scores and fill rest with default
                scores = scores + [5.0] * (expected_count - len(scores))
            
            # Ensure scores are in the valid range
            scores = [max(0.0, min(10.0, score)) for score in scores]
            
            # Convert scores to 0-1 range
            return [score / 10.0 for score in scores]
            
        except Exception as e:
            logger.warning(f"Failed to parse scores from response (length: {len(response_text)} chars). Error: {str(e)}")
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

