import numpy as np
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from google.genai import types
from google import genai
import time
import random
import tiktoken
from typing import List, Optional
from leanworks.setting import EMBEDDING_REQUESTS_PER_MINUTE, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


# Set up logging
logger = logging.getLogger(__name__)

class GoogleEmbedding:
    """
    Class to handle embedding generation using Google's embedding models with rate limiting.
    """
    def __init__(self, api_key: str, requests_per_minute: int = None):
        """
        Initialize the GoogleEmbedding class.
        
        Args:
            api_key: Google API key
            requests_per_minute: Maximum requests per minute (uses setting default if not provided)
        """
        self.embedding_model_client = genai.Client(api_key=api_key)
        self.requests_per_minute = requests_per_minute or EMBEDDING_REQUESTS_PER_MINUTE
        self.min_interval = 60.0 / self.requests_per_minute  # Minimum seconds between requests
        self.last_request_time = 0
        self.model_name = EMBEDDING_MODEL  # Use configured embedding model
        self.tokenizer = tiktoken.get_encoding("o200k_base")  # GPT-4o tokenizer for token counting
        logger.info(f"GoogleEmbedding initialized with model: {self.model_name}, rate limit: {self.requests_per_minute} requests/minute")
    
    def _rate_limit_wait(self):
        """Ensure we don't exceed the rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _validate_and_truncate_text(self, text: str) -> str:
        """
        Validate and truncate text according to API limits.
        
        Args:
            text: Input text to validate
            
        Returns:
            Validated and potentially truncated text
        """
        # Check for empty text
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided, using placeholder")
            return "empty text"
        
        # Tokenize and check length (API limit is 2048 tokens per input)
        tokens = self.tokenizer.encode(text)
        if len(tokens) > 2048:
            logger.warning(f"Text has {len(tokens)} tokens, truncating to 2048 tokens")
            truncated_tokens = tokens[:2048]
            text = self.tokenizer.decode(truncated_tokens)
        
        return text

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str, task_type: str, max_retries: int = 5) -> np.ndarray:
        """
        Generate embedding for input text using Google embedding model with caching and retry logic.
        
        Args:
            text: The text to generate an embedding for (max 2048 tokens)
            task_type: The task type to use for the embedding, can be "RETRIEVAL_QUERY" or "RETRIEVAL_DOCUMENT"
            max_retries: Maximum number of retry attempts for rate limit errors
            
        Returns:
            numpy array containing the embedding vector
        """
        # Validate and truncate text if necessary
        text = self._validate_and_truncate_text(text)
        logger.debug(f"Generating embedding for text of length: {len(text)}")
        
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit_wait()
                
                result = self.embedding_model_client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                return np.array(result.embeddings[0].values)
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RATE_LIMIT_EXCEEDED" in error_str:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        base_delay = 2 ** attempt
                        jitter = random.uniform(0.1, 0.5)
                        delay = base_delay + jitter
                        
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                                     f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limit error: {error_str}")
                        # Return a zero embedding as fallback
                        return np.zeros(768)
                else:
                    # For non-rate-limit errors, don't retry
                    logger.error(f"Error generating embedding: {error_str}")
                    return np.zeros(768)
        
        # Should not reach here, but return zero embedding as fallback
        return np.zeros(768)
    
    def get_embeddings_batch(self, texts: List[str], task_type: str, batch_size: int = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with proper rate limiting.
        
        Args:
            texts: List of texts to generate embeddings for
            task_type: The task type to use for the embedding
            batch_size: Number of requests to process in parallel (uses setting default if not provided)
            
        Returns:
            List of numpy arrays containing the embedding vectors
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        embeddings = []
        
        # Process in smaller batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                embedding = self.get_embedding(text, task_type)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Add a small delay between batches
            if i + batch_size < len(texts):
                time.sleep(1.0)
        
        return embeddings
