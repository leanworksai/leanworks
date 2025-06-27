import numpy as np
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from google.genai import types
from google import genai
# Set up logging
logger = logging.getLogger(__name__)

class GoogleEmbedding:
    """
    Class to handle embedding generation using Google's embedding models.
    """
    def __init__(self, api_key: str):
        """
        Initialize the GoogleEmbedding class.
        
        Args:
            embedding_model_client: Initialized Google embedding model client
        """
        self.embedding_model_client = genai.Client(api_key=api_key)
        logger.info("GoogleEmbedding initialized successfully")
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str, task_type: str) -> np.ndarray:
        """
        Generate embedding for input text using Google embedding model with caching.
        
        Args:
            text: The text to generate an embedding for
            task_type: The task type to use for the embedding, can be "RETRIEVAL_QUERY" or "RETRIEVAL_DOCUMENT"
            
        Returns:
            numpy array containing the embedding vector
        """
        logger.debug(f"Generating embedding for text of length: {len(text)}")
        try:
            result = self.embedding_model_client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return np.array(result.embeddings[0].values)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero embedding as fallback
            return np.zeros(768)  # Standard embedding dimension
