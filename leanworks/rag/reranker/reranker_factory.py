from typing import Optional
import logging
from leanworks.rag.reranker.base_reranker import BaseReranker
from leanworks.rag.reranker.llm_reranker import CrossEncoderReranker
from leanworks.setting import (
    RERANKER_TYPE, BGE_MODEL_NAME, BGE_DEVICE, BGE_MAX_WORKERS, BGE_CACHE_SIZE,
    BGE_MAX_LENGTH, BGE_BATCH_SIZE, BGE_INTRA_OP_THREADS, BGE_INTER_OP_THREADS
)

logger = logging.getLogger(__name__)

class RerankerFactory:
    """
    Factory class for creating different types of rerankers based on configuration.
    """
    
    @staticmethod
    def create_reranker(
        reranker_type: str = RERANKER_TYPE,
        model_client: Optional[object] = None,
        **kwargs
    ) -> BaseReranker:
        """
        Create a reranker instance based on the specified type.
        
        Args:
            reranker_type: Type of reranker to create ("llm" or "bge")
            model_client: Model client for LLM-based reranker (required for "llm" type)
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseReranker instance
            
        Raises:
            ValueError: If reranker type is unsupported or required parameters are missing
        """
        reranker_type = reranker_type.lower()
        
        if reranker_type == "llm":
            if model_client is None:
                raise ValueError("model_client is required for LLM-based reranker")
            
            cache_size = kwargs.get("cache_size", 1000)
            max_concurrent_requests = kwargs.get("max_concurrent_requests", 3)
            
            logger.info("Creating CrossEncoderReranker (LLM-based)")
            return CrossEncoderReranker(
                model_client=model_client,
                cache_size=cache_size,
                max_concurrent_requests=max_concurrent_requests
            )
            
        elif reranker_type == "bge":
            model_name = kwargs.get("model_name", BGE_MODEL_NAME)
            cache_size = kwargs.get("cache_size", BGE_CACHE_SIZE)
            max_workers = kwargs.get("max_workers", BGE_MAX_WORKERS)
            batch_size = kwargs.get("batch_size", BGE_BATCH_SIZE)
            max_sequence_length = kwargs.get("max_sequence_length", BGE_MAX_LENGTH)
            intra_op_threads = kwargs.get("intra_op_threads", BGE_INTRA_OP_THREADS)
            inter_op_threads = kwargs.get("inter_op_threads", BGE_INTER_OP_THREADS)
            
            logger.info(f"Creating BGEOptimizedReranker (default BGE) with model: {model_name}")
            from leanworks.rag.reranker.bge_reranker import BGEReranker

            return BGEReranker(
                model_name=model_name,
                cache_size=cache_size,
                max_workers=max_workers,
                batch_size=batch_size,
                max_sequence_length=max_sequence_length,
                intra_op_threads=intra_op_threads,
                inter_op_threads=inter_op_threads
            )
        else:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")
    
    @staticmethod
    def get_available_types() -> list:
        """
        Get list of available reranker types.
        
        Returns:
            List of supported reranker type strings
        """
        return ["llm", "bge", "bge-onnx", "onnx", "bge-optimized", "optimized"]
