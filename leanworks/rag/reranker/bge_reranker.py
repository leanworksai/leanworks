"""
Highly optimized BGE reranker with ONNX + INT8 quantization and advanced performance tuning.
Implements all performance best practices for maximum CPU efficiency.
"""

import os
import time
import logging
import threading
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

from leanworks.rag.reranker.base_reranker import BaseReranker
from leanworks.setting import BGE_MODEL_NAME, BGE_CACHE_SIZE, BGE_MAX_WORKERS

logger = logging.getLogger(__name__)

class BGEReranker(BaseReranker):
    """
    Highly optimized BGE reranker with ONNX + INT8 quantization.
    
    Performance optimizations:
    1. ONNX + INT8 dynamic quantization (Linear/Gemm ops quantized, embeddings FP32)
    2. Smart batching (24-32 batch size for 300-340 token pairs)
    3. Sequence length trimming to 95th percentile (384 tokens)
    4. Optimized threading (intra-op=6, inter-op=1)
    5. Fast tokenizer with pre-batch tokenization
    """
    
    def __init__(
        self,
        model_name: str = BGE_MODEL_NAME,
        cache_size: int = BGE_CACHE_SIZE,
        max_workers: int = BGE_MAX_WORKERS,
        onnx_cache_dir: Optional[str] = None,
        batch_size: int = 28,  # Optimal for 300-340 token pairs
        max_sequence_length: int = 384,  # 95th percentile instead of 512
        intra_op_threads: int = 6,
        inter_op_threads: int = 1
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        
        # Cache directories
        self.onnx_cache_dir = onnx_cache_dir if onnx_cache_dir else str(Path.home() / ".cache" / "leanworks_optimized_onnx")
        self.quantized_cache_dir = str(Path(self.onnx_cache_dir) / "quantized")
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._session = None
        self._model_lock = threading.Lock()
        
        # Caching
        self._score_cache = {}
        self._text_cache = {}
        self._token_cache = {}  # Cache tokenized inputs
        self._cache_lock = threading.Lock()
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self._tokenization_times = []
        self._inference_times = []
        
        logger.info(f"BGEOptimizedReranker initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max sequence length: {max_sequence_length}")
        logger.info(f"  Threading: intra_op={intra_op_threads}, inter_op={inter_op_threads}")
    
    def _setup_threading_environment(self):
        """Set up optimal threading environment variables."""
        os.environ["OMP_NUM_THREADS"] = str(self.intra_op_threads)
        os.environ["OMP_PROC_BIND"] = "TRUE"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.intra_op_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.intra_op_threads)
        logger.info(f"Threading environment configured: OMP_NUM_THREADS={self.intra_op_threads}")
    
    def _create_quantization_config(self) -> AutoQuantizationConfig:
        """Create INT8 dynamic quantization config."""
        # Quantize Linear/Gemm operations, keep embeddings as FP32
        quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False)
        
        # Custom configuration for BGE model
        quantization_config.operators_to_quantize = ["MatMul", "Gemm", "Add"]
        quantization_config.per_channel = True
        quantization_config.reduce_range = True
        
        # Keep embeddings in FP32 for better accuracy
        quantization_config.nodes_to_exclude = [
            "embeddings", "LayerNorm", "Softmax", "Gelu"
        ]
        
        logger.info("Created INT8 dynamic quantization config")
        return quantization_config
    
    def _load_model(self):
        """Load and optimize the ONNX model with INT8 quantization."""
        if self._model is not None and self._session is not None:
            return
            
        with self._model_lock:
            if self._model is not None and self._session is not None:
                return
            
            logger.info(f"Loading optimized ONNX model: {self.model_name}")
            
            # Setup threading environment
            self._setup_threading_environment()
            
            try:
                # Load fast tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,  # Use Rust-based fast tokenizer
                    model_max_length=self.max_sequence_length,
                    padding_side="right",
                    truncation_side="right"
                )
                logger.info(f"Fast tokenizer loaded (is_fast: {self._tokenizer.is_fast})")
                
                # Check for quantized model
                quantized_model_path = Path(self.quantized_cache_dir) / "model_quantized.onnx"
                
                if quantized_model_path.exists():
                    logger.info("Loading cached quantized ONNX model")
                    self._load_quantized_model(str(quantized_model_path))
                else:
                    logger.info("Creating and quantizing ONNX model (first time setup)")
                    self._create_and_quantize_model()
                
                logger.info("Optimized ONNX model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load optimized ONNX model: {str(e)}")
                raise
    
    def _create_and_quantize_model(self):
        """Create ONNX model and apply INT8 quantization."""
        # Create directories
        os.makedirs(self.onnx_cache_dir, exist_ok=True)
        os.makedirs(self.quantized_cache_dir, exist_ok=True)
        
        # Step 1: Export to ONNX
        logger.info("Exporting PyTorch model to ONNX...")
        onnx_model_dir = Path(self.onnx_cache_dir) / "base_onnx"
        
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_name,
            export=True,
            provider="CPUExecutionProvider"
        )
        ort_model.save_pretrained(str(onnx_model_dir))
        
        # Step 2: Apply INT8 quantization
        logger.info("Applying INT8 dynamic quantization...")
        quantization_config = self._create_quantization_config()
        
        quantizer = ORTQuantizer.from_pretrained(str(onnx_model_dir))
        quantizer.quantize(
            save_dir=self.quantized_cache_dir,
            quantization_config=quantization_config
        )
        
        # Step 3: Load quantized model
        quantized_model_path = Path(self.quantized_cache_dir) / "model_quantized.onnx"
        self._load_quantized_model(str(quantized_model_path))
        
        logger.info(f"Model quantized and cached at: {self.quantized_cache_dir}")
    
    def _load_quantized_model(self, model_path: str):
        """Load the quantized ONNX model with optimized session options."""
        # Configure session options for optimal CPU performance
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.intra_op_threads
        session_options.inter_op_num_threads = self.inter_op_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # CPU-specific optimizations (removed conflicting disable_cpu_ep_fallback)
        session_options.add_session_config_entry("session.use_env_allocators", "1")
        session_options.add_session_config_entry("session.use_deterministic_compute", "0")
        
        # Load the session with CPU provider
        providers = ["CPUExecutionProvider"]
        provider_options = [{
            "use_arena": True,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cpu_mem_arena": True,
            "use_gemm_conv_optimization": True
        }]
        
        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options
        )
        
        logger.info(f"Quantized ONNX session created with {self.intra_op_threads} intra-op threads")
    
    def _preprocess_and_cache_tokens(self, texts: List[str], query: str) -> List[dict]:
        """Pre-process and cache tokenized inputs to avoid Python overhead."""
        cache_key = f"{hash(query)}_{hash(tuple(texts))}"
        
        with self._cache_lock:
            if cache_key in self._token_cache:
                return self._token_cache[cache_key]
        
        # Fast batch tokenization
        start_time = time.time()
        
        # Create query-document pairs
        pairs = [(query, text) for text in texts]
        
        # Batch tokenize all pairs at once (major performance win)
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="np",
            return_attention_mask=True
        )
        
        tokenization_time = time.time() - start_time
        self._tokenization_times.append(tokenization_time)
        
        # Convert to list of dicts for ONNX input
        batch_inputs = []
        for i in range(len(pairs)):
            batch_inputs.append({
                "input_ids": encoded["input_ids"][i:i+1],
                "attention_mask": encoded["attention_mask"][i:i+1]
            })
        
        # Cache the tokenized inputs
        with self._cache_lock:
            if len(self._token_cache) >= self.cache_size // 4:
                # Clear oldest 25% of cache
                keys_to_remove = list(self._token_cache.keys())[:len(self._token_cache) // 4]
                for k in keys_to_remove:
                    self._token_cache.pop(k, None)
            
            self._token_cache[cache_key] = batch_inputs
        
        logger.debug(f"Tokenized {len(pairs)} pairs in {tokenization_time*1000:.1f}ms")
        return batch_inputs
    
    def _compute_similarity_scores_optimized(self, query: str, texts: List[str]) -> List[float]:
        """Compute similarity scores with all optimizations."""
        self._load_model()
        
        if not texts:
            return []
        
        # Pre-process and cache tokenized inputs
        tokenized_inputs = self._preprocess_and_cache_tokens(texts, query)
        
        # Smart batching for optimal performance
        all_scores = []
        
        start_time = time.time()
        
        for i in range(0, len(tokenized_inputs), self.batch_size):
            batch_inputs = tokenized_inputs[i:i + self.batch_size]
            
            # Prepare batch for ONNX
            if len(batch_inputs) == 1:
                # Single input
                onnx_inputs = {
                    "input_ids": batch_inputs[0]["input_ids"],
                    "attention_mask": batch_inputs[0]["attention_mask"]
                }
            else:
                # Batch inputs
                input_ids = np.vstack([inp["input_ids"] for inp in batch_inputs])
                attention_mask = np.vstack([inp["attention_mask"] for inp in batch_inputs])
                
                onnx_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            
            # Run inference
            outputs = self._session.run(None, onnx_inputs)
            logits = outputs[0]  # Shape: (batch_size, num_classes)
            
            # Extract scores (assuming binary classification with positive class at index 1)
            if logits.shape[1] == 2:
                batch_scores = logits[:, 1].tolist()  # Positive class scores
            else:
                batch_scores = logits[:, 0].tolist()  # Single output
            
            all_scores.extend(batch_scores)
        
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)
        
        logger.debug(f"Inference for {len(texts)} texts: {inference_time*1000:.1f}ms")
        
        return all_scores
    
    async def _score_documents_async(self, query: str, documents: List[str]) -> List[float]:
        """Async scoring with optimized implementation."""
        loop = asyncio.get_event_loop()
        
        # Run the optimized scoring in thread pool
        scores = await loop.run_in_executor(
            self._executor,
            self._compute_similarity_scores_optimized,
            query,
            documents
        )
        
        return scores
    
    def rerank(self, query: str, documents: List[Any], top_k: int = 10, **kwargs) -> List[Any]:
        """Synchronous reranking with all optimizations."""
        if not documents:
            return []
        
        # Extract full text from documents
        doc_texts = []
        for doc in documents:
            text = doc.metadata.get("chunk_text", "")
            doc_texts.append(text)
        
        # Get scores
        scores = self._compute_similarity_scores_optimized(query, doc_texts)
        
        # Combine documents with scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.rerank_score = score
            scored_docs.append(doc)
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Return top_k
        return scored_docs[:top_k]

    async def rerank_async(self, query: str, documents: List[Any], top_k: int = 10, **kwargs) -> List[Any]:
        """Async reranking with all optimizations."""
        if not documents:
            return []
        
        # Extract full text from documents
        doc_texts = []
        for doc in documents:
            text = doc.metadata.get("chunk_text", "")
            doc_texts.append(text)
        
        # Get scores
        scores = await self._score_documents_async(query, doc_texts)
        
        # Combine documents with scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.rerank_score = score
            scored_docs.append(doc)
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Return top_k
        return scored_docs[:top_k]
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            "total_tokenization_calls": len(self._tokenization_times),
            "total_inference_calls": len(self._inference_times),
            "avg_tokenization_ms": np.mean(self._tokenization_times) * 1000 if self._tokenization_times else 0,
            "avg_inference_ms": np.mean(self._inference_times) * 1000 if self._inference_times else 0,
            "cache_sizes": {
                "score_cache": len(self._score_cache),
                "text_cache": len(self._text_cache),
                "token_cache": len(self._token_cache)
            },
            "configuration": {
                "batch_size": self.batch_size,
                "max_sequence_length": self.max_sequence_length,
                "intra_op_threads": self.intra_op_threads,
                "inter_op_threads": self.inter_op_threads
            }
        }
        
        return stats
    
    def clear_performance_stats(self):
        """Clear performance tracking data."""
        self._tokenization_times.clear()
        self._inference_times.clear()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
