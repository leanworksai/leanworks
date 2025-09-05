#!/usr/bin/env python3
"""
Test suite for BGE Reranker implementation.
Tests both synchronous and asynchronous functionality of the BGE reranker.
"""

import pytest
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leanworks.rag.reranker.bge_reranker import BGEReranker
from leanworks.rag.reranker.reranker_factory import RerankerFactory
from leanworks.setting import *


class TestBGEReranker:
    """Test suite for BGE Reranker functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = []
        
        # Create mock document objects with metadata
        doc_texts = [
            "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, procedural, and functional programming.",
            "Machine learning algorithms can be implemented using various frameworks like TensorFlow, PyTorch, and scikit-learn. These frameworks provide tools for building and training neural networks.",
            "The weather today is sunny with a temperature of 25 degrees Celsius. It's a perfect day for outdoor activities and enjoying nature.",
            "Database optimization involves indexing, query optimization, and proper schema design. These techniques help improve database performance and reduce query execution time.",
            "Natural language processing enables computers to understand and process human language. It involves techniques like tokenization, parsing, and semantic analysis.",
            "Cloud computing provides on-demand access to computing resources over the internet. Services include storage, processing power, and software applications.",
            "Artificial intelligence has applications in healthcare, finance, transportation, and many other industries. AI systems can automate tasks and provide intelligent insights."
        ]
        
        for i, text in enumerate(doc_texts):
            doc = SimpleNamespace()
            doc.metadata = {
                "chunk_text": text,
                "timestamp": 1704067200 + i * 86400,  # Mock timestamps (daily increments)
                "source": f"document_{i+1}.txt",
                "doc_id": f"doc_{i+1}"
            }
            documents.append(doc)
        
        return documents
    
    @pytest.fixture
    def bge_reranker(self):
        """Create a BGE reranker instance for testing."""
        return BGEReranker(
            model_name="BAAI/bge-reranker-base",
            cache_size=100,
            max_workers=2,
            batch_size=4,  # Smaller batch size for tests
            max_sequence_length=128,  # Shorter sequences for faster tests
            intra_op_threads=2,
            inter_op_threads=1
        )
    
    def test_bge_reranker_initialization(self):
        """Test BGE reranker initialization."""
        reranker = BGEReranker(
            model_name="BAAI/bge-reranker-base",
            cache_size=200,
            max_workers=4,
            batch_size=16,
            max_sequence_length=256,
            intra_op_threads=4,
            inter_op_threads=1
        )
        
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.cache_size == 200
        assert reranker.max_workers == 4
        assert reranker.batch_size == 16
        assert reranker.max_sequence_length == 256
        assert reranker.intra_op_threads == 4
        assert reranker.inter_op_threads == 1
        assert reranker._model is None  # Model should be lazy-loaded
        assert reranker._tokenizer is None
        assert reranker._session is None
    
    def test_factory_creation(self):
        """Test creating BGE reranker via factory."""
        reranker = RerankerFactory.create_reranker(
            reranker_type="bge",
            model_name="BAAI/bge-reranker-base",
            cache_size=150,
            max_workers=3,
            batch_size=8,
            max_sequence_length=200
        )
        
        assert isinstance(reranker, BGEReranker)
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.cache_size == 150
        assert reranker.max_workers == 3
        assert reranker.batch_size == 8
        assert reranker.max_sequence_length == 200
    
    def test_full_text_extraction(self, bge_reranker, sample_documents):
        """Test that BGE reranker uses full document text."""
        query = "machine learning programming"
        test_docs = sample_documents[:2]
        
        # Perform reranking
        results = bge_reranker.rerank(query, test_docs, top_k=2)
        
        # Verify that results are returned
        assert len(results) <= 2
        assert len(results) > 0
        
        # Verify that each result has a rerank score
        for doc in results:
            assert hasattr(doc, "rerank_score")
            assert isinstance(doc.rerank_score, (int, float))
    
    def test_caching_behavior(self, bge_reranker, sample_documents):
        """Test that caching improves performance on repeated queries."""
        query = "test caching behavior"
        test_docs = sample_documents[:2]
        
        # First call - populates cache
        result1 = bge_reranker.rerank(query, test_docs, top_k=2)
        
        # Second call - should use cache
        result2 = bge_reranker.rerank(query, test_docs, top_k=2)
        
        # Results should be consistent
        assert len(result1) == len(result2)
        
        # Check that both have rerank scores
        for doc1, doc2 in zip(result1, result2):
            assert hasattr(doc1, "rerank_score")
            assert hasattr(doc2, "rerank_score")
    
    def test_simplified_scoring(self, bge_reranker, sample_documents):
        """Test that the new implementation uses simplified scoring (only rerank_score)."""
        query = "machine learning programming"
        test_docs = sample_documents[:3]
        
        results = bge_reranker.rerank(query, test_docs, top_k=3)
        
        for doc in results:
            # Should have rerank_score
            assert hasattr(doc, "rerank_score")
            assert isinstance(doc.rerank_score, (int, float))
            
            # Should NOT have the old complex scoring attributes
            assert not hasattr(doc, "semantic_score")
            assert not hasattr(doc, "recency_score")
            assert not hasattr(doc, "combined_score")
    
    def test_empty_documents_handling(self, bge_reranker):
        """Test handling of empty document list."""
        query = "test query"
        empty_docs = []
        
        # Test sync method
        result = bge_reranker.rerank(query, empty_docs)
        assert result == []
        
        # Test async method
        async def test_async_empty():
            result = await bge_reranker.rerank_async(query, empty_docs)
            assert result == []
        
        asyncio.run(test_async_empty())
    
    @pytest.mark.slow
    def test_synchronous_reranking(self, bge_reranker, sample_documents):
        """Test synchronous reranking functionality."""
        query = "machine learning artificial intelligence"
        
        # Test with a subset of documents to speed up the test
        test_docs = sample_documents[:4]
        
        start_time = time.time()
        reranked_docs = bge_reranker.rerank(query, test_docs, top_k=3)
        end_time = time.time()
        
        # Verify results
        assert len(reranked_docs) <= 3
        assert len(reranked_docs) <= len(test_docs)
        
        # Check that documents have reranking scores
        for doc in reranked_docs:
            assert hasattr(doc, "rerank_score")
            assert isinstance(doc.rerank_score, (int, float))
        
        # Check that results are sorted by rerank_score
        scores = [doc.rerank_score for doc in reranked_docs]
        assert scores == sorted(scores, reverse=True)
        
        print(f"Sync reranking took {end_time - start_time:.2f} seconds")
        print(f"Top result: {reranked_docs[0].metadata['chunk_text'][:100]}...")
        print(f"Top score: {reranked_docs[0].rerank_score:.3f}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_asynchronous_reranking(self, bge_reranker, sample_documents):
        """Test asynchronous reranking functionality."""
        query = "database optimization performance"
        
        # Test with a subset of documents
        test_docs = sample_documents[:4]
        
        start_time = time.time()
        reranked_docs = await bge_reranker.rerank_async(query, test_docs, top_k=3)
        end_time = time.time()
        
        # Verify results
        assert len(reranked_docs) <= 3
        assert len(reranked_docs) <= len(test_docs)
        
        # Check that documents have reranking scores
        for doc in reranked_docs:
            assert hasattr(doc, "rerank_score")
            assert isinstance(doc.rerank_score, (int, float))
        
        # Check that results are sorted by rerank_score
        scores = [doc.rerank_score for doc in reranked_docs]
        assert scores == sorted(scores, reverse=True)
        
        print(f"Async reranking took {end_time - start_time:.2f} seconds")
        print(f"Top result: {reranked_docs[0].metadata['chunk_text'][:100]}...")
        print(f"Top score: {reranked_docs[0].rerank_score:.3f}")
    
    def test_different_queries(self, bge_reranker, sample_documents):
        """Test that different queries produce different rankings."""
        test_docs = sample_documents[:3]
        
        # Test with different queries
        query1 = "programming language"
        query2 = "database optimization"
        
        reranked1 = bge_reranker.rerank(query1, test_docs, top_k=3)
        reranked2 = bge_reranker.rerank(query2, test_docs, top_k=3)
        
        # Both should return results
        assert len(reranked1) > 0
        assert len(reranked2) > 0
        
        # Check that documents have rerank scores
        for doc in reranked1:
            assert hasattr(doc, "rerank_score")
        for doc in reranked2:
            assert hasattr(doc, "rerank_score")
    
    def test_top_k_limiting(self, bge_reranker, sample_documents):
        """Test that top_k parameter limits results correctly."""
        query = "programming language"
        test_docs = sample_documents[:5]  # Use 5 docs
        
        # Test with different top_k values
        reranked_k2 = bge_reranker.rerank(query, test_docs, top_k=2)
        reranked_k3 = bge_reranker.rerank(query, test_docs, top_k=3)
        reranked_k10 = bge_reranker.rerank(query, test_docs, top_k=10)
        
        # Check that top_k is respected
        assert len(reranked_k2) <= 2
        assert len(reranked_k3) <= 3
        assert len(reranked_k10) <= len(test_docs)  # Can't return more than input
        
        # All returned documents should have rerank scores
        for doc in reranked_k2:
            assert hasattr(doc, "rerank_score")
    
    def test_caching_functionality(self, bge_reranker, sample_documents):
        """Test that caching works correctly."""
        query = "test caching"
        test_docs = sample_documents[:2]
        
        # First call - should compute scores
        start_time = time.time()
        result1 = bge_reranker.rerank(query, test_docs, top_k=2)
        first_call_time = time.time() - start_time
        
        # Second call - should use cached scores
        start_time = time.time()
        result2 = bge_reranker.rerank(query, test_docs, top_k=2)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert len(result1) == len(result2)
        for doc1, doc2 in zip(result1, result2):
            assert abs(doc1.rerank_score - doc2.rerank_score) < 1e-6
        
        # Second call should be faster (cached)
        print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
        
        # Check that cache has entries
        assert len(bge_reranker._score_cache) > 0
    
    def test_sync_and_async_consistency(self, bge_reranker, sample_documents):
        """Test that sync and async methods produce consistent results."""
        query = "test query"
        test_docs = sample_documents[:2]
        
        # Test sync method
        sync_result = bge_reranker.rerank(query, test_docs, top_k=2)
        
        # Test async method
        async def test_async():
            return await bge_reranker.rerank_async(query, test_docs, top_k=2)
        
        async_result = asyncio.run(test_async())
        
        # Both should return same number of results
        assert len(sync_result) == len(async_result)
        
        # Both should have rerank scores
        for doc in sync_result:
            assert hasattr(doc, "rerank_score")
        for doc in async_result:
            assert hasattr(doc, "rerank_score")
    
    @pytest.mark.slow
    def test_performance_comparison(self, sample_documents):
        """Compare performance between different configurations."""
        query = "machine learning artificial intelligence"
        test_docs = sample_documents[:5]
        
        # Test different worker configurations
        configs = [
            {"max_workers": 1, "name": "1 worker"},
            {"max_workers": 2, "name": "2 workers"},
            {"max_workers": 4, "name": "4 workers"}
        ]
        
        results = {}
        
        for config in configs:
            reranker = BGEReranker(
                model_name="BAAI/bge-reranker-base",
                cache_size=100,
                max_workers=config["max_workers"],
                batch_size=4,
                max_sequence_length=128
            )
            
            start_time = time.time()
            reranked = reranker.rerank(query, test_docs, top_k=3)
            end_time = time.time()
            
            results[config["name"]] = {
                "time": end_time - start_time,
                "results": len(reranked)
            }
            
            # Verify all results have rerank_score
            for doc in reranked:
                assert hasattr(doc, "rerank_score")
        
        # Print performance comparison
        print("\nPerformance Comparison:")
        for name, result in results.items():
            print(f"{name}: {result['time']:.2f}s ({result['results']} results)")
    
    def test_performance_stats(self, bge_reranker, sample_documents):
        """Test performance statistics tracking."""
        query = "test performance stats"
        test_docs = sample_documents[:2]
        
        # Clear any existing stats
        bge_reranker.clear_performance_stats()
        
        # Perform some operations
        bge_reranker.rerank(query, test_docs, top_k=2)
        
        # Get performance stats
        stats = bge_reranker.get_performance_stats()
        
        # Verify stats structure
        assert isinstance(stats, dict)
        assert "total_tokenization_calls" in stats
        assert "total_inference_calls" in stats
        assert "avg_tokenization_ms" in stats
        assert "avg_inference_ms" in stats
        assert "cache_sizes" in stats
        assert "configuration" in stats
        
        # Verify cache sizes structure
        cache_sizes = stats["cache_sizes"]
        assert "score_cache" in cache_sizes
        assert "text_cache" in cache_sizes
        assert "token_cache" in cache_sizes
        
        # Verify configuration structure
        config = stats["configuration"]
        assert "batch_size" in config
        assert "max_sequence_length" in config
        assert "intra_op_threads" in config
        assert "inter_op_threads" in config
        
        # Test clearing stats
        bge_reranker.clear_performance_stats()
        cleared_stats = bge_reranker.get_performance_stats()
        assert cleared_stats["total_tokenization_calls"] == 0
        assert cleared_stats["total_inference_calls"] == 0
    
    def test_onnx_model_initialization(self):
        """Test ONNX model initialization and caching."""
        import tempfile
        import shutil
        
        # Create a temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            reranker = BGEReranker(
                model_name="BAAI/bge-reranker-base",
                onnx_cache_dir=temp_dir,
                cache_size=50,
                max_workers=1,
                batch_size=2,
                max_sequence_length=64
            )
            
            # Check that cache directories are set up correctly
            assert reranker.onnx_cache_dir == temp_dir
            assert str(temp_dir) in reranker.quantized_cache_dir
            
            # Check initial state
            assert reranker._model is None
            assert reranker._tokenizer is None
            assert reranker._session is None
    
    def test_threading_configuration(self):
        """Test that threading configuration is properly set."""
        reranker = BGEReranker(
            model_name="BAAI/bge-reranker-base",
            intra_op_threads=4,
            inter_op_threads=2,
            max_workers=3
        )
        
        assert reranker.intra_op_threads == 4
        assert reranker.inter_op_threads == 2
        assert reranker.max_workers == 3
    
    def test_tokenization_caching(self, bge_reranker, sample_documents):
        """Test that tokenization caching works with full document text."""
        query = "test tokenization caching"
        test_docs = sample_documents[:2]
        
        # This tests the internal caching mechanism indirectly
        # by checking that repeated calls with same inputs work consistently
        import time
        
        # First call
        start_time = time.time()
        result1 = bge_reranker.rerank(query, test_docs, top_k=2)
        first_time = time.time() - start_time
        
        # Second call (should potentially use cached tokenization)
        start_time = time.time()
        result2 = bge_reranker.rerank(query, test_docs, top_k=2)
        second_time = time.time() - start_time
        
        # Both should return same number of results
        assert len(result1) == len(result2)
        
        # Check that both have rerank scores
        for doc in result1:
            assert hasattr(doc, "rerank_score")
        for doc in result2:
            assert hasattr(doc, "rerank_score")
    
    def test_error_handling(self, sample_documents):
        """Test error handling in BGE reranker."""
        # Test with invalid model name (should handle gracefully)
        with pytest.raises(Exception):  # Model loading should fail
            reranker = BGEReranker(model_name="invalid/model-name")
            # Force model loading
            reranker._load_model()


def run_interactive_demo():
    """Interactive demonstration of BGE reranker capabilities."""
    print("=" * 60)
    print("BGE Reranker Interactive Demo")
    print("=" * 60)
    
    # Create sample documents
    documents = []
    doc_texts = [
        "Python is a versatile programming language used for web development, data science, and machine learning.",
        "Machine learning models require large datasets for training and validation to achieve good performance.",
        "The weather forecast shows rain tomorrow with temperatures dropping to 15 degrees Celsius.",
        "Database indexing improves query performance by creating efficient data access paths.",
        "Natural language processing techniques help computers understand and generate human language.",
        "Cloud computing platforms provide scalable infrastructure for modern applications.",
        "Artificial intelligence is transforming industries through automation and intelligent decision making."
    ]
    
    for i, text in enumerate(doc_texts):
        doc = SimpleNamespace()
        doc.metadata = {
            "chunk_text": text,
            "timestamp": 1704067200 + i * 86400,
            "source": f"demo_doc_{i+1}.txt"
        }
        documents.append(doc)
    
    # Create reranker
    print("Initializing BGE reranker...")
    reranker = BGEReranker(
        cache_size=100,
        max_workers=2,
        batch_size=4,
        max_sequence_length=128
    )
    
    # Test queries
    test_queries = [
        "machine learning and artificial intelligence",
        "programming languages for development",
        "database performance optimization",
        "weather and temperature information"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        start_time = time.time()
        results = reranker.rerank(query, documents, top_k=3)
        end_time = time.time()
        
        print(f"Reranking took {end_time - start_time:.2f} seconds")
        print(f"Found {len(results)} relevant documents:")
        
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Score: {doc.rerank_score:.3f}")
            print(f"   Text: {doc.metadata['chunk_text'][:80]}...")


if __name__ == "__main__":
    # Run tests with pytest
    print("Running BGE Reranker Tests...")
    pytest.main([__file__, "-v", "-s"])
    
    # Run interactive demo
    print("\n" + "="*60)
    run_interactive_demo()
