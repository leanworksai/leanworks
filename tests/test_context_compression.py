"""
Test suite for Context Compression module.

Tests the 2-pass context compression functionality including:
- Pass A: Lossless preserve & trim with deduplication
- Pass B: Lossy glue with question-aware synthesis
"""

import pytest
import unittest.mock as mock
from types import SimpleNamespace
from leanworks.rag.context_compression import ContextCompressor, CompressedSpan

class TestContextCompressor:
    """Test cases for the ContextCompressor class."""
    
    @pytest.fixture
    def compressor(self):
        """Create a ContextCompressor instance for testing."""
        return ContextCompressor(
            model_client=None,  # No model client for basic tests
            trim_window=5,
            similarity_threshold=0.9,
            min_span_length=10,
            max_span_length=100
        )
    
    @pytest.fixture
    def mock_model_client(self):
        """Create a mock model client for synthesis tests."""
        client = mock.Mock()
        response = mock.Mock()
        response.choices = [mock.Mock()]
        response.choices[0].message.content = "This is a synthesized summary."
        client.chat.completions.create.return_value = response
        return client
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with selected spans for testing."""
        docs = []
        
        # Document 1
        doc1 = SimpleNamespace()
        doc1.id = "doc1"
        doc1.metadata = {
            "selected_spans": [
                "The system can handle 600 req/min with proper configuration.",
                "Error codes include 404, 500, and timeout exceptions.",
                "Configuration requires setting timeout to 60 seconds."
            ],
            "selected_span_indices": [0, 1, 2],
            "data_source": "system_docs",
            "link": "https://docs.example.com/system"
        }
        docs.append(doc1)
        
        # Document 2 (similar content for deduplication testing)
        doc2 = SimpleNamespace()
        doc2.id = "doc2"
        doc2.metadata = {
            "selected_spans": [
                "The system handles up to 600 requests per minute when configured properly.",
                "Common error responses are 404 Not Found and 500 Internal Server Error.",
                "Set the timeout parameter to 60s for optimal performance."
            ],
            "selected_span_indices": [0, 1, 2],
            "data_source": "api_docs",
            "link": "https://api.example.com/docs"
        }
        docs.append(doc2)
        
        # Document 3 (different content)
        doc3 = SimpleNamespace()
        doc3.id = "doc3"
        doc3.metadata = {
            "selected_spans": [
                "Database connections are pooled with max_connections=100.",
                "Connection timeout is set to 30 seconds by default."
            ],
            "selected_span_indices": [0, 1],
            "data_source": "database_docs",
            "link": "https://db.example.com/docs"
        }
        docs.append(doc3)
        
        return docs
    
    def test_extract_spans_from_documents(self, compressor, sample_documents):
        """Test extraction of spans from documents."""
        spans = compressor._extract_spans_from_documents(sample_documents)
        
        # Should extract all spans from all documents
        assert len(spans) == 8  # 3 + 3 + 2 spans (one additional from fallback)
        
        # Check span properties
        assert all(isinstance(span, CompressedSpan) for span in spans)
        assert all(span.compression_type == "original" for span in spans)
        assert all(len(span.text) >= compressor.min_span_length for span in spans)
        
        # Check sources
        sources = {span.source for span in spans}
        assert sources == {"system_docs", "api_docs", "database_docs"}
    
    def test_filter_answer_bearing_spans(self, compressor):
        """Test filtering of answer-bearing spans."""
        query = "What is the system timeout configuration?"
        
        spans = [
            CompressedSpan("The system can handle 600 req/min.", "source1", "doc1", [0], "original"),
            CompressedSpan("Configuration requires setting timeout to 60 seconds.", "source1", "doc1", [1], "original"),
            CompressedSpan("This is unrelated content about colors.", "source1", "doc1", [2], "original"),
            CompressedSpan("Error code 404 means not found.", "source1", "doc1", [3], "original")
        ]
        
        filtered = compressor._filter_answer_bearing_spans(query, spans)
        
        # Should keep spans related to timeout and configuration
        assert len(filtered) >= 1
        timeout_span = next((s for s in filtered if "timeout" in s.text.lower()), None)
        assert timeout_span is not None
        assert timeout_span.compression_type == "preserved"
    
    def test_trim_spans_preserve_critical(self, compressor):
        """Test trimming spans while preserving critical content."""
        long_text = "This is a very long sentence that contains important information like 600 req/min and timeout of 60 seconds and error code 404 which should be preserved during trimming process."
        
        spans = [
            CompressedSpan(long_text, "source1", "doc1", [0], "original")
        ]
        
        trimmed = compressor._trim_spans_preserve_critical(spans)
        
        assert len(trimmed) == 1
        trimmed_span = trimmed[0]
        
        # Should preserve critical patterns
        assert "600 req/min" in trimmed_span.text or "60 seconds" in trimmed_span.text or "404" in trimmed_span.text
        assert trimmed_span.compression_type == "trimmed"
        assert len(trimmed_span.text) <= len(long_text)  # Should be shorter
    
    def test_canonicalize_spans(self, compressor):
        """Test canonicalization of spans."""
        spans = [
            CompressedSpan("Set timeout to 60 seconds for best performance.", "source1", "doc1", [0], "original"),
            CompressedSpan("Configure   timeout   to    60s   please.", "source1", "doc1", [1], "original"),
            CompressedSpan("Date: 2024-1-5 is important.", "source1", "doc1", [2], "original")
        ]
        
        canonicalized = compressor._canonicalize_spans(spans)
        
        # Should normalize units and whitespace
        for span in canonicalized:
            # No excessive whitespace
            assert "  " not in span.text
            # Should be trimmed
            assert span.text == span.text.strip()
        
        # Check specific normalizations
        timeout_spans = [s for s in canonicalized if "timeout" in s.text.lower()]
        assert len(timeout_spans) >= 1
        # Should normalize seconds to 's'
        normalized_span = next((s for s in timeout_spans if "60 s" in s.text), None)
        assert normalized_span is not None or any("60s" in s.text for s in timeout_spans)
    
    def test_deduplicate_spans(self, compressor):
        """Test deduplication of similar spans."""
        # Use a lower threshold for this test to ensure deduplication works
        compressor.similarity_threshold = 0.15  # Lower threshold to catch word-level similarity
        
        spans = [
            CompressedSpan("The system can handle 600 req/min.", "source1", "doc1", [0], "original"),
            CompressedSpan("System handles 600 requests per minute.", "source1", "doc1", [1], "original"),  # Similar
            CompressedSpan("Database timeout is 30 seconds.", "source1", "doc1", [2], "original"),  # Different
            CompressedSpan("The system handles 600 req/min properly.", "source1", "doc1", [3], "original")  # Similar to first
        ]
        
        deduplicated = compressor._deduplicate_spans(spans)
        
        # Should remove similar spans
        assert len(deduplicated) < len(spans)
        assert len(deduplicated) >= 2  # Should keep at least the different ones
        
        # Should keep the database timeout span (different content)
        database_span = next((s for s in deduplicated if "database" in s.text.lower()), None)
        assert database_span is not None
    
    def test_group_spans_by_source(self, compressor):
        """Test grouping spans by source."""
        spans = [
            CompressedSpan("Text from source B", "source_b", "doc2", [0], "original"),
            CompressedSpan("Text from source A", "source_a", "doc1", [0], "original"),
            CompressedSpan("More text from source B", "source_b", "doc2", [1], "original"),
            CompressedSpan("Another from source A", "source_a", "doc1", [1], "original")
        ]
        
        grouped = compressor._group_spans_by_source(spans)
        
        # Should maintain grouping by source
        assert len(grouped) == 4
        
        # Check that sources are grouped together
        source_positions = {}
        for i, span in enumerate(grouped):
            if span.source not in source_positions:
                source_positions[span.source] = []
            source_positions[span.source].append(i)
        
        # Each source's spans should be consecutive
        for source, positions in source_positions.items():
            assert positions == list(range(min(positions), max(positions) + 1))
    
    def test_pass_a_lossless_compression(self, compressor, sample_documents):
        """Test the complete Pass A lossless compression."""
        query = "What is the timeout configuration?"
        
        spans = compressor._extract_spans_from_documents(sample_documents)
        compressed_spans = compressor._pass_a_lossless_compression(query, spans)
        
        # Should have fewer or equal spans after compression
        assert len(compressed_spans) <= len(spans)
        
        # Should preserve important information
        timeout_spans = [s for s in compressed_spans if "timeout" in s.text.lower()]
        assert len(timeout_spans) >= 1
        
        # Should have proper compression types
        compression_types = {span.compression_type for span in compressed_spans}
        assert "preserved" in compression_types or "trimmed" in compression_types
    
    def test_pass_b_lossy_synthesis(self, compressor):
        """Test Pass B synthesis with mock Claude model client."""
        compressor.model_client = mock.Mock()
        response = mock.Mock()
        response.choices = [mock.Mock()]
        response.choices[0].message.content = "System supports 600 req/min with 60s timeout."
        compressor.model_client.chat.completions.create.return_value = response
        
        query = "What are the system limits?"
        spans = [
            CompressedSpan("System handles 600 req/min.", "system_docs", "doc1", [0], "preserved"),
            CompressedSpan("Timeout is 60 seconds.", "system_docs", "doc1", [1], "preserved")
        ]
        
        synthesized_spans = compressor._pass_b_lossy_synthesis(query, spans)
        
        # Should include synthesis + original spans
        assert len(synthesized_spans) >= len(spans)
        
        # Should have a synthesized span
        synthesis_spans = [s for s in synthesized_spans if s.compression_type == "synthesized"]
        assert len(synthesis_spans) >= 1
        
        # Synthesis should contain relevant information
        synthesis_span = synthesis_spans[0]
        assert "600" in synthesis_span.text or "timeout" in synthesis_span.text.lower()
    
    def test_compress_context_full_pipeline(self, compressor, sample_documents):
        """Test the complete compression pipeline."""
        query = "What is the system timeout and request limit?"
        
        compressed_spans, stats = compressor.compress_context(query, sample_documents, enable_pass_b=False)
        
        # Should return compressed spans and stats
        assert isinstance(compressed_spans, list)
        assert isinstance(stats, dict)
        
        # Stats should contain expected keys
        expected_keys = ["original_spans", "final_spans", "compression_ratio", "span_reduction"]
        for key in expected_keys:
            assert key in stats
        
        # Should have reduced number of spans or characters
        assert stats["compression_ratio"] <= 1.0
        assert stats["span_reduction"] >= 0.0
        
        # Should preserve important information
        all_text = " ".join(span.text for span in compressed_spans)
        assert "600" in all_text or "timeout" in all_text.lower()
    
    def test_format_compressed_context(self, compressor):
        """Test formatting of compressed context for LLM."""
        spans = [
            CompressedSpan("System info", "source_a", "doc1", [0], "preserved"),
            CompressedSpan("This is synthesis", "source_a", "doc1", [], "synthesized"),
            CompressedSpan("Database info", "source_b", "doc2", [0], "trimmed")
        ]
        
        formatted = compressor.format_compressed_context(spans)
        
        # Should include source headers
        assert "Source: source_a" in formatted
        assert "Source: source_b" in formatted
        
        # Should include compression indicators
        assert "[SYNTHESIS]" in formatted
        assert "[EXCERPT]" in formatted
        
        # Should include all span text
        for span in spans:
            assert span.text in formatted
    
    def test_empty_documents(self, compressor):
        """Test compression with empty document list."""
        compressed_spans, stats = compressor.compress_context("test query", [])
        
        assert compressed_spans == []
        assert stats["total_documents"] == 0
        assert stats["compressed_spans"] == 0
    
    def test_documents_without_spans(self, compressor):
        """Test compression with documents that have no selected spans."""
        doc = SimpleNamespace()
        doc.id = "doc1"
        doc.metadata = {"data_source": "test", "link": "http://test.com"}
        
        compressed_spans, stats = compressor.compress_context("test query", [doc])
        
        # Should handle gracefully
        assert isinstance(compressed_spans, list)
        assert isinstance(stats, dict)
    
    def test_preserve_patterns(self, compressor):
        """Test that critical patterns are preserved during compression."""
        # Test with spans containing various critical patterns
        spans = [
            CompressedSpan("API limit is 600 req/min for premium users.", "api", "doc1", [0], "original"),
            CompressedSpan("Error code HTTP_404 indicates resource not found.", "api", "doc1", [1], "original"),
            CompressedSpan("Version v2.1.3 includes new features.", "api", "doc1", [2], "original"),
            CompressedSpan("Timeout configured as 30000 ms maximum.", "api", "doc1", [3], "original"),
            CompressedSpan("Date 2024-12-01 marks the release.", "api", "doc1", [4], "original")
        ]
        
        # Apply Pass A compression
        query = "What are the API limits?"
        filtered = compressor._filter_answer_bearing_spans(query, spans)
        
        # Should preserve spans with critical patterns
        preserved_text = " ".join(span.text for span in filtered)
        
        # Check that critical patterns are preserved
        assert "600 req/min" in preserved_text
        assert "404" in preserved_text or "HTTP_404" in preserved_text
        assert "v2.1.3" in preserved_text
        assert "30000 ms" in preserved_text or "30000ms" in preserved_text
        assert "2024-12-01" in preserved_text

if __name__ == "__main__":
    pytest.main([__file__])
