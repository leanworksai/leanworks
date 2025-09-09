"""
Test suite for LLM-based Span Selection functionality.
"""

import pytest
import json
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from leanworks.rag.span_selection.span_selection_llm import LLMSpanSelector
from leanworks.rag.span_selection.span_selection_factory import SpanSelectionFactory


class TestLLMSpanSelector:
    """Test suite for LLMSpanSelector functionality."""
    
    @pytest.fixture
    def mock_llm_reranker(self):
        """Create a mock LLM reranker for testing."""
        mock_reranker = Mock()
        
        # Mock the _score_documents_async method to return scores directly
        async def mock_score_documents_async(query, documents):
            # Return decreasing scores for testing
            return [0.8 - (i * 0.1) for i in range(len(documents))]
        
        # Mock the rerank method to return scored documents (fallback)
        def mock_rerank(query, documents, top_k=None):
            # Create mock scored documents with rerank_score
            scored_docs = []
            for i, doc in enumerate(documents):
                # Create a copy of the document with rerank_score
                scored_doc = Mock()
                scored_doc.metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                scored_doc.rerank_score = 0.8 - (i * 0.1)  # Decreasing scores
                scored_docs.append(scored_doc)
            return scored_docs
        
        mock_reranker._score_documents_async = mock_score_documents_async
        mock_reranker.rerank.side_effect = mock_rerank
        return mock_reranker
    
    @pytest.fixture
    def llm_span_selector(self, mock_llm_reranker):
        """Create an LLMSpanSelector instance for testing."""
        return LLMSpanSelector(
            top_spans_per_doc=4,
            context_window=1,
            min_span_length=10,
            max_span_length=500,
            llm_reranker=mock_llm_reranker,
            use_sliding_windows=True,
            window_size=96,
            window_stride=48,
            max_span_candidates=60,
            max_final_spans=18,
            use_bm25_prefilter=True,
            bm25_k1=1.2,
            bm25_b=0.75
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = []
        
        # Regular text document
        doc1 = SimpleNamespace()
        doc1.id = "doc1"
        doc1.metadata = {
            "chunk_text": "Python is a high-level programming language. It supports multiple programming paradigms including object-oriented programming. Machine learning algorithms can be implemented using various frameworks. These frameworks provide tools for building neural networks. The weather today is sunny with a temperature of 25 degrees. Data science involves statistical analysis and machine learning techniques.",
            "data_source": "programming_guide.txt",
            "timestamp": 1704067200
        }
        documents.append(doc1)
        
        # JSON document
        doc2 = SimpleNamespace()
        doc2.id = "doc2"
        doc2.metadata = {
            "chunk_text": '{"name": "John Doe", "age": 30, "skills": ["Python", "Machine Learning", "Data Science"], "projects": [{"title": "AI Assistant", "status": "completed", "technologies": ["Python", "TensorFlow"]}, {"title": "Data Pipeline", "status": "in_progress", "technologies": ["Python", "Pandas"]}]}',
            "data_source": "profile.json",
            "timestamp": 1704153600
        }
        documents.append(doc2)
        
        # JSON Array document
        doc3 = SimpleNamespace()
        doc3.id = "doc3"
        doc3.metadata = {
            "chunk_text": '[{"product": "laptop", "price": 1200, "specs": {"cpu": "Intel i7", "ram": "16GB"}}, {"product": "mouse", "price": 25, "specs": {"type": "wireless"}}, {"product": "keyboard", "price": 80, "specs": {"type": "mechanical"}}]',
            "data_source": "products.json",
            "timestamp": 1704240000
        }
        documents.append(doc3)
        
        # Short document (should be kept as-is)
        doc4 = SimpleNamespace()
        doc4.id = "doc4"
        doc4.metadata = {
            "chunk_text": "This is a short document for testing.",
            "data_source": "short.txt",
            "timestamp": 1704326400
        }
        documents.append(doc4)
        
        return documents
    
    @pytest.fixture
    def empty_documents(self):
        """Create documents with empty or missing content."""
        documents = []
        
        # Document with empty chunk_text
        doc1 = SimpleNamespace()
        doc1.id = "empty1"
        doc1.metadata = {
            "chunk_text": "",
            "data_source": "empty.txt"
        }
        documents.append(doc1)
        
        # Document with missing chunk_text
        doc2 = SimpleNamespace()
        doc2.id = "empty2"
        doc2.metadata = {
            "data_source": "missing.txt"
        }
        documents.append(doc2)
        
        return documents

    def test_initialization(self, mock_llm_reranker):
        """Test LLMSpanSelector initialization."""
        selector = LLMSpanSelector(
            top_spans_per_doc=5,
            context_window=2,
            min_span_length=15,
            max_span_length=1000,
            llm_reranker=mock_llm_reranker,
            use_sliding_windows=False,
            window_size=128,
            window_stride=64,
            max_span_candidates=100,
            max_final_spans=30,
            use_bm25_prefilter=False,
            bm25_k1=1.5,
            bm25_b=0.8
        )
        
        assert selector.top_spans_per_doc == 5
        assert selector.context_window == 2
        assert selector.min_span_length == 15
        assert selector.max_span_length == 1000
        assert selector.llm_reranker == mock_llm_reranker
        assert selector.use_sliding_windows is False
        assert selector.window_size == 128
        assert selector.window_stride == 64
        assert selector.max_span_candidates == 100
        assert selector.max_final_spans == 30
        assert selector.use_bm25_prefilter is False
        assert selector.bm25_k1 == 1.5
        assert selector.bm25_b == 0.8

    def test_initialization_without_reranker(self):
        """Test LLMSpanSelector initialization without reranker (should use BM25 fallback)."""
        selector = LLMSpanSelector()
        assert selector.llm_reranker is None

    def test_extract_document_content(self, llm_span_selector, sample_documents):
        """Test document content extraction."""
        # Test normal document
        content = llm_span_selector._extract_document_content(sample_documents[0])
        assert "Python is a high-level programming language" in content
        
        # Test JSON document
        content = llm_span_selector._extract_document_content(sample_documents[1])
        assert '"name": "John Doe"' in content
        
        # Test document with page_content
        doc_with_page_content = SimpleNamespace()
        doc_with_page_content.page_content = "This is page content"
        content = llm_span_selector._extract_document_content(doc_with_page_content)
        assert content == "This is page content"
        
        # Test document with content attribute
        doc_with_content = SimpleNamespace()
        doc_with_content.content = "This is content"
        content = llm_span_selector._extract_document_content(doc_with_content)
        assert content == "This is content"

    def test_flatten_json_content_object(self, llm_span_selector):
        """Test JSON object flattening."""
        json_text = '{"name": "John", "age": 30, "address": {"city": "New York", "zip": "10001"}}'
        flattened = llm_span_selector._flatten_json_content(json_text)
        
        assert "name is John" in flattened
        assert "age is 30" in flattened
        assert "address.city is New York" in flattened
        assert "address.zip is 10001" in flattened

    def test_flatten_json_content_array(self, llm_span_selector):
        """Test JSON array flattening."""
        json_text = '[{"product": "laptop", "price": 1200}, {"product": "mouse", "price": 25}]'
        flattened = llm_span_selector._flatten_json_content(json_text)
        
        assert "[0].product is laptop" in flattened
        assert "[0].price is 1200" in flattened
        assert "[1].product is mouse" in flattened
        assert "[1].price is 25" in flattened

    def test_flatten_json_content_nested_structures(self, llm_span_selector):
        """Test flattening of nested JSON structures."""
        json_text = '{"user": {"profile": {"name": "Alice", "settings": {"theme": "dark"}}}, "items": [{"id": 1, "tags": ["urgent", "important"]}]}'
        flattened = llm_span_selector._flatten_json_content(json_text)
        
        assert "user.profile.name is Alice" in flattened
        assert "user.profile.settings.theme is dark" in flattened
        assert "items[0].id is 1" in flattened
        assert "items[0].tags[0] contains urgent" in flattened
        assert "items[0].tags[1] contains important" in flattened

    def test_flatten_json_content_non_json(self, llm_span_selector):
        """Test flattening non-JSON content."""
        regular_text = "This is just regular text."
        flattened = llm_span_selector._flatten_json_content(regular_text)
        assert flattened == regular_text

    def test_generate_sliding_window_candidates(self, llm_span_selector):
        """Test sliding window span generation."""
        # Create a much longer text to ensure multiple sliding windows
        text = "This is a very long text that should definitely be split into multiple sliding windows for testing purposes. " * 10
        
        spans = llm_span_selector._generate_sliding_window_candidates(text)
        
        # Should generate at least one span
        assert len(spans) >= 1
        assert all(len(span) >= llm_span_selector.min_span_length for span in spans)
        assert all(len(span) <= llm_span_selector.max_span_length for span in spans)
        
        # Check that spans overlap (sliding window property) if we have multiple spans
        if len(spans) > 1:
            # For sliding windows, consecutive spans should have some overlap
            # We'll check that the end of one span appears in the next span
            has_overlap = False
            for i in range(len(spans)-1):
                # Check if there's any common substring between consecutive spans
                span1_words = set(spans[i].split())
                span2_words = set(spans[i+1].split())
                if len(span1_words.intersection(span2_words)) > 5:  # At least 5 common words
                    has_overlap = True
                    break
            assert has_overlap, "Sliding windows should have overlapping content"

    def test_generate_sentence_candidates(self, llm_span_selector):
        """Test sentence-based span generation."""
        text = "This is the first sentence. This is the second sentence! This is the third sentence? This is a very short one. This is another longer sentence that meets the minimum length requirement."
        
        spans = llm_span_selector._generate_sentence_candidates(text)
        
        assert len(spans) >= 3  # Should have at least 3 valid sentences
        assert all(len(span) >= llm_span_selector.min_span_length for span in spans)
        assert all(len(span) <= llm_span_selector.max_span_length for span in spans)

    def test_preprocess_text(self, llm_span_selector):
        """Test text preprocessing for BM25."""
        text = "Python is a great programming language for machine learning!"
        terms = llm_span_selector._preprocess_text(text)
        
        # Should remove stopwords and short words
        assert "python" in terms
        assert "great" in terms
        assert "programming" in terms
        assert "language" in terms
        assert "machine" in terms
        assert "learning" in terms
        
        # Should not contain stopwords or punctuation
        assert "is" not in terms
        assert "a" not in terms
        assert "!" not in terms

    def test_calculate_bm25_scores(self, llm_span_selector):
        """Test BM25 score calculation."""
        query_terms = ["python", "programming"]
        sentence_terms = [
            ["python", "programming", "language"],  # High relevance
            ["java", "programming", "language"],    # Medium relevance
            ["weather", "sunny", "temperature"]     # Low relevance
        ]
        
        scores = llm_span_selector._calculate_bm25_scores(query_terms, sentence_terms)
        
        assert len(scores) == 3
        assert scores[0] > scores[1]  # Python sentence should score higher than Java
        assert scores[1] > scores[2]  # Java sentence should score higher than weather
        assert all(score >= 0 for score in scores)  # All scores should be non-negative

    def test_bm25_prefilter(self, llm_span_selector):
        """Test BM25 pre-filtering functionality."""
        query_terms = ["python", "machine", "learning"]
        span_candidates = [
            "Python is great for machine learning applications.",
            "Java is also used in enterprise applications.",
            "Machine learning algorithms require careful tuning.",
            "The weather is nice today.",
            "Python frameworks like TensorFlow are popular for learning.",
            "Data science involves statistical analysis and machine learning."
        ]
        doc_span_mapping = {i: (Mock(), i) for i in range(len(span_candidates))}
        
        filtered_candidates, filtered_mapping = llm_span_selector._bm25_prefilter(
            query_terms, span_candidates, doc_span_mapping
        )
        
        assert len(filtered_candidates) <= len(span_candidates)
        assert len(filtered_mapping) == len(filtered_candidates)
        
        # Should prioritize spans with query terms
        filtered_text = " ".join(filtered_candidates)
        assert "python" in filtered_text.lower() or "machine" in filtered_text.lower()

    def test_score_spans_with_llm(self, llm_span_selector):
        """Test LLM span scoring."""
        query = "Python programming language"
        span_candidates = [
            "Python is a programming language.",
            "Java is also a programming language.",
            "The weather is nice today."
        ]
        
        scores = llm_span_selector._score_spans_with_llm(query, span_candidates)
        
        assert len(scores) == len(span_candidates)
        assert all(0.0 <= score <= 1.0 for score in scores)
        
        # Verify that the LLM reranker was called
        # Note: We can't easily test async method calls in sync context, so we just verify scores were returned
        assert len(scores) == len(span_candidates)

    def test_score_spans_with_llm_error_handling(self, llm_span_selector):
        """Test LLM span scoring error handling."""
        # Mock the async method to raise an exception
        async def mock_error_async(query, documents):
            raise Exception("LLM API Error")
        
        llm_span_selector.llm_reranker._score_documents_async = mock_error_async
        
        query = "test query"
        span_candidates = ["span1", "span2"]
        
        scores = llm_span_selector._score_spans_with_llm(query, span_candidates)
        
        # Should return default scores as fallback
        assert scores == [0.5, 0.5]

    def test_select_top_spans_by_document(self, llm_span_selector):
        """Test top span selection by document."""
        span_scores = [0.9, 0.7, 0.8, 0.6, 0.5]
        span_candidates = [
            "Python is great for machine learning.",
            "Java is used in enterprise applications.",
            "Machine learning algorithms are powerful.",
            "The weather is nice today.",
            "Data science involves statistics."
        ]
        doc_span_mapping = {
            0: (Mock(id="doc1"), 0),
            1: (Mock(id="doc1"), 1),
            2: (Mock(id="doc2"), 0),
            3: (Mock(id="doc2"), 1),
            4: (Mock(id="doc2"), 2)
        }
        
        selected_spans_by_doc = llm_span_selector._select_top_spans_by_document(
            span_scores, span_candidates, doc_span_mapping
        )
        
        assert len(selected_spans_by_doc) > 0
        
        # Check that spans are grouped by document
        for doc_id, spans_with_scores in selected_spans_by_doc.items():
            assert len(spans_with_scores) > 0
            for span, score, original_idx in spans_with_scores:
                assert isinstance(span, str)
                assert isinstance(score, float)
                assert isinstance(original_idx, int)

    def test_update_documents_with_spans(self, llm_span_selector):
        """Test document updating with selected spans."""
        documents = [Mock(id="doc1", metadata={"chunk_text": "original text"})]
        selected_spans_by_doc = {
            "doc1": [
                ("Python is great for machine learning.", 0.9, 0),
                ("Machine learning algorithms are powerful.", 0.8, 2)
            ]
        }
        
        processed_docs = llm_span_selector._update_documents_with_spans(
            documents, selected_spans_by_doc
        )
        
        assert len(processed_docs) == 1
        doc = processed_docs[0]
        assert doc.metadata["span_selection_applied"] is True
        assert doc.metadata["span_selection_method"] == "llm"
        assert "selected_spans" in doc.metadata
        assert "span_scores" in doc.metadata
        assert "selected_span_indices" in doc.metadata
        assert len(doc.metadata["selected_spans"]) == 2

    def test_add_context_to_span(self, llm_span_selector):
        """Test adding context to ultra-short spans."""
        # Create a mock document with content
        doc = Mock()
        doc.metadata = {"chunk_text": "This is a longer document with more context around the short span."}
        
        short_span = "short span"
        expanded_span = llm_span_selector._add_context_to_span(short_span, doc)
        
        # Should add context if span is very short
        if len(short_span) < 50:
            assert len(expanded_span) > len(short_span)
            assert short_span in expanded_span

    def test_select_spans_with_llm(self, llm_span_selector, sample_documents):
        """Test full LLM span selection process."""
        query = "Python programming machine learning"
        
        result = llm_span_selector.select_spans(query, sample_documents)
        
        assert len(result) == len(sample_documents)
        
        # Check that some documents have spans selected
        documents_with_spans = [doc for doc in result if doc.metadata.get("span_selection_applied")]
        assert len(documents_with_spans) > 0
        
        # Verify LLM reranker was used (scores were generated)
        # Note: We can't easily test async method calls in sync context, so we verify the results
        assert any(doc.metadata.get("span_selection_applied") for doc in result)

    def test_select_spans_without_llm_reranker(self, sample_documents):
        """Test span selection without LLM reranker (BM25 fallback)."""
        selector = LLMSpanSelector()  # No reranker provided
        
        query = "Python programming"
        result = selector.select_spans(query, sample_documents)
        
        assert len(result) == len(sample_documents)
        
        # Should use BM25 fallback
        documents_with_spans = [doc for doc in result if doc.metadata.get("span_selection_applied")]
        assert len(documents_with_spans) > 0
        
        # Check that BM25 method was used
        for doc in documents_with_spans:
            assert doc.metadata.get("span_selection_method") == "bm25"

    def test_select_spans_empty_documents(self, llm_span_selector, empty_documents):
        """Test span selection on empty documents."""
        query = "test query"
        
        result = llm_span_selector.select_spans(query, empty_documents)
        
        # Should return documents but without span selection applied
        assert len(result) == len(empty_documents)
        for doc in result:
            assert not doc.metadata.get("span_selection_applied", False)

    def test_select_spans_no_documents(self, llm_span_selector):
        """Test span selection with no documents."""
        query = "test query"
        
        result = llm_span_selector.select_spans(query, [])
        
        assert result == []

    def test_select_spans_llm_error_fallback(self, sample_documents):
        """Test that LLM errors fall back to BM25."""
        # Create selector with a reranker that will fail
        mock_reranker = Mock()
        mock_reranker.rerank.side_effect = Exception("LLM API Error")
        
        selector = LLMSpanSelector(llm_reranker=mock_reranker)
        
        query = "Python programming"
        result = selector.select_spans(query, sample_documents)
        
        assert len(result) == len(sample_documents)
        
        # Should fall back to BM25
        documents_with_spans = [doc for doc in result if doc.metadata.get("span_selection_applied")]
        assert len(documents_with_spans) > 0

    def test_sliding_windows_vs_sentences(self, mock_llm_reranker, sample_documents):
        """Test difference between sliding windows and sentence-based approaches."""
        # Test with sliding windows
        selector_windows = LLMSpanSelector(
            llm_reranker=mock_llm_reranker,
            use_sliding_windows=True
        )
        
        # Test with sentences
        selector_sentences = LLMSpanSelector(
            llm_reranker=mock_llm_reranker,
            use_sliding_windows=False
        )
        
        query = "Python programming"
        
        result_windows = selector_windows.select_spans(query, sample_documents[:1])
        result_sentences = selector_sentences.select_spans(query, sample_documents[:1])
        
        # Both should work
        assert len(result_windows) == 1
        assert len(result_sentences) == 1
        
        # Results might be different due to different span generation methods
        # This test mainly ensures both methods work without errors

    def test_max_span_candidates_capping(self, mock_llm_reranker):
        """Test that span candidates are properly capped."""
        # Create a long document that would generate many candidates
        long_text = ". ".join([f"This is sentence number {i} about Python programming and machine learning." for i in range(100)])
        
        doc = SimpleNamespace()
        doc.id = "long_doc"
        doc.metadata = {"chunk_text": long_text}
        
        selector = LLMSpanSelector(
            llm_reranker=mock_llm_reranker,
            max_span_candidates=20,  # Low limit for testing
            use_sliding_windows=True
        )
        
        query = "Python programming"
        result = selector.select_spans(query, [doc])
        
        # Should still work despite many candidates
        assert len(result) == 1
        assert result[0].metadata.get("span_selection_applied") is True

    def test_max_final_spans_limiting(self, mock_llm_reranker, sample_documents):
        """Test that final spans are properly limited."""
        selector = LLMSpanSelector(
            llm_reranker=mock_llm_reranker,
            max_final_spans=5,  # Low limit for testing
            top_spans_per_doc=10  # High per-doc limit
        )
        
        query = "Python programming machine learning"
        result = selector.select_spans(query, sample_documents)
        
        # Count total selected spans across all documents
        total_spans = sum(len(doc.metadata.get("selected_spans", [])) for doc in result)
        assert total_spans <= 5  # Should respect max_final_spans limit


class TestSpanSelectionFactory:
    """Test suite for SpanSelectionFactory functionality."""
    
    @pytest.fixture
    def mock_llm_reranker(self):
        """Create a mock LLM reranker for testing."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = []
        return mock_reranker
    
    @pytest.fixture
    def mock_bge_reranker(self):
        """Create a mock BGE reranker for testing."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = []
        return mock_reranker

    def test_create_llm_span_selector(self, mock_llm_reranker):
        """Test creating LLM span selector via factory."""
        selector = SpanSelectionFactory.create_span_selector(
            span_selection_type="llm",
            reranker=mock_llm_reranker,
            top_spans_per_doc=5,
            context_window=2,
            min_span_length=15,
            max_span_length=1000
        )
        
        assert isinstance(selector, LLMSpanSelector)
        assert selector.llm_reranker == mock_llm_reranker
        assert selector.top_spans_per_doc == 5
        assert selector.context_window == 2
        assert selector.min_span_length == 15
        assert selector.max_span_length == 1000

    def test_create_llm_span_selector_without_reranker(self):
        """Test creating LLM span selector without reranker should raise error."""
        with pytest.raises(ValueError, match="reranker is required for LLM-based span selector"):
            SpanSelectionFactory.create_span_selector(
                span_selection_type="llm",
                reranker=None
            )

    def test_create_span_selector_invalid_type(self):
        """Test creating span selector with invalid type."""
        with pytest.raises(ValueError, match="Unsupported span selection type"):
            SpanSelectionFactory.create_span_selector(
                span_selection_type="invalid_type",
                reranker=Mock()
            )

    def test_get_available_types(self):
        """Test getting available span selection types."""
        types = SpanSelectionFactory.get_available_types()
        assert "llm" in types
        assert "bge" in types
        assert len(types) == 2


class TestSpanSelectorIntegration:
    """Integration tests for the unified SpanSelector class."""
    
    @pytest.fixture
    def mock_llm_reranker(self):
        """Create a mock LLM reranker for testing."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = []
        return mock_reranker
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = []
        
        doc1 = SimpleNamespace()
        doc1.id = "doc1"
        doc1.metadata = {
            "chunk_text": "Python is a high-level programming language. It supports multiple programming paradigms including object-oriented programming. Machine learning algorithms can be implemented using various frameworks.",
            "data_source": "programming_guide.txt"
        }
        documents.append(doc1)
        
        return documents

    def test_span_selector_with_llm_type(self, mock_llm_reranker, sample_documents):
        """Test unified SpanSelector with LLM type."""
        from leanworks.rag.span_selection.span_selection_factory import SpanSelector
        
        selector = SpanSelector(
            span_selection_type="llm",
            reranker=mock_llm_reranker,
            top_spans_per_doc=3
        )
        
        query = "Python programming"
        result = selector.select_spans(query, sample_documents)
        
        assert len(result) == len(sample_documents)
        assert hasattr(selector.span_selector, 'llm_reranker')

    def test_span_selector_get_selection_stats(self, mock_llm_reranker, sample_documents):
        """Test getting selection statistics."""
        from leanworks.rag.span_selection.span_selection_factory import SpanSelector
        
        selector = SpanSelector(
            span_selection_type="llm",
            reranker=mock_llm_reranker
        )
        
        query = "Python programming"
        processed_docs = selector.select_spans(query, sample_documents)
        stats = selector.get_selection_stats(processed_docs)
        
        assert "total_documents" in stats
        assert "documents_with_spans" in stats
        assert "total_selected_spans" in stats
        assert "avg_spans_per_doc" in stats
        assert "selection_method" in stats
        assert stats["total_documents"] == len(sample_documents)


if __name__ == "__main__":
    pytest.main([__file__])
