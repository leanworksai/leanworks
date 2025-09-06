"""
Test suite for SpanSelector functionality.
"""

import pytest
import json
from types import SimpleNamespace
from leanworks.rag.span_selection import SpanSelector


class TestSpanSelector:
    """Test suite for SpanSelector functionality."""
    
    @pytest.fixture
    def span_selector(self):
        """Create a SpanSelector instance for testing."""
        return SpanSelector(
            top_sentences_per_doc=4,
            context_window=1,
            min_sentence_length=10,
            max_sentence_length=500
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = []
        
        # Regular text document
        doc1 = SimpleNamespace()
        doc1.id = "doc1"
        doc1.metadata = {
            "chunk_text": "Python is a high-level programming language. It supports multiple programming paradigms including object-oriented programming. Machine learning algorithms can be implemented using various frameworks. These frameworks provide tools for building neural networks. The weather today is sunny with a temperature of 25 degrees.",
            "data_source": "programming_guide.txt",
            "timestamp": 1704067200
        }
        documents.append(doc1)
        
        # JSON document
        doc2 = SimpleNamespace()
        doc2.id = "doc2"
        doc2.metadata = {
            "chunk_text": '{"name": "John Doe", "age": 30, "skills": ["Python", "Machine Learning"], "projects": [{"title": "AI Assistant", "status": "completed"}, {"title": "Data Pipeline", "status": "in_progress"}]}',
            "data_source": "profile.json",
            "timestamp": 1704153600
        }
        documents.append(doc2)
        
        # JSON Array document
        doc3 = SimpleNamespace()
        doc3.id = "doc3"
        doc3.metadata = {
            "chunk_text": '[{"product": "laptop", "price": 1200}, {"product": "mouse", "price": 25}, {"product": "keyboard", "price": 80}]',
            "data_source": "products.json",
            "timestamp": 1704240000
        }
        documents.append(doc3)
        
        # Short document (should be kept as-is)
        doc4 = SimpleNamespace()
        doc4.id = "doc4"
        doc4.metadata = {
            "chunk_text": "This is a short document.",
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
    
    def test_initialization(self):
        """Test SpanSelector initialization."""
        selector = SpanSelector()
        assert selector.top_sentences_per_doc >= 3
        assert selector.top_sentences_per_doc <= 5
        assert selector.context_window == 1
        assert selector.min_sentence_length == 10
        assert selector.max_sentence_length == 500
        assert selector.bm25_k1 == 1.2
        assert selector.bm25_b == 0.75
    
    def test_initialization_with_custom_params(self):
        """Test SpanSelector initialization with custom parameters."""
        selector = SpanSelector(
            top_sentences_per_doc=3,
            context_window=2,
            min_sentence_length=5,
            max_sentence_length=1000,
            bm25_k1=1.5,
            bm25_b=0.8
        )
        assert selector.top_sentences_per_doc == 3
        assert selector.context_window == 2
        assert selector.min_sentence_length == 5
        assert selector.max_sentence_length == 1000
        assert selector.bm25_k1 == 1.5
        assert selector.bm25_b == 0.8
    
    def test_extract_document_content(self, span_selector, sample_documents):
        """Test document content extraction."""
        # Test normal document
        content = span_selector._extract_document_content(sample_documents[0])
        assert "Python is a high-level programming language" in content
        
        # Test JSON document
        content = span_selector._extract_document_content(sample_documents[1])
        assert '"name": "John Doe"' in content
    
    def test_flatten_json_content_object(self, span_selector):
        """Test JSON object flattening."""
        json_text = '{"name": "John", "age": 30, "address": {"city": "New York", "zip": "10001"}}'
        flattened = span_selector._flatten_json_content(json_text)
        
        assert "name is John" in flattened
        assert "age is 30" in flattened
        assert "address.city is New York" in flattened
        assert "address.zip is 10001" in flattened
    
    def test_flatten_json_content_array(self, span_selector):
        """Test JSON array flattening."""
        json_text = '[{"product": "laptop", "price": 1200}, {"product": "mouse", "price": 25}]'
        flattened = span_selector._flatten_json_content(json_text)
        
        assert "[0].product is laptop" in flattened
        assert "[0].price is 1200" in flattened
        assert "[1].product is mouse" in flattened
        assert "[1].price is 25" in flattened
    
    def test_flatten_json_content_non_json(self, span_selector):
        """Test flattening non-JSON content."""
        regular_text = "This is just regular text."
        flattened = span_selector._flatten_json_content(regular_text)
        assert flattened == regular_text
    
    def test_split_into_sentences(self, span_selector):
        """Test sentence splitting."""
        text = "This is the first sentence. This is the second sentence! This is the third sentence? This is a very short one."
        sentences = span_selector._split_into_sentences(text)
        
        assert len(sentences) >= 3  # At least 3 valid sentences
        assert "This is the first sentence" in sentences[0]
        assert "This is the second sentence" in sentences[1]
    
    def test_split_into_sentences_with_filtering(self, span_selector):
        """Test sentence splitting with length filtering."""
        text = "Short. This is a longer sentence that meets the minimum length requirement. X."
        sentences = span_selector._split_into_sentences(text)
        
        # Should filter out very short sentences
        assert len(sentences) == 1
        assert "This is a longer sentence" in sentences[0]
    
    def test_preprocess_text(self, span_selector):
        """Test text preprocessing for BM25."""
        text = "Python is a great programming language for machine learning!"
        terms = span_selector._preprocess_text(text)
        
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
    
    def test_calculate_bm25_scores(self, span_selector):
        """Test BM25 score calculation."""
        query_terms = ["python", "programming"]
        sentence_terms = [
            ["python", "programming", "language"],  # High relevance
            ["java", "programming", "language"],    # Medium relevance
            ["weather", "sunny", "temperature"]     # Low relevance
        ]
        
        scores = span_selector._calculate_bm25_scores(query_terms, sentence_terms)
        
        assert len(scores) == 3
        assert scores[0] > scores[1]  # Python sentence should score higher than Java
        assert scores[1] > scores[2]  # Java sentence should score higher than weather
        assert all(score >= 0 for score in scores)  # All scores should be non-negative
    
    def test_select_top_sentences_bm25(self, span_selector):
        """Test BM25-based sentence selection."""
        query_terms = ["python", "machine", "learning"]
        sentences = [
            "Python is great for machine learning applications.",
            "Java is also used in enterprise applications.",
            "Machine learning algorithms require careful tuning.",
            "The weather is nice today.",
            "Python frameworks like TensorFlow are popular for learning."
        ]
        
        selected_indices = span_selector._select_top_sentences_bm25(query_terms, sentences)
        
        assert len(selected_indices) <= span_selector.top_sentences_per_doc
        assert 0 in selected_indices  # First sentence should be selected (contains Python, machine, learning)
        assert 4 in selected_indices  # Last sentence should be selected (contains Python, learning)
    
    def test_expand_with_context(self, span_selector):
        """Test context expansion around selected sentences."""
        selected_indices = [2, 5]
        total_sentences = 8
        
        expanded_indices = span_selector._expand_with_context(selected_indices, total_sentences)
        
        # Should include original indices plus context
        assert 2 in expanded_indices
        assert 5 in expanded_indices
        
        # Should include context around index 2: [1, 2, 3]
        assert 1 in expanded_indices
        assert 3 in expanded_indices
        
        # Should include context around index 5: [4, 5, 6]
        assert 4 in expanded_indices
        assert 6 in expanded_indices
        
        # Should not include out-of-bounds indices
        assert -1 not in expanded_indices
        assert 8 not in expanded_indices
    
    def test_expand_with_context_boundary(self, span_selector):
        """Test context expansion at boundaries."""
        selected_indices = [0, 4]  # First and last sentences
        total_sentences = 5
        
        expanded_indices = span_selector._expand_with_context(selected_indices, total_sentences)
        
        # Should handle boundaries correctly
        assert 0 in expanded_indices
        assert 1 in expanded_indices  # Context for index 0
        assert 3 in expanded_indices  # Context for index 4
        assert 4 in expanded_indices
        
        # Should not include negative or out-of-bounds indices
        assert all(0 <= idx < total_sentences for idx in expanded_indices)
    
    def test_select_spans_regular_text(self, span_selector, sample_documents):
        """Test span selection on regular text documents."""
        query = "Python programming machine learning"
        docs = [sample_documents[0]]  # Regular text document
        
        result = span_selector.select_spans(query, docs)
        
        assert len(result) == 1
        doc = result[0]
        assert doc.metadata.get("span_selection_applied") is True
        assert "selected_spans" in doc.metadata
        assert "selected_span_indices" in doc.metadata
        assert "total_sentences" in doc.metadata
        assert len(doc.metadata["selected_spans"]) > 0
    
    def test_select_spans_json_document(self, span_selector, sample_documents):
        """Test span selection on JSON documents."""
        query = "John Doe Python skills"
        docs = [sample_documents[1]]  # JSON document
        
        result = span_selector.select_spans(query, docs)
        
        assert len(result) == 1
        doc = result[0]
        assert doc.metadata.get("span_selection_applied") is True
        selected_spans = doc.metadata.get("selected_spans", [])
        assert len(selected_spans) > 0
        
        # Should contain flattened JSON paths
        selected_text = " ".join(selected_spans)
        assert any("name is" in span or "skills" in span for span in selected_spans)
    
    def test_select_spans_json_array(self, span_selector, sample_documents):
        """Test span selection on JSON array documents."""
        query = "laptop price product"
        docs = [sample_documents[2]]  # JSON array document
        
        result = span_selector.select_spans(query, docs)
        
        assert len(result) == 1
        doc = result[0]
        assert doc.metadata.get("span_selection_applied") is True
        selected_spans = doc.metadata.get("selected_spans", [])
        assert len(selected_spans) > 0
        
        # Should contain flattened JSON array paths
        selected_text = " ".join(selected_spans)
        assert any("product is laptop" in span or "price is 1200" in span for span in selected_spans)
    
    def test_select_spans_empty_documents(self, span_selector, empty_documents):
        """Test span selection on empty documents."""
        query = "test query"
        
        result = span_selector.select_spans(query, empty_documents)
        
        # Should return documents but without span selection applied
        assert len(result) == len(empty_documents)
        for doc in result:
            assert not doc.metadata.get("span_selection_applied", False)
    
    def test_select_spans_no_documents(self, span_selector):
        """Test span selection with no documents."""
        query = "test query"
        
        result = span_selector.select_spans(query, [])
        
        assert result == []
    
    def test_update_document_metadata(self, span_selector):
        """Test document metadata updating."""
        doc = SimpleNamespace()
        doc.id = "test"
        doc.metadata = {"chunk_text": "Original text", "data_source": "test.txt"}
        
        selected_spans = ["Selected span 1", "Selected span 2"]
        selected_indices = [1, 3]
        original_sentences = ["Sent 0", "Sent 1", "Sent 2", "Sent 3", "Sent 4"]
        
        updated_doc = span_selector._update_document_metadata(
            doc, selected_spans, selected_indices, original_sentences
        )
        
        assert updated_doc.metadata["selected_spans"] == selected_spans
        assert updated_doc.metadata["selected_span_indices"] == selected_indices
        assert updated_doc.metadata["total_sentences"] == 5
        assert updated_doc.metadata["span_selection_applied"] is True
        assert updated_doc.metadata["chunk_text"] == "Selected span 1 Selected span 2"
        assert updated_doc.metadata["original_chunk_text"] == "Original text"
    
    def test_get_selection_stats(self, span_selector, sample_documents):
        """Test selection statistics calculation."""
        query = "Python programming"
        processed_docs = span_selector.select_spans(query, sample_documents)
        
        stats = span_selector.get_selection_stats(processed_docs)
        
        assert "total_documents" in stats
        assert "documents_with_spans" in stats
        assert "total_selected_spans" in stats
        assert "avg_spans_per_doc" in stats
        assert "total_original_sentences" in stats
        assert "selection_ratio" in stats
        
        assert stats["total_documents"] == len(sample_documents)
        assert stats["documents_with_spans"] >= 0
        assert stats["total_selected_spans"] >= 0
        assert stats["avg_spans_per_doc"] >= 0.0
        assert stats["selection_ratio"] >= 0.0
    
    def test_integration_with_different_queries(self, span_selector, sample_documents):
        """Test span selection with different query types."""
        queries = [
            "Python programming language",
            "machine learning algorithms",
            "John Doe profile information",
            "laptop price products",
            "weather temperature sunny"
        ]
        
        for query in queries:
            result = span_selector.select_spans(query, sample_documents)
            
            # Should always return same number of documents
            assert len(result) == len(sample_documents)
            
            # Should have some documents with spans selected (except for irrelevant queries)
            stats = span_selector.get_selection_stats(result)
            if any(term in query.lower() for term in ["python", "john", "laptop"]):
                assert stats["documents_with_spans"] > 0
    
    def test_error_handling(self, span_selector):
        """Test error handling in span selection."""
        # Test with malformed document
        bad_doc = SimpleNamespace()
        bad_doc.id = "bad"
        bad_doc.metadata = None  # This should cause an error
        
        query = "test query"
        result = span_selector.select_spans(query, [bad_doc])
        
        # Should return the original document even if processing fails
        assert len(result) == 1
        assert result[0] == bad_doc
    
    def test_sentence_length_filtering(self, span_selector):
        """Test that sentences are properly filtered by length."""
        # Create a document with sentences of various lengths
        doc = SimpleNamespace()
        doc.id = "length_test"
        doc.metadata = {
            "chunk_text": "Hi. This is a sentence that meets the minimum length requirement for processing. X. Another good sentence for testing the filtering mechanism. Very short. This is an extremely long sentence that goes on and on and on and contains many words and phrases and clauses that might exceed the maximum length limit for sentence processing in the span selector module and should therefore be filtered out during the preprocessing stage."
        }
        
        query = "sentence testing filtering"
        result = span_selector.select_spans(query, [doc])
        
        # Check that very short and very long sentences are filtered
        selected_spans = result[0].metadata.get("selected_spans", [])
        selected_text = " ".join(selected_spans)
        
        # Should not contain very short sentences
        assert "Hi" not in selected_text
        assert "X" not in selected_text
        
        # Should contain appropriately sized sentences
        assert any("minimum length requirement" in span for span in selected_spans)
