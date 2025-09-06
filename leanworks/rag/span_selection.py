"""
Span Selection Module for RAG Pipeline

This module performs sentence-level selection from reranked documents using BM25 scoring.
It extracts the most relevant 3-5 sentences per document with ±1 neighbor context.
Supports both regular text and JSON/JSON Array formats with JSONPath flattening.
"""

import json
import logging
import re
import math
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Set up logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpanSelector:
    """
    Span selection module that selects the most relevant sentences from documents
    using BM25 scoring with context expansion.
    """
    
    def __init__(
        self,
        top_sentences_per_doc: int = 4,
        context_window: int = 1,
        min_sentence_length: int = 10,
        max_sentence_length: int = 500,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75
    ):
        """
        Initialize the SpanSelector.
        
        Args:
            top_sentences_per_doc: Number of top sentences to select per document (3-5)
            context_window: Number of neighbor sentences to include (±1)
            min_sentence_length: Minimum sentence length to consider
            max_sentence_length: Maximum sentence length to consider
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.top_sentences_per_doc = max(3, min(5, top_sentences_per_doc))
        self.context_window = context_window
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Initialize stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            logger.warning("English stopwords not available, using empty set")
            self.stopwords = set()
        
        logger.info(f"SpanSelector initialized with top_sentences_per_doc={self.top_sentences_per_doc}, "
                   f"context_window={self.context_window}")
    
    def select_spans(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Select relevant spans from a list of documents.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            
        Returns:
            List of documents with selected spans in metadata
        """
        if not documents:
            logger.warning("No documents provided for span selection")
            return []
        
        logger.info(f"Selecting spans from {len(documents)} documents for query: '{query}'")
        
        # Process query terms
        query_terms = self._preprocess_text(query)
        
        processed_docs = []
        for doc in documents:
            try:
                # Extract and process document content
                content = self._extract_document_content(doc)
                if not content or len(content.strip()) < self.min_sentence_length:
                    logger.warning(f"No meaningful content found in document {getattr(doc, 'id', 'unknown')}")
                    # Keep original document without processing
                    processed_docs.append(doc)
                    continue
                
                # Flatten JSON if needed
                flattened_content = self._flatten_json_content(content)
                
                # Split into sentences
                sentences = self._split_into_sentences(flattened_content)
                if not sentences:
                    logger.warning(f"No valid sentences found in document {getattr(doc, 'id', 'unknown')}")
                    continue
                
                # Select top sentences using BM25
                selected_indices = self._select_top_sentences_bm25(query_terms, sentences)
                
                # Expand with context
                expanded_indices = self._expand_with_context(selected_indices, len(sentences))
                
                # Extract selected spans
                selected_spans = [sentences[i] for i in sorted(expanded_indices)]
                
                # Update document metadata
                updated_doc = self._update_document_metadata(doc, selected_spans, expanded_indices, sentences)
                processed_docs.append(updated_doc)
                
                logger.debug(f"Selected {len(selected_spans)} spans from document {getattr(doc, 'id', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error processing document {getattr(doc, 'id', 'unknown')}: {str(e)}")
                # Keep original document in case of error
                processed_docs.append(doc)
        
        logger.info(f"Successfully processed {len(processed_docs)} documents with span selection")
        return processed_docs
    
    def _extract_document_content(self, doc: Any) -> str:
        """
        Extract content from document object.
        
        Args:
            doc: Document object with metadata
            
        Returns:
            Document content as string
        """
        # Try different content extraction methods
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            # Primary: chunk_text from metadata
            content = doc.metadata.get("chunk_text", "")
            if content:
                return content
            
            # Fallback: other text fields
            for field in ["text", "content", "body", "description"]:
                content = doc.metadata.get(field, "")
                if content:
                    return content
        
        # Try direct content access
        if hasattr(doc, 'page_content') and doc.page_content:
            return doc.page_content
        
        if hasattr(doc, 'content') and doc.content:
            return doc.content
        
        # Return empty string if no meaningful content found
        return ""
    
    def _flatten_json_content(self, content: str) -> str:
        """
        Flatten JSON or JSON Array content into JSONPath format for easier processing.
        
        Args:
            content: Raw content string
            
        Returns:
            Flattened content string
        """
        # Check if content looks like JSON
        content_stripped = content.strip()
        if not (content_stripped.startswith('{') or content_stripped.startswith('[')):
            return content
        
        try:
            # Try to parse as JSON
            data = json.loads(content_stripped)
            flattened_parts = []
            
            if isinstance(data, list):
                # Handle JSON Array
                for i, item in enumerate(data):
                    flattened_parts.extend(self._flatten_json_object(item, f"[{i}]"))
            elif isinstance(data, dict):
                # Handle JSON Object
                flattened_parts.extend(self._flatten_json_object(data, ""))
            else:
                # Simple value
                return str(data)
            
            # Join flattened parts with periods for sentence-like structure
            return ". ".join(flattened_parts) + "."
            
        except json.JSONDecodeError:
            # Not valid JSON, return as-is
            return content
    
    def _flatten_json_object(self, obj: Any, path_prefix: str = "") -> List[str]:
        """
        Recursively flatten a JSON object into JSONPath-like strings.
        
        Args:
            obj: JSON object to flatten
            path_prefix: Current JSONPath prefix
            
        Returns:
            List of flattened string representations
        """
        flattened = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path_prefix}.{key}" if path_prefix else key
                
                if isinstance(value, (dict, list)):
                    flattened.extend(self._flatten_json_object(value, current_path))
                else:
                    # Create a readable sentence from the key-value pair
                    flattened.append(f"{current_path} is {value}")
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path_prefix}[{i}]"
                
                if isinstance(item, (dict, list)):
                    flattened.extend(self._flatten_json_object(item, current_path))
                else:
                    flattened.append(f"{current_path} contains {item}")
        
        else:
            # Simple value
            if path_prefix:
                flattened.append(f"{path_prefix} is {obj}")
            else:
                flattened.append(str(obj))
        
        return flattened
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with filtering.
        
        Args:
            text: Input text
            
        Returns:
            List of filtered sentences
        """
        try:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}, using simple splitting")
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        # Filter sentences by length and content
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) >= self.min_sentence_length and 
                len(sentence) <= self.max_sentence_length and
                sentence):  # Not empty
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into terms for BM25 scoring.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed terms
        """
        try:
            # Tokenize and lowercase
            tokens = word_tokenize(text.lower())
        except Exception as e:
            logger.warning(f"NLTK word tokenization failed: {e}, using simple splitting")
            tokens = text.lower().split()
        
        # Filter out stopwords, punctuation, and short words
        terms = []
        for token in tokens:
            if (len(token) > 2 and 
                token.isalnum() and 
                token not in self.stopwords):
                terms.append(token)
        
        return terms
    
    def _select_top_sentences_bm25(self, query_terms: List[str], sentences: List[str]) -> List[int]:
        """
        Select top sentences using BM25 scoring.
        
        Args:
            query_terms: Preprocessed query terms
            sentences: List of sentences
            
        Returns:
            List of indices of top sentences
        """
        if not query_terms or not sentences:
            return []
        
        # Preprocess all sentences
        sentence_terms = []
        for sentence in sentences:
            terms = self._preprocess_text(sentence)
            sentence_terms.append(terms)
        
        # Calculate BM25 scores
        scores = self._calculate_bm25_scores(query_terms, sentence_terms)
        
        # Get top sentences
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N indices
        top_indices = [i for i, _ in scored_indices[:self.top_sentences_per_doc]]
        
        logger.debug(f"Selected {len(top_indices)} sentences with BM25 scores: "
                    f"{[scores[i] for i in top_indices]}")
        
        return top_indices
    
    def _calculate_bm25_scores(self, query_terms: List[str], sentence_terms: List[List[str]]) -> List[float]:
        """
        Calculate BM25 scores for sentences given query terms.
        
        Args:
            query_terms: Query terms
            sentence_terms: List of term lists for each sentence
            
        Returns:
            List of BM25 scores
        """
        if not sentence_terms:
            return []
        
        # Calculate document frequency for each term
        df = defaultdict(int)
        for terms in sentence_terms:
            unique_terms = set(terms)
            for term in unique_terms:
                df[term] += 1
        
        # Calculate average document length
        total_length = sum(len(terms) for terms in sentence_terms)
        avg_doc_length = total_length / len(sentence_terms) if sentence_terms else 0
        
        # Calculate BM25 score for each sentence
        scores = []
        N = len(sentence_terms)  # Total number of documents (sentences)
        
        for doc_terms in sentence_terms:
            score = 0.0
            doc_length = len(doc_terms)
            term_freq = Counter(doc_terms)
            
            for query_term in query_terms:
                if query_term in term_freq:
                    # Term frequency in document
                    tf = term_freq[query_term]
                    
                    # Document frequency
                    doc_freq = df[query_term]
                    
                    # IDF calculation - ensure positive IDF
                    idf = max(0.1, math.log((N - doc_freq + 0.5) / (doc_freq + 0.5)))
                    
                    # BM25 formula
                    numerator = tf * (self.bm25_k1 + 1)
                    denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (doc_length / avg_doc_length))
                    
                    score += idf * (numerator / denominator)
            
            scores.append(max(0.0, score))  # Ensure non-negative scores
        
        return scores
    
    def _expand_with_context(self, selected_indices: List[int], total_sentences: int) -> List[int]:
        """
        Expand selected sentence indices with context window.
        
        Args:
            selected_indices: Indices of selected sentences
            total_sentences: Total number of sentences
            
        Returns:
            Expanded list of indices including context
        """
        expanded_indices = set()
        
        for idx in selected_indices:
            # Add the selected sentence
            expanded_indices.add(idx)
            
            # Add context window
            for offset in range(-self.context_window, self.context_window + 1):
                context_idx = idx + offset
                if 0 <= context_idx < total_sentences:
                    expanded_indices.add(context_idx)
        
        return list(expanded_indices)
    
    def _update_document_metadata(
        self, 
        doc: Any, 
        selected_spans: List[str], 
        selected_indices: List[int],
        original_sentences: List[str]
    ) -> Any:
        """
        Update document metadata with selected spans.
        
        Args:
            doc: Original document
            selected_spans: Selected span texts
            selected_indices: Indices of selected spans
            original_sentences: All original sentences
            
        Returns:
            Updated document
        """
        # Create a copy of the document to avoid modifying the original
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            # Keep original text for reference BEFORE modifying chunk_text
            if "original_chunk_text" not in doc.metadata:
                doc.metadata["original_chunk_text"] = doc.metadata.get("chunk_text", "")
            
            # Update metadata with span information
            doc.metadata["selected_spans"] = selected_spans
            doc.metadata["selected_span_indices"] = selected_indices
            doc.metadata["total_sentences"] = len(original_sentences)
            doc.metadata["span_selection_applied"] = True
            
            # Replace chunk_text with selected spans joined
            doc.metadata["chunk_text"] = " ".join(selected_spans)
        
        return doc

    def get_selection_stats(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about span selection results.
        
        Args:
            documents: Processed documents
            
        Returns:
            Dictionary with selection statistics
        """
        stats = {
            "total_documents": len(documents),
            "documents_with_spans": 0,
            "total_selected_spans": 0,
            "avg_spans_per_doc": 0.0,
            "total_original_sentences": 0,
            "selection_ratio": 0.0
        }
        
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata.get("span_selection_applied"):
                stats["documents_with_spans"] += 1
                selected_spans = doc.metadata.get("selected_spans", [])
                stats["total_selected_spans"] += len(selected_spans)
                stats["total_original_sentences"] += doc.metadata.get("total_sentences", 0)
        
        if stats["documents_with_spans"] > 0:
            stats["avg_spans_per_doc"] = stats["total_selected_spans"] / stats["documents_with_spans"]
        
        if stats["total_original_sentences"] > 0:
            stats["selection_ratio"] = stats["total_selected_spans"] / stats["total_original_sentences"]
        
        return stats
