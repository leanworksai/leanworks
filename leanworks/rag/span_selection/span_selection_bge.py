"""
BGE-based Span Selection Module for RAG Pipeline

This module performs span-level selection from reranked documents using BGE semantic scoring.
It extracts the most relevant spans using sliding windows or sentences with BGE reranking.
Supports both regular text and JSON/JSON Array formats with JSONPath flattening.
Uses efficient BGE scoring with capping to maintain performance.
"""

import json
import logging
import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from .span_selection_base import BaseSpanSelector

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

class BGESpanSelector(BaseSpanSelector):
    """
    BGE-based span selection module that selects the most relevant spans from documents
    using BGE semantic scoring with sliding windows or sentence-based candidates.
    """
    
    def __init__(
        self,
        top_spans_per_doc: int = 6,
        context_window: int = 1,
        min_span_length: int = 10,
        max_span_length: int = 500,
        bge_reranker: Optional[Any] = None,
        use_sliding_windows: bool = True,
        window_size: int = 96,
        window_stride: int = 48,
        max_span_candidates: int = 60,
        max_final_spans: int = 18,
        use_bm25_prefilter: bool = True,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75
    ):
        """
        Initialize the BGESpanSelector.
        
        Args:
            top_spans_per_doc: Number of top spans to select per document (default 6)
            context_window: Number of neighbor sentences to include (±1)
            min_span_length: Minimum span length to consider
            max_span_length: Maximum span length to consider
            bge_reranker: BGE reranker instance for semantic scoring
            use_sliding_windows: Whether to use sliding windows vs sentences
            window_size: Size of sliding windows in tokens (default 96)
            window_stride: Stride of sliding windows in tokens (default 48)
            max_span_candidates: Maximum span candidates to score with BGE (default 60)
            max_final_spans: Maximum final spans to return (default 18)
            use_bm25_prefilter: Whether to use BM25 pre-filtering before BGE
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        super().__init__()
        self.top_spans_per_doc = max(3, min(10, top_spans_per_doc))
        self.context_window = context_window
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.bge_reranker = bge_reranker
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_span_candidates = max_span_candidates
        self.max_final_spans = max_final_spans
        self.use_bm25_prefilter = use_bm25_prefilter
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Initialize stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            logger.warning("English stopwords not available, using empty set")
            self.stopwords = set()
        
        logger.info(f"BGESpanSelector initialized with top_spans_per_doc={self.top_spans_per_doc}, "
                   f"context_window={self.context_window}, use_sliding_windows={self.use_sliding_windows}, "
                   f"window_size={self.window_size}, max_span_candidates={self.max_span_candidates}, "
                   f"bge_reranker={'available' if self.bge_reranker else 'none'}")
    
    def select_spans(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Select relevant spans from a list of documents using BGE semantic scoring.
        
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
        
        # If no BGE reranker available, fall back to BM25-only approach
        if not self.bge_reranker:
            logger.warning("No BGE reranker available, falling back to BM25-only span selection")
            return self._select_spans_bm25_fallback(query, documents)
        
        try:
            # Step 1: Generate span candidates from all documents
            all_span_candidates = []
            doc_span_mapping = {}  # Map span index to (doc, original_index)
            
            for doc_idx, doc in enumerate(documents):
                try:
                    # Extract and process document content
                    content = self._extract_document_content(doc)
                    if not content or len(content.strip()) < self.min_span_length:
                        logger.warning(f"No meaningful content found in document {getattr(doc, 'id', 'unknown')}")
                        continue
                    
                    # Flatten JSON if needed
                    flattened_content = self._flatten_json_content(content)
                    
                    # Generate span candidates
                    if self.use_sliding_windows:
                        spans = self._generate_sliding_window_candidates(flattened_content)
                    else:
                        spans = self._generate_sentence_candidates(flattened_content)
                    
                    if not spans:
                        logger.warning(f"No valid spans found in document {getattr(doc, 'id', 'unknown')}")
                        continue
                    
                    # Store span candidates with document mapping
                    start_idx = len(all_span_candidates)
                    all_span_candidates.extend(spans)
                    for span_idx in range(start_idx, len(all_span_candidates)):
                        doc_span_mapping[span_idx] = (doc, span_idx - start_idx)
                    
                    logger.debug(f"Generated {len(spans)} span candidates from document {getattr(doc, 'id', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error processing document {getattr(doc, 'id', 'unknown')}: {str(e)}")
                    continue
            
            if not all_span_candidates:
                logger.warning("No span candidates generated from any documents")
                return documents
            
            logger.info(f"Generated {len(all_span_candidates)} total span candidates")
            
            # Step 2: Optional BM25 pre-filtering to reduce candidates
            if self.use_bm25_prefilter and len(all_span_candidates) > self.max_span_candidates:
                query_terms = self._preprocess_text(query)
                filtered_candidates, filtered_mapping = self._bm25_prefilter(
                    query_terms, all_span_candidates, doc_span_mapping
                )
                all_span_candidates = filtered_candidates
                doc_span_mapping = filtered_mapping
                logger.info(f"BM25 pre-filtering reduced candidates to {len(all_span_candidates)}")
            
            # Step 3: Cap candidates to max_span_candidates
            if len(all_span_candidates) > self.max_span_candidates:
                all_span_candidates = all_span_candidates[:self.max_span_candidates]
                # Update mapping to only include selected candidates
                doc_span_mapping = {i: doc_span_mapping[i] for i in range(len(all_span_candidates))}
                logger.info(f"Capped span candidates to {len(all_span_candidates)}")
            
            # Step 4: Score spans with BGE
            span_scores = self._score_spans_with_bge(query, all_span_candidates)
            
            # Step 5: Select top spans and group by document
            selected_spans_by_doc = self._select_top_spans_by_document(
                span_scores, all_span_candidates, doc_span_mapping
            )
            
            # Step 6: Update documents with selected spans
            processed_docs = self._update_documents_with_spans(documents, selected_spans_by_doc)
            
            logger.info(f"Successfully processed {len(processed_docs)} documents with BGE span selection")
            return processed_docs
            
        except Exception as e:
            logger.error(f"BGE span selection failed: {str(e)}, falling back to BM25")
            return self._select_spans_bm25_fallback(query, documents)
    
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
            if (len(sentence) >= self.min_span_length and 
                len(sentence) <= self.max_span_length and
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
    
    def _generate_sliding_window_candidates(self, text: str) -> List[str]:
        """
        Generate sliding window span candidates from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sliding window spans
        """
        if not hasattr(self.bge_reranker, '_tokenizer') or not self.bge_reranker._tokenizer:
            # Fallback to character-based sliding windows if no tokenizer available
            return self._generate_character_sliding_windows(text)
        
        try:
            # Use BGE tokenizer for accurate token-based windows
            tokenizer = self.bge_reranker._tokenizer
            
            # Tokenize the text
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            
            if len(tokens) <= self.window_size:
                # Text is shorter than window size, return as single span
                return [text] if len(text.strip()) >= self.min_span_length else []
            
            # Generate sliding windows
            spans = []
            for i in range(0, max(1, len(tokens) - self.window_size + 1), self.window_stride):
                window_tokens = tokens[i:i + self.window_size]
                window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
                
                # Filter by length
                if (len(window_text.strip()) >= self.min_span_length and 
                    len(window_text.strip()) <= self.max_span_length):
                    spans.append(window_text.strip())
            
            return spans
            
        except Exception as e:
            logger.warning(f"Token-based sliding windows failed: {e}, using character-based fallback")
            return self._generate_character_sliding_windows(text)
    
    def _generate_character_sliding_windows(self, text: str) -> List[str]:
        """
        Generate character-based sliding window spans as fallback.
        
        Args:
            text: Input text
            
        Returns:
            List of sliding window spans
        """
        # Rough estimation: ~4 characters per token
        char_window_size = self.window_size * 4
        char_stride = self.window_stride * 4
        
        if len(text) <= char_window_size:
            return [text] if len(text.strip()) >= self.min_span_length else []
        
        spans = []
        for i in range(0, max(1, len(text) - char_window_size + 1), char_stride):
            window_text = text[i:i + char_window_size].strip()
            
            if (len(window_text) >= self.min_span_length and 
                len(window_text) <= self.max_span_length):
                spans.append(window_text)
        
        return spans
    
    def _generate_sentence_candidates(self, text: str) -> List[str]:
        """
        Generate sentence-based span candidates from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence spans
        """
        sentences = self._split_into_sentences(text)
        
        # Filter sentences by length
        filtered_sentences = []
        for sentence in sentences:
            if (len(sentence) >= self.min_span_length and 
                len(sentence) <= self.max_span_length):
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _bm25_prefilter(
        self, 
        query_terms: List[str], 
        span_candidates: List[str], 
        doc_span_mapping: Dict[int, Tuple[Any, int]]
    ) -> Tuple[List[str], Dict[int, Tuple[Any, int]]]:
        """
        Pre-filter span candidates using BM25 to reduce the number before BGE scoring.
        
        Args:
            query_terms: Preprocessed query terms
            span_candidates: List of span candidates
            doc_span_mapping: Mapping from span index to (doc, original_index)
            
        Returns:
            Tuple of (filtered_candidates, filtered_mapping)
        """
        if not query_terms or not span_candidates:
            return span_candidates, doc_span_mapping
        
        # Preprocess span candidates
        span_terms = []
        for span in span_candidates:
            terms = self._preprocess_text(span)
            span_terms.append(terms)
        
        # Calculate BM25 scores
        bm25_scores = self._calculate_bm25_scores(query_terms, span_terms)
        
        # Group spans by document ID and select top spans per document
        doc_spans = defaultdict(list)
        for i, (span, score) in enumerate(zip(span_candidates, bm25_scores)):
            doc, _ = doc_span_mapping[i]
            doc_id = getattr(doc, 'id', id(doc))  # Use document ID as key
            doc_spans[doc_id].append((i, span, score, doc))
        
        # Select top spans per document
        filtered_candidates = []
        filtered_mapping = {}
        new_idx = 0
        
        for doc_id, spans_with_scores in doc_spans.items():
            # Sort by BM25 score and take top spans per document
            spans_with_scores.sort(key=lambda x: x[2], reverse=True)
            top_spans = spans_with_scores[:self.top_spans_per_doc]
            
            for original_idx, span, score, doc in top_spans:
                filtered_candidates.append(span)
                filtered_mapping[new_idx] = (doc, original_idx)
                new_idx += 1
        
        logger.debug(f"BM25 pre-filtering: {len(span_candidates)} -> {len(filtered_candidates)} candidates")
        return filtered_candidates, filtered_mapping
    
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
    
    def _score_spans_with_bge(self, query: str, span_candidates: List[str]) -> List[float]:
        """
        Score span candidates using BGE reranker.
        
        Args:
            query: Query string
            span_candidates: List of span candidates
            
        Returns:
            List of BGE scores
        """
        if not span_candidates:
            return []
        
        try:
            # Create query-span pairs for BGE
            pairs = [(query, span) for span in span_candidates]
            
            # Use BGE reranker to score pairs
            # Configure for span scoring (shorter max_length, appropriate batch size)
            scores = []
            
            # Process in batches for efficiency
            batch_size = min(32, len(pairs))  # Smaller batch size for spans
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Extract just the spans for scoring
                batch_spans = [span for _, span in batch_pairs]
                
                # Use BGE reranker's internal scoring method
                batch_scores = self.bge_reranker._compute_similarity_scores_optimized(query, batch_spans)
                scores.extend(batch_scores)
            
            logger.debug(f"BGE scored {len(span_candidates)} span candidates")
            return scores
            
        except Exception as e:
            logger.error(f"BGE span scoring failed: {str(e)}")
            # Return zero scores as fallback
            return [0.0] * len(span_candidates)
    
    def _select_top_spans_by_document(
        self, 
        span_scores: List[float], 
        span_candidates: List[str], 
        doc_span_mapping: Dict[int, Tuple[Any, int]]
    ) -> Dict[Any, List[Tuple[str, float, int]]]:
        """
        Select top spans grouped by document.
        
        Args:
            span_scores: BGE scores for each span
            span_candidates: List of span candidates
            doc_span_mapping: Mapping from span index to (doc, original_index)
            
        Returns:
            Dictionary mapping documents to list of (span, score, original_index) tuples
        """
        # Create scored spans with document mapping
        scored_spans = []
        for i, (span, score) in enumerate(zip(span_candidates, span_scores)):
            doc, original_idx = doc_span_mapping[i]
            scored_spans.append((doc, span, score, original_idx))
        
        # Group by document ID (to avoid unhashable type issues)
        doc_spans = defaultdict(list)
        for doc, span, score, original_idx in scored_spans:
            # Use document ID as key instead of document object
            doc_id = getattr(doc, 'id', id(doc))  # Use doc.id if available, otherwise use object id
            doc_spans[doc_id].append((doc, span, score, original_idx))
        
        # Select top spans per document and globally
        selected_spans_by_doc = {}
        all_spans_global = []
        
        # First, collect all spans for global ranking
        for doc_id, spans_with_scores in doc_spans.items():
            for doc, span, score, original_idx in spans_with_scores:
                all_spans_global.append((doc, span, score, original_idx))
        
        # Sort globally by score
        all_spans_global.sort(key=lambda x: x[2], reverse=True)
        
        # Select top spans globally, then group by document
        top_spans_global = all_spans_global[:self.max_final_spans]
        
        for doc, span, score, original_idx in top_spans_global:
            # Use document ID as key
            doc_id = getattr(doc, 'id', id(doc))
            if doc_id not in selected_spans_by_doc:
                selected_spans_by_doc[doc_id] = []
            selected_spans_by_doc[doc_id].append((span, score, original_idx))
        
        logger.debug(f"Selected spans for {len(selected_spans_by_doc)} documents")
        return selected_spans_by_doc
    
    def _update_documents_with_spans(
        self, 
        documents: List[Any], 
        selected_spans_by_doc: Dict[Any, List[Tuple[str, float, int]]]
    ) -> List[Any]:
        """
        Update documents with selected spans.
        
        Args:
            documents: Original documents
            selected_spans_by_doc: Selected spans grouped by document ID
            
        Returns:
            Updated documents with span metadata
        """
        processed_docs = []
        
        for doc in documents:
            # Get document ID to match with selected spans
            doc_id = getattr(doc, 'id', id(doc))
            
            if doc_id in selected_spans_by_doc:
                spans_with_scores = selected_spans_by_doc[doc_id]
                
                # Extract spans and add context
                selected_spans = []
                span_scores = []
                original_indices = []
                
                for span, score, original_idx in spans_with_scores:
                    # Add context for ultra-short spans
                    expanded_span = self._add_context_to_span(span, doc)
                    selected_spans.append(expanded_span)
                    span_scores.append(score)
                    original_indices.append(original_idx)
                
                # Update document metadata
                updated_doc = self._update_document_metadata_bge(
                    doc, selected_spans, span_scores, original_indices
                )
                processed_docs.append(updated_doc)
                
                logger.debug(f"Updated document {doc_id} with {len(selected_spans)} spans")
            else:
                # Keep original document if no spans selected
                processed_docs.append(doc)
        
        return processed_docs
    
    def _add_context_to_span(self, span: str, doc: Any) -> str:
        """
        Add context to ultra-short spans by including neighboring content.
        
        Args:
            span: Original span
            doc: Document object
            
        Returns:
            Span with added context
        """
        # If span is very short, try to add context
        if len(span.strip()) < 50:  # Ultra-short threshold
            try:
                # Extract full content and find span position
                content = self._extract_document_content(doc)
                if not content:
                    return span
                
                # Find span in content and add surrounding context
                span_start = content.find(span)
                if span_start != -1:
                    # Add context before and after
                    context_before = max(0, span_start - 100)
                    context_after = min(len(content), span_start + len(span) + 100)
                    expanded_span = content[context_before:context_after].strip()
                    
                    # Ensure we don't exceed max length
                    if len(expanded_span) <= self.max_span_length:
                        return expanded_span
                
            except Exception as e:
                logger.debug(f"Failed to add context to span: {e}")
        
        return span
    
    def _update_document_metadata_bge(
        self, 
        doc: Any, 
        selected_spans: List[str], 
        span_scores: List[float],
        original_indices: List[int]
    ) -> Any:
        """
        Update document metadata with BGE-selected spans.
        
        Args:
            doc: Original document
            selected_spans: Selected span texts
            span_scores: BGE scores for spans
            original_indices: Original indices of spans
            
        Returns:
            Updated document
        """
        # Create a copy of the document to avoid modifying the original
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            # Keep original text for reference BEFORE modifying chunk_text
            if "original_chunk_text" not in doc.metadata:
                doc.metadata["original_chunk_text"] = doc.metadata.get("chunk_text", "")
            
            # Update metadata with BGE span information
            doc.metadata["selected_spans"] = selected_spans
            doc.metadata["span_scores"] = span_scores
            doc.metadata["selected_span_indices"] = original_indices
            doc.metadata["span_selection_method"] = "bge"
            doc.metadata["span_selection_applied"] = True
            
            # Replace chunk_text with selected spans joined
            doc.metadata["chunk_text"] = " ".join(selected_spans)
        
        return doc
    
    def _select_spans_bm25_fallback(self, query: str, documents: List[Any]) -> List[Any]:
        """
        Fallback BM25-only span selection when BGE is not available.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            
        Returns:
            List of documents with selected spans in metadata
        """
        logger.info("Using BM25-only fallback for span selection")
        
        # Process query terms
        query_terms = self._preprocess_text(query)
        
        processed_docs = []
        for doc in documents:
            try:
                # Extract and process document content
                content = self._extract_document_content(doc)
                if not content or len(content.strip()) < self.min_span_length:
                    logger.warning(f"No meaningful content found in document {getattr(doc, 'id', 'unknown')}")
                    processed_docs.append(doc)
                    continue
                
                # Flatten JSON if needed
                flattened_content = self._flatten_json_content(content)
                
                # Split into sentences
                sentences = self._split_into_sentences(flattened_content)
                if not sentences:
                    logger.warning(f"No valid sentences found in document {getattr(doc, 'id', 'unknown')}")
                    processed_docs.append(doc)
                    continue
                
                # Select top sentences using BM25
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                selected_indices = self._select_top_sentences_bm25(query_terms, sentences, doc_metadata)
                
                # Expand with context
                expanded_indices = self._expand_with_context(selected_indices, len(sentences))
                
                # Extract selected spans
                selected_spans = [sentences[i] for i in sorted(expanded_indices)]
                
                # Update document metadata
                updated_doc = self._update_document_metadata_bm25(
                    doc, selected_spans, expanded_indices, sentences
                )
                processed_docs.append(updated_doc)
                
                logger.debug(f"Selected {len(selected_spans)} spans from document {getattr(doc, 'id', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error processing document {getattr(doc, 'id', 'unknown')}: {str(e)}")
                processed_docs.append(doc)
        
        logger.info(f"Successfully processed {len(processed_docs)} documents with BM25 fallback")
        return processed_docs
    
    def _select_top_sentences_bm25(self, query_terms: List[str], sentences: List[str], doc_metadata: dict = None) -> List[int]:
        """
        Select top sentences using BM25 scoring.
        
        Args:
            query_terms: Preprocessed query terms
            sentences: List of sentences
            doc_metadata: Document metadata for context-aware selection
            
        Returns:
            List of indices of top sentences
        """
        if not query_terms or not sentences:
            return []
        
        # Special handling for GitHub commits - prioritize commit messages
        if doc_metadata and doc_metadata.get("data_source") == "github_commits":
            return self._select_github_commit_sentences(sentences)
        
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
        top_indices = [i for i, _ in scored_indices[:self.top_spans_per_doc]]
        
        logger.debug(f"Selected {len(top_indices)} sentences with BM25 scores: "
                    f"{[scores[i] for i in top_indices]}")
        
        return top_indices
    
    def _select_github_commit_sentences(self, sentences: List[str]) -> List[int]:
        """
        Special sentence selection for GitHub commits that prioritizes commit messages.
        
        Args:
            sentences: List of sentences from flattened JSON
            
        Returns:
            List of indices of selected sentences
        """
        # Priority order for GitHub commit sentences
        priority_patterns = [
            r'message is ',  # Commit messages (highest priority)
            r'repo_name is ',  # Repository names
            r'author_name is ',  # Author names
            r'date is ',  # Dates
            r'html_url is ',  # URLs
            r'sha is ',  # Commit hashes
        ]
        
        # Score sentences based on priority patterns
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            for j, pattern in enumerate(priority_patterns):
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Higher score for higher priority patterns
                    score = len(priority_patterns) - j
                    break
            
            scored_sentences.append((i, score, sentence))
        
        # Sort by score (descending) and then by sentence length (descending for longer commit messages)
        scored_sentences.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
        
        # Select top sentences, ensuring we get at least one commit message if available
        selected_indices = []
        commit_message_found = False
        
        for i, score, sentence in scored_sentences:
            if len(selected_indices) >= self.top_spans_per_doc:
                break
            
            # Always include commit messages
            if 'message is ' in sentence:
                selected_indices.append(i)
                commit_message_found = True
            # Include other high-priority sentences
            elif score > 0:
                selected_indices.append(i)
        
        # If no commit message found, take the first few sentences
        if not commit_message_found and sentences:
            selected_indices = list(range(min(len(sentences), self.top_spans_per_doc)))
        
        logger.debug(f"Selected {len(selected_indices)} GitHub commit sentences: "
                    f"{[sentences[i][:50] + '...' for i in selected_indices]}")
        
        return selected_indices
    
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
    
    def _update_document_metadata_bm25(
        self, 
        doc: Any, 
        selected_spans: List[str], 
        selected_indices: List[int],
        original_sentences: List[str]
    ) -> Any:
        """
        Update document metadata with BM25-selected spans.
        
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
            
            # Update metadata with BM25 span information
            doc.metadata["selected_spans"] = selected_spans
            doc.metadata["selected_span_indices"] = selected_indices
            doc.metadata["total_sentences"] = len(original_sentences)
            doc.metadata["span_selection_method"] = "bm25"
            doc.metadata["span_selection_applied"] = True
            
            # Replace chunk_text with selected spans joined
            doc.metadata["chunk_text"] = " ".join(selected_spans)
        
        return doc
