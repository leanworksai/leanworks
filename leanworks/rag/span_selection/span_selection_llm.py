"""
LLM-based Span Selection Module for RAG Pipeline

This module performs span-level selection from reranked documents using LLM semantic scoring.
It extracts the most relevant spans using sliding windows or sentences with LLM reranking.
Supports both regular text and JSON/JSON Array formats with JSONPath flattening.
Uses efficient LLM scoring with capping to maintain performance.
"""

import json
import logging
import re
import math
import asyncio
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from .span_selection_base import BaseSpanSelector
from leanworks.setting import RERANK_MODEL

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

class LLMSpanSelector(BaseSpanSelector):
    """
    LLM-based span selection module that selects the most relevant spans from documents
    using LLM semantic scoring with sliding windows or sentence-based candidates.
    """
    
    def __init__(
        self,
        top_spans_per_doc: int = 6,
        context_window: int = 1,
        min_span_length: int = 10,
        max_span_length: int = 500,
        llm_reranker: Optional[Any] = None,
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
        Initialize the LLMSpanSelector.
        
        Args:
            top_spans_per_doc: Number of top spans to select per document (default 6)
            context_window: Number of neighbor sentences to include (±1)
            min_span_length: Minimum span length to consider
            max_span_length: Maximum span length to consider
            llm_reranker: LLM reranker instance for semantic scoring
            use_sliding_windows: Whether to use sliding windows vs sentences
            window_size: Size of sliding windows in tokens (default 96)
            window_stride: Stride of sliding windows in tokens (default 48)
            max_span_candidates: Maximum span candidates to score with LLM (default 60)
            max_final_spans: Maximum final spans to return (default 18)
            use_bm25_prefilter: Whether to use BM25 pre-filtering before LLM
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        super().__init__()
        self.top_spans_per_doc = max(3, min(10, top_spans_per_doc))
        self.context_window = context_window
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.llm_reranker = llm_reranker
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
        
        logger.info(f"LLMSpanSelector initialized with top_spans_per_doc={self.top_spans_per_doc}, "
                   f"context_window={self.context_window}, use_sliding_windows={self.use_sliding_windows}, "
                   f"window_size={self.window_size}, max_span_candidates={self.max_span_candidates}, "
                   f"llm_reranker={'available' if self.llm_reranker else 'none'}")
    
    def select_spans(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Select relevant spans from a list of documents using LLM semantic scoring.
        
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
        
        # If no LLM reranker available, fall back to BM25-only approach
        if not self.llm_reranker:
            logger.warning("No LLM reranker available, falling back to BM25-only span selection")
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
                    all_span_candidates.extend(spans)  # spans are tuples (text,start,end)
                    for j, (t, s, e) in enumerate(spans):
                        doc_span_mapping[start_idx + j] = (doc, s, e)  # keep offsets
                    
                    logger.debug(f"Generated {len(spans)} span candidates from document {getattr(doc, 'id', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error processing document {getattr(doc, 'id', 'unknown')}: {str(e)}")
                    continue
            
            if not all_span_candidates:
                logger.warning("No span candidates generated from any documents")
                # Return documents sorted by any available scores (e.g., from reranker)
                return self._sort_documents_by_available_scores(documents)
            
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
            
            # Step 4: Score spans with LLM
            # Extract just the text for scoring
            span_texts = [c[0] if isinstance(c, tuple) else c for c in all_span_candidates]
            span_scores = self._score_spans_with_llm(query, span_texts)
            
            # Step 5: Select top spans and group by document
            selected_spans_by_doc = self._select_top_spans_by_document(
                span_scores, all_span_candidates, doc_span_mapping
            )
            
            # Step 6: Update documents with selected spans
            processed_docs = self._update_documents_with_spans(documents, selected_spans_by_doc)
            
            logger.info(f"Successfully processed {len(processed_docs)} documents with LLM span selection")
            return processed_docs
            
        except Exception as e:
            logger.error(f"LLM span selection failed: {str(e)}, falling back to BM25")
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
    
    def _generate_sliding_window_candidates(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Generate sliding window span candidates from text with offsets.
        
        Args:
            text: Input text
            
        Returns:
            List of (text, start, end) tuples
        """
        # Use character-based sliding windows for LLM approach
        return self._generate_character_sliding_windows(text)
    
    def _generate_character_sliding_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Generate character-based sliding window spans with positional offsets.
        Tries to respect sentence boundaries when possible.
        
        Args:
            text: Input text
            
        Returns:
            List of (text, start, end) tuples
        """
        # Rough estimation: ~4 characters per token
        char_window_size = self.window_size * 4
        char_stride = self.window_stride * 4
        
        if len(text) <= char_window_size:
            t = text.strip()
            return [(t, 0, len(t))] if len(t) >= self.min_span_length else []
        
        spans = []
        for i in range(0, max(1, len(text) - char_window_size + 1), char_stride):
            # Use consistent window size for end position calculation
            end_pos = min(i + char_window_size, len(text))
            window_text = text[i:end_pos].strip()
            
            # Try to adjust end position to respect sentence boundaries
            if end_pos < len(text):
                # Look for sentence endings within a reasonable distance
                for j in range(end_pos, min(end_pos + 50, len(text))):
                    if text[j] in '.!?':
                        end_pos = j + 1
                        window_text = text[i:end_pos].strip()
                        break
            
            if (len(window_text) >= self.min_span_length and 
                len(window_text) <= self.max_span_length):
                spans.append((window_text, i, end_pos))
        
        return spans
    
    def _generate_sentence_candidates(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Generate sentence-based span candidates from text with offsets.
        
        Args:
            text: Input text
            
        Returns:
            List of (text, start, end) tuples
        """
        sentences = self._split_into_sentences(text)
        
        # Filter sentences by length and find their positions
        filtered_sentences = []
        current_pos = 0
        
        for sentence in sentences:
            if (len(sentence) >= self.min_span_length and 
                len(sentence) <= self.max_span_length):
                # Find the sentence position in the original text
                start_pos = text.find(sentence, current_pos)
                if start_pos != -1:
                    end_pos = start_pos + len(sentence)
                    filtered_sentences.append((sentence, start_pos, end_pos))
                    current_pos = end_pos
                else:
                    # Fallback: estimate position
                    filtered_sentences.append((sentence, current_pos, current_pos + len(sentence)))
                    current_pos += len(sentence)
        
        return filtered_sentences
    
    def _bm25_prefilter(
        self, 
        query_terms: List[str], 
        span_candidates: List[Union[str, Tuple[str, int, int]]], 
        doc_span_mapping: Dict[int, Tuple[Any, int, int]]
    ) -> Tuple[List[Union[str, Tuple[str, int, int]]], Dict[int, Tuple[Any, int, int]]]:
        """
        Pre-filter span candidates using BM25 to reduce the number before LLM scoring.
        
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
            # Extract text from tuple or use string directly
            span_text = span[0] if isinstance(span, tuple) else span
            terms = self._preprocess_text(span_text)
            span_terms.append(terms)
        
        # Calculate BM25 scores
        bm25_scores = self._calculate_bm25_scores(query_terms, span_terms)
        
        # Group spans by document ID and select top spans per document
        doc_spans = defaultdict(list)
        for i, (span, score) in enumerate(zip(span_candidates, bm25_scores)):
            doc, start, end = doc_span_mapping[i]
            doc_id = getattr(doc, 'id', id(doc))  # Use document ID as key
            doc_spans[doc_id].append((i, span, score, doc, start, end))
        
        # Select top spans per document
        filtered_candidates = []
        filtered_mapping = {}
        new_idx = 0
        
        for doc_id, spans_with_scores in doc_spans.items():
            # Sort by BM25 score and take top spans per document
            spans_with_scores.sort(key=lambda x: x[2], reverse=True)
            top_spans = spans_with_scores[:self.top_spans_per_doc]
            
            for original_idx, span, score, doc, start, end in top_spans:
                filtered_candidates.append(span)
                filtered_mapping[new_idx] = (doc, start, end)
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
    
    def _score_spans_with_llm(self, query: str, span_candidates: List[str]) -> List[float]:
        """
        Score span candidates using LLM reranker directly.
        
        Args:
            query: Query string
            span_candidates: List of span candidates
            
        Returns:
            List of LLM scores
        """
        if not span_candidates:
            return []
        
        try:
            # Use the LLM reranker's direct scoring method
            if hasattr(self.llm_reranker, '_score_documents_async'):
                # Create a synchronous wrapper for the async method
                import asyncio
                import threading
                
                try:
                    loop = asyncio.get_running_loop()
                    # loop is running -> run the coro in a thread with its own loop
                    return self._run_coro_in_thread(self.llm_reranker._score_documents_async(query, span_candidates))
                except RuntimeError:
                    # no loop -> safe to run directly
                    scores = asyncio.run(self.llm_reranker._score_documents_async(query, span_candidates))
            else:
                # Fallback to simple scoring if async method not available
                scores = self._score_spans_simple(query, span_candidates)
            
            logger.debug(f"LLM scored {len(span_candidates)} span candidates with scores: {scores[:5]}...")
            return scores
            
        except Exception as e:
            logger.error(f"LLM span scoring failed: {str(e)}")
            # Return default scores as fallback
            return [0.5] * len(span_candidates)
    
    def _score_spans_simple(self, query: str, span_candidates: List[str]) -> List[float]:
        """
        Simple fallback scoring when async method is not available or fails.
        
        Args:
            query: Query string
            span_candidates: List of span candidates
            
        Returns:
            List of simple scores
        """
        # Simple keyword-based scoring as fallback
        query_terms = set(query.lower().split())
        scores = []
        
        for span in span_candidates:
            span_lower = span.lower()
            # Count term matches
            matches = sum(1 for term in query_terms if term in span_lower)
            # Normalize to 0-1 range
            score = min(1.0, matches / max(1, len(query_terms)))
            scores.append(score)
        
        logger.debug(f"Simple scoring for {len(span_candidates)} span candidates")
        return scores
    
    def _select_top_spans_by_document(
        self, 
        span_scores: List[float], 
        span_candidates: List[Union[str, Tuple[str, int, int]]], 
        doc_span_mapping: Dict[int, Tuple[Any, int, int]]
    ) -> Dict[Any, List[Tuple[str, float, int]]]:
        """
        Select top spans grouped by document.
        
        Args:
            span_scores: LLM scores for each span
            span_candidates: List of span candidates
            doc_span_mapping: Mapping from span index to (doc, original_index)
            
        Returns:
            Dictionary mapping documents to list of (span, score, original_index) tuples
        """
        # Create scored spans with document mapping
        scored_spans = []
        for i, (span, score) in enumerate(zip(span_candidates, span_scores)):
            doc, start, end = doc_span_mapping[i]
            # Extract text from tuple or use string directly
            span_text = span[0] if isinstance(span, tuple) else span
            scored_spans.append((doc, span_text, score, start, end))
        
        # Group by document ID (to avoid unhashable type issues)
        doc_spans = defaultdict(list)
        for doc, span, score, start, end in scored_spans:
            # Use document ID as key instead of document object
            doc_id = self._doc_key(doc)
            doc_spans[doc_id].append((doc, span, score, start, end))
        
        # Select top spans per document and globally
        selected_spans_by_doc = {}
        all_spans_global = []
        
        # First, collect all spans for global ranking
        for doc_id, spans_with_scores in doc_spans.items():
            for doc, span, score, start, end in spans_with_scores:
                all_spans_global.append((doc, span, score, start, end))
        
        # Sort globally by score
        all_spans_global.sort(key=lambda x: x[2], reverse=True)
        
        # Select top spans globally, then group by document
        top_spans_global = all_spans_global[:self.max_final_spans]
        
        # Build per-doc buckets with offsets for NMS
        per_doc = defaultdict(list)
        for doc, span, score, start, end in top_spans_global:
            doc_id = self._doc_key(doc)
            per_doc[doc_id].append((span, score, start, end))
        
        # Apply NMS + exact/near dedup + per-doc cap
        for doc_id, items in per_doc.items():
            # Use stricter IoU threshold to reduce overlapping spans
            items = self._nms_char_ranges(items, iou_thresh=0.5)
            # Apply per-doc cap
            items = items[:self.top_spans_per_doc]
            selected_spans_by_doc[doc_id] = [(text, score, start) for text, score, start, end in items]
        
        logger.debug(f"Selected spans for {len(selected_spans_by_doc)} documents")
        return selected_spans_by_doc
    
    def _update_documents_with_spans(
        self, 
        documents: List[Any], 
        selected_spans_by_doc: Dict[Any, List[Tuple[str, float, int]]]
    ) -> List[Any]:
        """
        Update documents with selected spans, maintaining score-based ordering.
        
        Args:
            documents: Original documents
            selected_spans_by_doc: Selected spans grouped by document ID
            
        Returns:
            Updated documents with span metadata, sorted by document scores
        """
        # First, collect all documents with their scores for sorting
        docs_with_scores = []
        
        for doc in documents:
            # Get document ID to match with selected spans
            doc_id = getattr(doc, 'id', id(doc))
            
            # Get the highest span score for this document as its overall score
            doc_score = 0.0
            if doc_id in selected_spans_by_doc:
                spans_with_scores = selected_spans_by_doc[doc_id]
                if spans_with_scores:
                    # Use the highest span score as the document score
                    doc_score = max(score for _, score, _ in spans_with_scores)
            
            docs_with_scores.append((doc, doc_score, doc_id))
        
        # Sort documents by their scores (descending)
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Process documents in score order
        processed_docs = []
        
        for doc, doc_score, doc_id in docs_with_scores:
            if doc_id in selected_spans_by_doc:
                spans_with_scores = selected_spans_by_doc[doc_id]
                
                # Extract spans and add context
                selected_spans = []
                span_scores = []
                original_indices = []
                
                for span, score, start in spans_with_scores:
                    # Add context for ultra-short spans using offsets
                    expanded_span = self._add_context_to_span((span, start, start + len(span)), doc)
                    selected_spans.append(expanded_span)
                    span_scores.append(score)
                    original_indices.append(start)
                
                # Final safety: dedup again after context expansion
                packed = list(zip(selected_spans, span_scores, original_indices))
                packed = self._dedup_spans_exact(packed)
                if packed:
                    selected_spans, span_scores, original_indices = map(list, zip(*packed))
                
                # Update document metadata
                updated_doc = self._update_document_metadata_llm(
                    doc, selected_spans, span_scores, original_indices
                )
                processed_docs.append(updated_doc)
                
                logger.debug(f"Updated document {doc_id} with {len(selected_spans)} spans (score: {doc_score:.3f})")
            else:
                # Keep original document if no spans selected
                processed_docs.append(doc)
        
        return processed_docs
    
    def _add_context_to_span(self, span_tuple, doc: Any) -> str:
        """
        Add context to ultra-short spans by including neighboring content using offsets.
        
        Args:
            span_tuple: (text, start, end) tuple or string for backward compatibility
            doc: Document object
            
        Returns:
            Span with added context
        """
        # Handle both tuple and string formats for backward compatibility
        try:
            text, start, end = span_tuple
        except Exception:
            # Backward compat if older string-only spans slip through
            return span_tuple if isinstance(span_tuple, str) else str(span_tuple)
        
        # If span is very short, try to add context
        if len(text.strip()) < 50:  # Ultra-short threshold
            try:
                # Extract full content
                content = self._extract_document_content(doc)
                if not content:
                    return text
                
                # Expand around the known offsets, not by searching for text
                pad = 100
                L = max(0, start - pad)
                R = min(len(content), end + pad)
                expanded = content[L:R].strip()
                
                # Ensure we don't exceed max length
                if len(expanded) <= self.max_span_length:
                    return expanded
                else:
                    # If expanded is too long, return original span
                    return content[start:end]
                
            except Exception as e:
                logger.debug(f"Failed to add context to span: {e}")
        
        return text
    
    def _update_document_metadata_llm(
        self, 
        doc: Any, 
        selected_spans: List[str], 
        span_scores: List[float],
        original_indices: List[int]
    ) -> Any:
        """
        Update document metadata with LLM-selected spans.
        
        Args:
            doc: Original document
            selected_spans: Selected span texts
            span_scores: LLM scores for spans
            original_indices: Original indices of spans
            
        Returns:
            Updated document
        """
        # Create a copy of the document to avoid modifying the original
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            # Keep original text for reference BEFORE modifying chunk_text
            if "original_chunk_text" not in doc.metadata:
                doc.metadata["original_chunk_text"] = doc.metadata.get("chunk_text", "")
            
            # Update metadata with LLM span information
            doc.metadata["selected_spans"] = selected_spans
            doc.metadata["span_scores"] = span_scores
            doc.metadata["selected_span_indices"] = original_indices
            doc.metadata["span_selection_method"] = "llm"
            doc.metadata["span_selection_applied"] = True
            
            # Replace chunk_text with selected spans joined
            doc.metadata["chunk_text"] = " ".join(selected_spans)
        
        return doc
    
    def _sort_documents_by_available_scores(self, documents: List[Any]) -> List[Any]:
        """
        Sort documents by any available scores (e.g., from reranker).
        
        Args:
            documents: List of document objects
            
        Returns:
            Documents sorted by available scores in descending order
        """
        # Try to sort by rerank_score first, then by semantic_score, then by timestamp
        def get_doc_score(doc):
            # Priority 1: rerank_score (from reranker)
            if hasattr(doc, 'rerank_score'):
                return doc.rerank_score
            # Priority 2: semantic_score (from reranker)
            if hasattr(doc, 'semantic_score'):
                return doc.semantic_score
            # Priority 3: timestamp (most recent first)
            if hasattr(doc, 'metadata') and 'timestamp' in doc.metadata:
                try:
                    return float(doc.metadata['timestamp'])
                except (ValueError, TypeError):
                    pass
            # Default: no score
            return 0.0
        
        # Sort by score (descending)
        sorted_docs = sorted(documents, key=get_doc_score, reverse=True)
        logger.debug(f"Sorted {len(sorted_docs)} documents by available scores")
        return sorted_docs
    
    def _select_spans_bm25_fallback(self, query: str, documents: List[Any]) -> List[Any]:
        """
        Fallback BM25-only span selection when LLM is not available.
        
        Args:
            query: The user query
            documents: List of document objects with metadata
            
        Returns:
            List of documents with selected spans in metadata, sorted by scores
        """
        logger.info("Using BM25-only fallback for span selection")
        
        # Process query terms
        query_terms = self._preprocess_text(query)
        
        # First pass: collect documents with their scores
        docs_with_scores = []
        
        for doc in documents:
            try:
                # Extract and process document content
                content = self._extract_document_content(doc)
                if not content or len(content.strip()) < self.min_span_length:
                    logger.warning(f"No meaningful content found in document {getattr(doc, 'id', 'unknown')}")
                    docs_with_scores.append((doc, 0.0, None, None, None, None))
                    continue
                
                # Flatten JSON if needed
                flattened_content = self._flatten_json_content(content)
                
                # Split into sentences
                sentences = self._split_into_sentences(flattened_content)
                if not sentences:
                    logger.warning(f"No valid sentences found in document {getattr(doc, 'id', 'unknown')}")
                    docs_with_scores.append((doc, 0.0, None, None, None, None))
                    continue
                
                # Select top sentences using BM25
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                selected_indices = self._select_top_sentences_bm25(query_terms, sentences, doc_metadata)
                
                # Calculate document score as the average of selected sentence scores
                if selected_indices:
                    # Get BM25 scores for all sentences
                    sentence_terms = [self._preprocess_text(s) for s in sentences]
                    all_scores = self._calculate_bm25_scores(query_terms, sentence_terms)
                    # Average score of selected sentences
                    doc_score = sum(all_scores[i] for i in selected_indices) / len(selected_indices)
                else:
                    doc_score = 0.0
                
                # Expand with context
                expanded_indices = self._expand_with_context(selected_indices, len(sentences))
                
                # Extract selected spans
                selected_spans = [sentences[i] for i in sorted(expanded_indices)]
                
                docs_with_scores.append((doc, doc_score, selected_spans, expanded_indices, sentences, doc_metadata))
                
                logger.debug(f"Selected {len(selected_spans)} spans from document {getattr(doc, 'id', 'unknown')} (score: {doc_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing document {getattr(doc, 'id', 'unknown')}: {str(e)}")
                docs_with_scores.append((doc, 0.0, None, None, None, None))
        
        # Sort documents by their BM25 scores (descending)
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Second pass: update documents in score order
        processed_docs = []
        
        for doc, doc_score, selected_spans, expanded_indices, sentences, doc_metadata in docs_with_scores:
            if selected_spans is not None:
                # Update document metadata
                updated_doc = self._update_document_metadata_bm25(
                    doc, selected_spans, expanded_indices, sentences
                )
                processed_docs.append(updated_doc)
            else:
                # Keep original document if no spans selected
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
    
    def _doc_key(self, doc):
        """
        Generate a robust document key to avoid grouping id=None docs together.
        
        Args:
            doc: Document object
            
        Returns:
            Unique document key
        """
        key = getattr(doc, 'id', None)
        if not key:
            md = getattr(doc, 'metadata', {}) or {}
            key = (md.get('doc_id') or md.get('source_id') or md.get('file_id') or
                   md.get('source') or md.get('path'))
        return key or id(doc)
    
    def _nms_char_ranges(self, items, iou_thresh=0.7):
        """
        Apply non-max suppression on character ranges to remove overlapping spans.
        
        Args:
            items: List of (text, score, start, end) tuples
            iou_thresh: IoU threshold for suppression
            
        Returns:
            List of non-overlapping items
        """
        if not items:
            return []
            
        # Sort by score (descending)
        items = sorted(items, key=lambda x: x[1], reverse=True)
        kept = []
        
        def iou(a, b):
            s1, e1 = a
            s2, e2 = b
            # Calculate intersection
            inter = max(0, min(e1, e2) - max(s1, s2))
            # Calculate union correctly: total length minus intersection
            union = (e1 - s1) + (e2 - s2) - inter
            return inter / union if union > 0 else 0.0
        
        for t, sc, s, e in items:
            # Check for overlap with already kept items
            has_overlap = False
            for _, _, ks, ke in kept:
                if iou((s, e), (ks, ke)) >= iou_thresh:
                    has_overlap = True
                    break
            
            if not has_overlap:
                kept.append((t, sc, s, e))
        
        return kept
    
    def _run_coro_in_thread(self, coro):
        """
        Run an async coroutine in a separate thread with its own event loop.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        import asyncio
        import threading
        
        out = {}
        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                out['v'] = loop.run_until_complete(coro)
            finally:
                loop.close()
        
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        return out['v']
    
    def _dedup_spans_exact(self, items):
        """
        Remove exact and near-duplicate spans from a list.
        
        Args:
            items: List of (span, score, start) or similar tuples
            
        Returns:
            List with duplicates removed
        """
        if not items:
            return []
            
        seen = set()
        deduped = []
        
        for item in items:
            # Use the span text as the key for deduplication
            span_text = item[0] if isinstance(item, (tuple, list)) else str(item)
            
            # Normalize text for comparison (remove extra whitespace, lowercase)
            normalized_text = ' '.join(span_text.lower().split())
            
            # Check for exact matches
            if normalized_text in seen:
                continue
                
            # Check for near-duplicates (very similar text)
            is_duplicate = False
            for seen_text in seen:
                if self._is_near_duplicate(normalized_text, seen_text):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(normalized_text)
                deduped.append(item)
                
        return deduped
    
    def _is_near_duplicate(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """
        Check if two texts are near-duplicates using simple similarity metrics.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if texts are similar enough to be considered duplicates
        """
        if not text1 or not text2:
            return False
            
        # Quick length check - if very different lengths, likely not duplicates
        len1, len2 = len(text1), len(text2)
        if abs(len1 - len2) / max(len1, len2) > 0.3:
            return False
            
        # Simple character-level similarity
        if len1 == len2 and text1 == text2:
            return True
            
        # Check for substring relationships
        if len1 > len2:
            if text2 in text1 and len(text2) / len(text1) >= threshold:
                return True
        else:
            if text1 in text2 and len(text1) / len(text2) >= threshold:
                return True
                
        # Simple word overlap check
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
