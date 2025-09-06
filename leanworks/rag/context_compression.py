"""
Context Compression Module for RAG Pipeline

This module implements question-aware context compression using a 2-pass approach:
- Pass A: Lossless preserve & trim with deduplication
- Pass B: Lossy glue with question-aware synthesis

The compression works after span selection to further refine and synthesize context.
"""

import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import difflib
from leanworks.setting import OTHER_MODEL

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class CompressedSpan:
    """Represents a compressed span with metadata"""
    text: str
    source: str
    doc_id: str
    original_indices: List[int]
    compression_type: str  # "preserved", "trimmed", "synthesized"
    confidence: float = 1.0

class ContextCompressor:
    """
    Context compression module that implements 2-pass question-aware compression
    after span selection.
    """
    
    def __init__(
        self,
        model_client=None,
        trim_window: int = 10,  # ±tokens around answer-bearing content
        similarity_threshold: float = 0.9,  # For deduplication
        min_span_length: int = 5,
        max_span_length: int = 200,
        preserve_patterns: List[str] = None
    ):
        """
        Initialize the ContextCompressor.
        
        Args:
            model_client: LLM client for question-aware synthesis
            trim_window: Number of tokens to keep around answer-bearing content
            similarity_threshold: Similarity threshold for deduplication (0.9 = 90%)
            min_span_length: Minimum span length to preserve
            max_span_length: Maximum span length before trimming
            preserve_patterns: Regex patterns for content that should always be preserved
        """
        self.model_client = model_client
        self.trim_window = trim_window
        self.similarity_threshold = similarity_threshold
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        
        # Default patterns for content that should be preserved
        self.preserve_patterns = preserve_patterns or [
            r'\b\d+(?:\.\d+)?\s*(?:req/min|ms|s|MB|GB|TB|KB)\b',  # Numbers with units
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Large numbers with commas
            r'\b[A-Z][a-zA-Z0-9_-]*\b',  # Proper names/IDs (capitalized)
            r'\bv?\d+\.\d+(?:\.\d+)?\b',  # Version numbers
            r'\b[A-Z0-9]{2,}\b',  # Error codes, flags
            r'\b[a-zA-Z0-9_-]+\.[a-zA-Z0-9_.-]+\b',  # JSONPath-like structures
            r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
            r'\b[A-Za-z0-9+/]{20,}={0,2}\b',  # Base64-like strings
            r'\b(?:https?|ftp)://[^\s]+\b',  # URLs
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Emails
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.preserve_patterns]
        
        logger.info(f"ContextCompressor initialized with trim_window={trim_window}, "
                   f"similarity_threshold={similarity_threshold}")
    
    def compress_context(
        self, 
        query: str, 
        documents: List[Any], 
        enable_pass_b: bool = True
    ) -> Tuple[List[CompressedSpan], Dict[str, Any]]:
        """
        Apply 2-pass context compression to documents after span selection.
        
        Args:
            query: The user query for question-aware compression
            documents: List of documents with selected spans from span selection
            enable_pass_b: Whether to enable Pass B (lossy glue synthesis)
            
        Returns:
            Tuple of (compressed spans, compression stats)
        """
        if not documents:
            logger.warning("No documents provided for context compression")
            return [], {"total_documents": 0, "compressed_spans": 0}
        
        logger.info(f"Starting 2-pass context compression for {len(documents)} documents")
        
        # Extract spans from documents
        all_spans = self._extract_spans_from_documents(documents)
        logger.info(f"Extracted {len(all_spans)} spans for compression")
        
        # Pass A: Lossless preserve & trim
        pass_a_spans = self._pass_a_lossless_compression(query, all_spans)
        logger.info(f"Pass A completed: {len(pass_a_spans)} spans after lossless compression")
        
        # Pass B: Lossy glue (question-aware synthesis)
        if enable_pass_b and self.model_client:
            final_spans = self._pass_b_lossy_synthesis(query, pass_a_spans)
            logger.info(f"Pass B completed: {len(final_spans)} spans after synthesis")
        else:
            final_spans = pass_a_spans
            logger.info("Pass B skipped (disabled or no model client)")
        
        # Generate compression statistics
        stats = self._generate_compression_stats(all_spans, final_spans)
        
        return final_spans, stats
    
    def _extract_spans_from_documents(self, documents: List[Any]) -> List[CompressedSpan]:
        """Extract spans from documents processed by span selection."""
        spans = []
        
        for doc in documents:
            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                continue
                
            # Get selected spans from span selection
            selected_spans = doc.metadata.get("selected_spans", [])
            if not selected_spans:
                # Fallback to chunk_text if no selected spans
                chunk_text = doc.metadata.get("chunk_text", "")
                if chunk_text:
                    selected_spans = [chunk_text]
            
            # Extract metadata
            source = doc.metadata.get("data_source", "unknown")
            doc_id = getattr(doc, 'id', 'unknown')
            original_indices = doc.metadata.get("selected_span_indices", list(range(len(selected_spans))))
            
            # Create CompressedSpan objects
            for i, span_text in enumerate(selected_spans):
                if len(span_text.strip()) >= self.min_span_length:
                    spans.append(CompressedSpan(
                        text=span_text.strip(),
                        source=source,
                        doc_id=doc_id,
                        original_indices=[original_indices[i] if i < len(original_indices) else i],
                        compression_type="original"
                    ))
        
        return spans
    
    def _pass_a_lossless_compression(self, query: str, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """
        Pass A: Lossless preserve & trim with deduplication.
        
        Steps:
        1. Keep only answer-bearing spans
        2. Trim to clause around hits (±tokens) while preserving critical content
        3. Canonicalize without meaning change
        4. Deduplicate near-identicals
        5. Group by source and maintain natural order
        """
        logger.info("Starting Pass A: Lossless preserve & trim")
        
        # Step 1: Filter answer-bearing spans
        answer_bearing_spans = self._filter_answer_bearing_spans(query, spans)
        logger.debug(f"Step 1: {len(answer_bearing_spans)} answer-bearing spans")
        
        # Step 2: Trim spans while preserving critical content
        trimmed_spans = self._trim_spans_preserve_critical(answer_bearing_spans)
        logger.debug(f"Step 2: {len(trimmed_spans)} spans after trimming")
        
        # Step 3: Canonicalize spans
        canonicalized_spans = self._canonicalize_spans(trimmed_spans)
        logger.debug(f"Step 3: {len(canonicalized_spans)} spans after canonicalization")
        
        # Step 4: Deduplicate near-identical spans
        deduplicated_spans = self._deduplicate_spans(canonicalized_spans)
        logger.debug(f"Step 4: {len(deduplicated_spans)} spans after deduplication")
        
        # Step 5: Group by source and maintain order
        grouped_spans = self._group_spans_by_source(deduplicated_spans)
        logger.debug(f"Step 5: {len(grouped_spans)} spans after grouping")
        
        return grouped_spans
    
    def _filter_answer_bearing_spans(self, query: str, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """Filter spans that likely contain answer-bearing content."""
        query_terms = set(query.lower().split())
        answer_bearing = []
        
        for span in spans:
            span_terms = set(span.text.lower().split())
            
            # Check for query term overlap
            overlap = len(query_terms.intersection(span_terms))
            overlap_ratio = overlap / len(query_terms) if query_terms else 0
            
            # Check for critical patterns
            has_critical_content = any(pattern.search(span.text) for pattern in self.compiled_patterns)
            
            # Keep spans with good overlap or critical content
            if overlap_ratio > 0.1 or has_critical_content or len(span.text) < 50:
                span.compression_type = "preserved"
                span.confidence = min(1.0, overlap_ratio + 0.5 if has_critical_content else overlap_ratio)
                answer_bearing.append(span)
        
        return answer_bearing
    
    def _trim_spans_preserve_critical(self, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """Trim spans to ±window around critical content while preserving important elements."""
        trimmed = []
        
        for span in spans:
            if len(span.text) <= self.max_span_length:
                trimmed.append(span)
                continue
            
            # Find critical content positions
            critical_positions = []
            for pattern in self.compiled_patterns:
                for match in pattern.finditer(span.text):
                    critical_positions.append((match.start(), match.end()))
            
            if not critical_positions:
                # No critical content, trim from center
                words = span.text.split()
                if len(words) > self.trim_window * 2:
                    start_idx = max(0, len(words) // 2 - self.trim_window)
                    end_idx = min(len(words), len(words) // 2 + self.trim_window)
                    trimmed_text = " ".join(words[start_idx:end_idx])
                else:
                    trimmed_text = span.text
            else:
                # Trim around critical content
                words = span.text.split()
                word_positions = self._get_word_positions(span.text)
                
                # Find word indices that contain critical content
                critical_word_indices = set()
                for start_pos, end_pos in critical_positions:
                    for i, (word_start, word_end) in enumerate(word_positions):
                        if word_start < end_pos and word_end > start_pos:
                            critical_word_indices.add(i)
                
                if critical_word_indices:
                    # Expand around critical words
                    min_idx = max(0, min(critical_word_indices) - self.trim_window)
                    max_idx = min(len(words), max(critical_word_indices) + self.trim_window + 1)
                    trimmed_text = " ".join(words[min_idx:max_idx])
                else:
                    trimmed_text = span.text
            
            # Create new trimmed span
            trimmed_span = CompressedSpan(
                text=trimmed_text,
                source=span.source,
                doc_id=span.doc_id,
                original_indices=span.original_indices,
                compression_type="trimmed",
                confidence=span.confidence
            )
            trimmed.append(trimmed_span)
        
        return trimmed
    
    def _get_word_positions(self, text: str) -> List[Tuple[int, int]]:
        """Get character positions of words in text."""
        positions = []
        words = text.split()
        current_pos = 0
        
        for word in words:
            start_pos = text.find(word, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(word)
                positions.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # Fallback if word not found
                positions.append((current_pos, current_pos + len(word)))
                current_pos += len(word) + 1
        
        return positions
    
    def _canonicalize_spans(self, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """Canonicalize spans without changing meaning."""
        canonicalized = []
        
        for span in spans:
            text = span.text
            
            # Normalize units and dates
            text = re.sub(r'(\d+)\s*seconds?\b', r'\1 s', text, flags=re.IGNORECASE)
            text = re.sub(r'(\d+)\s*minutes?\b', r'\1 min', text, flags=re.IGNORECASE)
            text = re.sub(r'(\d+)\s*hours?\b', r'\1 h', text, flags=re.IGNORECASE)
            
            # Normalize date formats
            text = re.sub(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', r'\1-\2-\3', text)
            
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove common boilerplate patterns
            boilerplate_patterns = [
                r'^(?:Note|Important|Warning|Info):\s*',
                r'\s*\(see also:.*?\)$',
                r'\s*\[.*?\]$',  # Remove trailing references
            ]
            
            for pattern in boilerplate_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Create canonicalized span
            canonicalized_span = CompressedSpan(
                text=text.strip(),
                source=span.source,
                doc_id=span.doc_id,
                original_indices=span.original_indices,
                compression_type=span.compression_type,
                confidence=span.confidence
            )
            
            if canonicalized_span.text:  # Only add non-empty spans
                canonicalized.append(canonicalized_span)
        
        return canonicalized
    
    def _deduplicate_spans(self, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """Deduplicate near-identical spans using character n-grams similarity."""
        if not spans:
            return spans
        
        # Group spans by similarity
        unique_spans = []
        
        for span in spans:
            # Check if this span is similar to any existing span
            is_duplicate = False
            max_similarity = self._calculate_trigram_similarity(span.text, unique_spans)
            
            if max_similarity > self.similarity_threshold:
                is_duplicate = True
            
            if not is_duplicate:
                unique_spans.append(span)
        
        logger.debug(f"Deduplication: {len(spans)} -> {len(unique_spans)} spans")
        return unique_spans
    
    def _get_character_trigrams(self, text: str) -> Set[str]:
        """Get character trigrams for similarity comparison."""
        text = text.lower().replace(' ', '')
        return {text[i:i+3] for i in range(len(text) - 2)}
    
    def _calculate_trigram_similarity(self, text: str, existing_spans: List[CompressedSpan]) -> float:
        """Calculate maximum similarity with existing spans using word-based Jaccard similarity."""
        if not existing_spans:
            return 0.0
        
        # Use word-based similarity instead of character trigrams for better results
        text_words = set(text.lower().split())
        max_similarity = 0.0
        
        for existing_span in existing_spans:
            existing_words = set(existing_span.text.lower().split())
            
            if not text_words or not existing_words:
                continue
                
            intersection = len(text_words.intersection(existing_words))
            union = len(text_words.union(existing_words))
            
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _group_spans_by_source(self, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """Group spans by source and maintain natural order."""
        # Group by source
        source_groups = defaultdict(list)
        for span in spans:
            source_groups[span.source].append(span)
        
        # Maintain order within each source group
        grouped_spans = []
        for source in sorted(source_groups.keys()):  # Sort sources for consistency
            source_spans = source_groups[source]
            # Sort by original indices to maintain natural order
            source_spans.sort(key=lambda x: min(x.original_indices))
            grouped_spans.extend(source_spans)
        
        return grouped_spans
    
    def _pass_b_lossy_synthesis(self, query: str, spans: List[CompressedSpan]) -> List[CompressedSpan]:
        """
        Pass B: Lossy glue with question-aware synthesis.
        
        Add synthesis to bridge quotes and answer question slots.
        """
        if not self.model_client:
            logger.warning("No model client provided for Pass B synthesis")
            return spans
        
        logger.info("Starting Pass B: Lossy glue with question-aware synthesis")
        
        # Group spans by source for synthesis
        source_groups = defaultdict(list)
        for span in spans:
            source_groups[span.source].append(span)
        
        synthesized_spans = []
        
        for source, source_spans in source_groups.items():
            if len(source_spans) <= 1:
                # No synthesis needed for single spans
                synthesized_spans.extend(source_spans)
                continue
            
            try:
                # Generate synthesis for this source group
                synthesis = self._generate_source_synthesis(query, source, source_spans)
                
                if synthesis:
                    # Create synthesis span
                    synthesis_span = CompressedSpan(
                        text=synthesis,
                        source=source,
                        doc_id=f"synthesis_{source}",
                        original_indices=[],
                        compression_type="synthesized",
                        confidence=0.8
                    )
                    synthesized_spans.append(synthesis_span)
                
                # Add original spans after synthesis
                synthesized_spans.extend(source_spans)
                
            except Exception as e:
                logger.error(f"Error synthesizing spans for source {source}: {str(e)}")
                # Fallback to original spans
                synthesized_spans.extend(source_spans)
        
        return synthesized_spans
    
    def _generate_source_synthesis(
        self, 
        query: str, 
        source: str, 
        spans: List[CompressedSpan]
    ) -> Optional[str]:
        """Generate question-aware synthesis for spans from a single source."""
        
        # Prepare context for synthesis
        span_texts = [span.text for span in spans]
        context = "\n".join([f"- {text}" for text in span_texts])
        
        synthesis_prompt = f"""Given the user query and document excerpts below, create a brief synthesis that:
1. Answers the question slots (what, where, limit, how to)
2. Points out any contradictions if present
3. Adds short conditions or context
4. Does NOT paraphrase critical tokens (numbers, names, codes, etc.)

User Query: {query}
Source: {source}
Document Excerpts:
{context}

Provide a 1-2 sentence synthesis that bridges these excerpts to answer the query. If there are contradictions, mention them. Keep critical details exact."""
        
        try:
            response = self.model_client.chat.completions.create(
                model=OTHER_MODEL,  # Use Claude Haiku for faster synthesis
                messages=[
                    {"role": "system", "content": "You are a precise document synthesizer. Create brief, accurate syntheses that preserve critical details."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=150
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            # Validate synthesis is not too long and doesn't repeat original content
            if len(synthesis) > 300 or any(synthesis.lower() in span.text.lower() for span in spans):
                logger.warning(f"Synthesis too long or repetitive for source {source}")
                return None
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Error generating synthesis: {str(e)}")
            return None
    
    def _generate_compression_stats(
        self, 
        original_spans: List[CompressedSpan], 
        final_spans: List[CompressedSpan]
    ) -> Dict[str, Any]:
        """Generate compression statistics."""
        
        original_length = sum(len(span.text) for span in original_spans)
        final_length = sum(len(span.text) for span in final_spans)
        
        compression_types = Counter(span.compression_type for span in final_spans)
        
        stats = {
            "original_spans": len(original_spans),
            "final_spans": len(final_spans),
            "original_length": original_length,
            "final_length": final_length,
            "compression_ratio": final_length / original_length if original_length > 0 else 0,
            "span_reduction": (len(original_spans) - len(final_spans)) / len(original_spans) if original_spans else 0,
            "compression_types": dict(compression_types),
            "avg_confidence": sum(span.confidence for span in final_spans) / len(final_spans) if final_spans else 0
        }
        
        return stats
    
    def format_compressed_context(self, compressed_spans: List[CompressedSpan]) -> str:
        """Format compressed spans into readable context for the LLM."""
        if not compressed_spans:
            return ""
        
        formatted_parts = []
        current_source = None
        
        for span in compressed_spans:
            # Add source header if changed
            if span.source != current_source:
                if current_source is not None:
                    formatted_parts.append("")  # Add blank line between sources
                formatted_parts.append(f"Source: {span.source}")
                current_source = span.source
            
            # Add span with compression indicator
            if span.compression_type == "synthesized":
                formatted_parts.append(f"[SYNTHESIS]: {span.text}")
            elif span.compression_type == "trimmed":
                formatted_parts.append(f"[EXCERPT]: {span.text}")
            else:
                formatted_parts.append(span.text)
        
        return "\n".join(formatted_parts)
