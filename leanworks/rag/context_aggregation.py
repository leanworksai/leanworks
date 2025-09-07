"""
Context Aggregation Module for RAG Pipeline

This module implements context aggregation to group related information together
instead of having fragmented individual pieces. It identifies related context items
based on common attributes like doc_id, data_source, and semantic similarity.

The aggregation works by:
1. Grouping context items by common identifiers (doc_id, data_source, etc.)
2. Merging related information into coherent chunks
3. Preserving important metadata and relationships
4. Creating more meaningful context for generation models
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class AggregatedContext:
    """Represents an aggregated context item with grouped information"""
    content: str
    metadata: Dict[str, Any]
    source_items: List[Dict[str, Any]]  # Original items that were aggregated
    aggregation_type: str  # "commit", "document", "semantic_group", etc.
    confidence: float = 1.0

class ContextAggregator:
    """
    Scalable context aggregation module that groups related context items together
    to create more coherent and meaningful context for generation models.
    
    This implementation is designed to be flexible and handle various data formats
    including structured data, unstructured text, and mixed content types.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_group_size: int = 10,
        preserve_individual_items: bool = False,
        enable_llm_analysis: bool = False,
        model_client=None,
        embedding_client=None
    ):
        """
        Initialize the ContextAggregator.
        
        Args:
            similarity_threshold: Threshold for semantic similarity grouping
            max_group_size: Maximum number of items to group together
            preserve_individual_items: Whether to keep individual items alongside aggregated ones
            enable_llm_analysis: Whether to use LLM for intelligent content analysis
            model_client: LLM client for advanced content analysis (optional)
            embedding_client: Embedding client for cosine similarity (optional)
        """
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.preserve_individual_items = preserve_individual_items
        self.enable_llm_analysis = enable_llm_analysis
        self.model_client = model_client
        self.embedding_client = embedding_client
        
        # Flexible content type detection patterns
        self.content_type_patterns = {
            'structured_key_value': [
                r'(\w+)\s+is\s+([^.]*?)(?:\s*\.\s*|$)',  # "key is value." or "key is value"
                r'(\w+):\s*([^.]*?)(?:\s*\.\s*|$)',       # "key: value." or "key: value"
                r'(\w+)\s*=\s*([^.]*?)(?:\s*\.\s*|$)',    # "key = value." or "key = value"
            ],
            'urls': [
                r'https?://[^\s]+',
                r'www\.[^\s]+',
                r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
            ],
            'dates': [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
            ],
            'emails': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'identifiers': [
                r'\b[A-Z][a-zA-Z0-9_-]*\b',  # Proper names/IDs
                r'\b[a-z]+_[a-z]+\b',        # snake_case
                r'\b[a-z]+-[a-z]+\b',        # kebab-case
                r'\b[A-Z]+_[A-Z]+\b',        # CONSTANT_CASE
            ],
            'numbers': [
                r'\b\d+(?:\.\d+)?\b',        # Basic numbers
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Large numbers with commas
                r'\b\d+(?:\.\d+)?\s*(?:req/min|ms|s|MB|GB|TB|KB|bytes?)\b'  # Numbers with units
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for content_type, patterns in self.content_type_patterns.items():
            self.compiled_patterns[content_type] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
        
        logger.info(f"ContextAggregator initialized with similarity_threshold={similarity_threshold}, "
                   f"enable_llm_analysis={enable_llm_analysis}, embedding_client={'available' if embedding_client else 'none'}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Ensure vectors are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _cap_group(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cap the number of items in a group to max_group_size.
        
        Keeps the most representative items by length (longer content is more informative).
        """
        if len(items) <= self.max_group_size:
            return items
        
        # Sort by content length (descending) to keep most informative items
        sorted_items = sorted(items, key=lambda x: len(x.get('context', '')), reverse=True)
        
        logger.debug(f"Capping group from {len(items)} to {self.max_group_size} items")
        return sorted_items[:self.max_group_size]
    
    def aggregate_context(
        self, 
        context_items: List[Dict[str, Any]], 
        query: str = None
    ) -> List[AggregatedContext]:
        """
        Aggregate context items by grouping related information together.
        Uses multiple strategies to handle different data formats and structures.
        
        Args:
            context_items: List of context items to aggregate
            query: Optional query to help with semantic grouping
            
        Returns:
            List of aggregated context items
        """
        if not context_items:
            logger.warning("No context items provided for aggregation")
            return []
        
        logger.info(f"Starting scalable context aggregation for {len(context_items)} items")
        
        # Step 1: Analyze content types and structure
        content_analysis = self._analyze_content_types(context_items)
        logger.debug(f"Step 1: Content analysis - {content_analysis}")
        
        # Step 2: Group by multiple strategies
        groups = {}
        
        # Strategy 1: Group by common identifiers (doc_id, data_source, etc.)
        identifier_groups = self._group_by_identifiers(context_items)
        groups.update(identifier_groups)
        logger.debug(f"Step 2a: Created {len(identifier_groups)} identifier groups")
        
        # Strategy 2: Group by content structure (structured vs unstructured)
        structure_groups = self._group_by_content_structure(context_items, content_analysis)
        groups.update(structure_groups)
        logger.debug(f"Step 2b: Created {len(structure_groups)} structure groups")
        
        # Strategy 3: Group by semantic similarity for remaining items
        semantic_groups = self._group_by_semantic_similarity(context_items, query)
        groups.update(semantic_groups)
        logger.debug(f"Step 2c: Created {len(semantic_groups)} semantic groups")
        
        # Step 3: Create aggregated contexts using flexible strategies
        aggregated_contexts = self._create_flexible_aggregated_contexts(groups, content_analysis)
        logger.info(f"Step 3: Created {len(aggregated_contexts)} aggregated contexts")
        
        # Step 4: Sort by relevance and recency
        sorted_contexts = self._sort_contexts_by_relevance(aggregated_contexts, query)
        
        return sorted_contexts
    
    def _analyze_content_types(self, context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the content types and structures present in the context items."""
        analysis = {
            'structured_count': 0,
            'unstructured_count': 0,
            'mixed_count': 0,
            'content_types': defaultdict(int),
            'common_patterns': defaultdict(int),
            'data_sources': set(),
            'avg_length': 0
        }
        
        total_length = 0
        
        for item in context_items:
            context = item.get('context', '')
            total_length += len(context)
            
            # Track data sources
            analysis['data_sources'].add(item.get('data_source', 'unknown'))
            
            # Analyze content structure
            is_structured = self._is_structured_content(context)
            is_unstructured = self._is_unstructured_content(context)
            
            if is_structured and not is_unstructured:
                analysis['structured_count'] += 1
            elif is_unstructured and not is_structured:
                analysis['unstructured_count'] += 1
            else:
                analysis['mixed_count'] += 1
            
            # Detect content types
            for content_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(context):
                        analysis['content_types'][content_type] += 1
                        break
        
        analysis['avg_length'] = total_length / len(context_items) if context_items else 0
        analysis['data_sources'] = list(analysis['data_sources'])
        
        return analysis
    
    def _is_structured_content(self, text: str) -> bool:
        """Check if content appears to be structured (key-value pairs, etc.)."""
        structured_patterns = self.compiled_patterns['structured_key_value']
        for pattern in structured_patterns:
            if pattern.search(text.strip()):
                return True
        return False
    
    def _is_unstructured_content(self, text: str) -> bool:
        """Check if content appears to be unstructured (natural language, etc.)."""
        # Simple heuristic: if it's not structured and has multiple words/sentences
        words = text.split()
        sentences = text.split('.')
        return len(words) > 5 and len(sentences) > 1 and not self._is_structured_content(text)
    
    def _group_by_content_structure(
        self, 
        context_items: List[Dict[str, Any]], 
        content_analysis: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group context items by their content structure."""
        groups = defaultdict(list)
        
        # Group structured content by common patterns
        structured_items = []
        unstructured_items = []
        
        for item in context_items:
            context = item.get('context', '')
            if self._is_structured_content(context):
                structured_items.append(item)
            else:
                unstructured_items.append(item)
        
        # Group structured items by common keys/patterns
        if structured_items:
            key_groups = defaultdict(list)
            for item in structured_items:
                context = item.get('context', '')
                # Extract the key from structured content
                key = self._extract_primary_key(context)
                if key:
                    key_groups[key].append(item)
            
            # Only create groups with multiple items
            for key, items in key_groups.items():
                if len(items) > 1:
                    groups[f"structured_{key}"] = items
        
        # Group unstructured items by topic similarity
        if unstructured_items and len(unstructured_items) > 1:
            # Simple topic grouping based on common words
            topic_groups = defaultdict(list)
            for item in unstructured_items:
                context = item.get('context', '')
                # Extract topic keywords
                topic_key = self._extract_topic_keywords(context)
                if topic_key:
                    topic_groups[topic_key].append(item)
            
            for topic, items in topic_groups.items():
                if len(items) > 1:
                    groups[f"topic_{topic}"] = items
        
        return groups
    
    def _extract_primary_key(self, text: str) -> Optional[str]:
        """Extract the primary key from structured content."""
        structured_patterns = self.compiled_patterns['structured_key_value']
        for pattern in structured_patterns:
            match = pattern.search(text.strip())
            if match:
                return match.group(1).lower()
        return None
    
    def _extract_topic_keywords(self, text: str) -> Optional[str]:
        """Extract topic keywords from unstructured content."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        if meaningful_words:
            # Return the first few meaningful words as topic key
            return '_'.join(meaningful_words[:3])
        return None
    
    def _group_by_identifiers(self, context_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group context items by common identifiers like doc_id, data_source, etc."""
        groups = defaultdict(list)
        
        for item in context_items:
            # Extract key identifiers
            doc_id = item.get('doc_id', 'unknown')
            data_source = item.get('data_source', 'unknown')
            
            # Create a composite key for grouping
            # For git commits, group by the base doc_id (without chunk suffix)
            if '_chunk_' in doc_id:
                base_doc_id = doc_id.split('_chunk_')[0]
                group_key = f"{data_source}:{base_doc_id}"
            else:
                group_key = f"{data_source}:{doc_id}"
            
            groups[group_key].append(item)
        
        # Filter out groups with only one item (no aggregation needed)
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def _group_by_semantic_similarity(
        self, 
        context_items: List[Dict[str, Any]], 
        query: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group context items by semantic similarity with O(n) optimization."""
        groups = defaultdict(list)
        
        # Pre-bucket items by (data_source, doc_id base) to reduce pairwise comparisons
        buckets = defaultdict(list)
        for i, item in enumerate(context_items):
            data_source = item.get('data_source', 'unknown')
            doc_id = item.get('doc_id', 'unknown')
            
            # Extract base doc_id (remove chunk suffix if present)
            if '_chunk_' in doc_id:
                base_doc_id = doc_id.split('_chunk_')[0]
            else:
                base_doc_id = doc_id
            
            bucket_key = f"{data_source}:{base_doc_id}"
            buckets[bucket_key].append((i, item))
        
        logger.debug(f"Pre-bucketed {len(context_items)} items into {len(buckets)} buckets")
        
        # Process each bucket
        processed_items = set()
        for bucket_key, bucket_items in buckets.items():
            if len(bucket_items) == 1:
                # Skip single-item buckets - no semantic grouping needed
                continue
            
            # For multi-item buckets, perform semantic grouping
            for i, (idx, item) in enumerate(bucket_items):
                if idx in processed_items:
                    continue
                    
                current_group = [item]
                processed_items.add(idx)
                
                # Only check similarity within the same bucket
                for j, (other_idx, other_item) in enumerate(bucket_items[i+1:], i+1):
                    if other_idx in processed_items:
                        continue
                        
                    if self._are_semantically_similar(item, other_item, query):
                        current_group.append(other_item)
                        processed_items.add(other_idx)
                
                # Only create groups with multiple items
                if len(current_group) > 1:
                    group_key = f"semantic_group_{len(groups)}"
                    groups[group_key] = current_group
        
        # Also check for cross-bucket semantic similarity for remaining unprocessed items
        # This is a fallback for items that might be semantically similar across different buckets
        remaining_items = [(i, item) for i, item in enumerate(context_items) if i not in processed_items]
        
        if len(remaining_items) > 1:
            logger.debug(f"Checking cross-bucket similarity for {len(remaining_items)} remaining items")
            
            for i, (idx, item) in enumerate(remaining_items):
                if idx in processed_items:
                    continue
                    
                current_group = [item]
                processed_items.add(idx)
                
                # Check similarity with other remaining items
                for j, (other_idx, other_item) in enumerate(remaining_items[i+1:], i+1):
                    if other_idx in processed_items:
                        continue
                        
                    if self._are_semantically_similar(item, other_item, query):
                        current_group.append(other_item)
                        processed_items.add(other_idx)
                
                # Only create groups with multiple items
                if len(current_group) > 1:
                    group_key = f"semantic_group_{len(groups)}"
                    groups[group_key] = current_group
        
        logger.debug(f"Created {len(groups)} semantic groups from {len(context_items)} items")
        return groups
    
    def _are_semantically_similar(
        self, 
        item1: Dict[str, Any], 
        item2: Dict[str, Any], 
        query: str = None
    ) -> bool:
        """Check if two context items are semantically similar.
        
        Requires both lexical similarity (Jaccard >= 0.35) and optionally 
        embedding similarity (cosine >= 0.7) if embeddings are available.
        """
        context1 = item1.get('context', '').lower()
        context2 = item2.get('context', '').lower()
        
        # Check for common keywords (lexical similarity)
        words1 = set(context1.split())
        words2 = set(context2.split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Require minimum lexical similarity
        if jaccard_similarity < 0.35:
            return False
        
        # If embedding client is available, also check cosine similarity
        if self.embedding_client:
            try:
                # Get embeddings for both contexts
                embedding1 = self.embedding_client.get_embedding(context1, "RETRIEVAL_DOCUMENT")
                embedding2 = self.embedding_client.get_embedding(context2, "RETRIEVAL_DOCUMENT")
                
                # Calculate cosine similarity
                cosine_sim = self._cosine_similarity(embedding1, embedding2)
                
                # Require both lexical and embedding similarity
                return jaccard_similarity >= 0.35 and cosine_sim >= 0.7
                
            except Exception as e:
                logger.warning(f"Error computing embeddings for similarity: {e}")
                # Fall back to lexical similarity only if embedding fails
                return jaccard_similarity >= 0.35
        
        # If no embedding client, use lexical similarity only with higher threshold
        return jaccard_similarity >= 0.35
    
    def _extract_context_type(self, context: str) -> str:
        """Extract the type of context based on patterns."""
        context_lower = context.lower()
        
        if any(pattern in context_lower for pattern in ['repo_name', 'author_name', 'author_email', 'date is', 'html_url']):
            return 'commit'
        elif any(pattern in context_lower for pattern in ['title is', 'content is', 'url is']):
            return 'document'
        else:
            return 'metadata'
    
    def _create_flexible_aggregated_contexts(
        self, 
        groups: Dict[str, List[Dict[str, Any]]], 
        content_analysis: Dict[str, Any]
    ) -> List[AggregatedContext]:
        """Create aggregated contexts using flexible strategies based on content analysis."""
        aggregated_contexts = []
        seen_groups = set()
        used_items = set()  # Track individual items that have been used
        
        for group_key, items in groups.items():
            if not items:
                continue
            
            # Cap the group size first
            items = self._cap_group(items)
            
            # Create fingerprint of source items to detect duplicates
            item_ids = []
            for item in items:
                # Use id if available, otherwise fall back to doc_id, then context hash
                item_id = item.get("id") or item.get("doc_id", "")
                if not item_id:
                    # Create a hash of the context as fallback identifier
                    context = item.get("context", "")
                    item_id = hashlib.md5(context.encode()).hexdigest()[:8]
                item_ids.append(item_id)
            
            # Create fingerprint from sorted item IDs
            fingerprint = hashlib.md5(("|".join(sorted(item_ids))).encode()).hexdigest()
            
            # Skip if we've already seen this combination of source items
            if fingerprint in seen_groups:
                logger.debug(f"Skipping duplicate group {group_key} with fingerprint {fingerprint[:8]}")
                continue
            
            # Check if any items in this group have already been used
            group_item_ids = set(item_ids)
            if group_item_ids.intersection(used_items):
                logger.debug(f"Skipping group {group_key} - contains already used items")
                continue
            
            # Determine aggregation strategy based on group type and content
            if group_key.startswith('gitlab_commits:') or 'commit' in group_key.lower():
                # Use commit-specific aggregation
                agg_context = self._create_commit_aggregation(group_key, items)
            elif group_key.startswith('structured_'):
                # Use structured content aggregation
                agg_context = self._create_structured_aggregation(group_key, items)
            elif group_key.startswith('topic_'):
                # Use topic-based aggregation
                agg_context = self._create_topic_aggregation(group_key, items)
            elif group_key.startswith('semantic_group_'):
                # Use semantic aggregation
                agg_context = self._create_semantic_aggregation(group_key, items)
            else:
                # Use generic aggregation
                agg_context = self._create_generic_aggregation(group_key, items)
            
            if agg_context:
                seen_groups.add(fingerprint)
                used_items.update(group_item_ids)  # Mark these items as used
                aggregated_contexts.append(agg_context)
                logger.debug(f"Created aggregated context for group {group_key} with {len(items)} items")
        
        logger.info(f"Created {len(aggregated_contexts)} unique aggregated contexts from {len(groups)} groups")
        return aggregated_contexts
    
    def _create_structured_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context for structured content (key-value pairs)."""
        if not items:
            return None
        
        # Extract and organize key-value pairs
        key_value_pairs = {}
        data_source = items[0].get('data_source', 'unknown')
        
        for item in items:
            context = item.get('context', '')
            # Extract all key-value pairs using flexible patterns
            for pattern in self.compiled_patterns['structured_key_value']:
                for match in pattern.finditer(context.strip()):
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    # Keep first occurrence of each key (handle collisions)
                    if key not in key_value_pairs:
                        key_value_pairs[key] = value
                    # Budget limit to prevent excessive extraction
                    if len(key_value_pairs) >= 50:
                        break
                # Break outer loop if budget reached
                if len(key_value_pairs) >= 50:
                    break
            # Break item loop if budget reached
            if len(key_value_pairs) >= 50:
                break
        
        if not key_value_pairs:
            return None
        
        # Create organized content
        content_parts = []
        for key, value in sorted(key_value_pairs.items()):
            content_parts.append(f"{key}: {value}")
        
        aggregated_content = "\n".join(content_parts)
        
        return AggregatedContext(
            content=aggregated_content,
            metadata={
                'data_source': data_source,
                'aggregation_type': 'structured',
                'group_key': group_key,
                'item_count': len(items),
                'key_count': len(key_value_pairs)
            },
            source_items=items,
            aggregation_type='structured',
            confidence=0.9
        )
    
    def _create_topic_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context for topic-based grouping."""
        if not items:
            return None
        
        # Combine contexts with topic organization
        contexts = [item.get('context', '') for item in items]
        data_source = items[0].get('data_source', 'unknown')
        
        # Remove duplicates and organize by topic
        unique_contexts = []
        seen = set()
        
        for context in contexts:
            context_lower = context.lower()
            if context_lower not in seen:
                unique_contexts.append(context)
                seen.add(context_lower)
        
        # Create topic-organized content
        if len(unique_contexts) == 1:
            aggregated_content = unique_contexts[0]
        else:
            # Group by topic and create organized content
            topic_sections = []
            for context in unique_contexts:
                topic_sections.append(f"• {context}")
            aggregated_content = "\n".join(topic_sections)
        
        return AggregatedContext(
            content=aggregated_content,
            metadata={
                'data_source': data_source,
                'aggregation_type': 'topic',
                'group_key': group_key,
                'item_count': len(items)
            },
            source_items=items,
            aggregation_type='topic',
            confidence=0.8
        )
    
    def _create_generic_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context using generic strategies."""
        if not items:
            return None
        
        # Use LLM analysis if available and enabled
        if self.enable_llm_analysis and self.model_client:
            return self._create_llm_aggregation(group_key, items)
        
        # Fallback to simple combination
        contexts = [item.get('context', '') for item in items]
        data_source = items[0].get('data_source', 'unknown')
        
        # Remove duplicates
        unique_contexts = []
        seen = set()
        
        for context in contexts:
            context_lower = context.lower()
            if context_lower not in seen:
                unique_contexts.append(context)
                seen.add(context_lower)
        
        # Create simple aggregated content
        if len(unique_contexts) == 1:
            aggregated_content = unique_contexts[0]
        else:
            aggregated_content = "\n".join([f"- {ctx}" for ctx in unique_contexts])
        
        return AggregatedContext(
            content=aggregated_content,
            metadata={
                'data_source': data_source,
                'aggregation_type': 'generic',
                'group_key': group_key,
                'item_count': len(items)
            },
            source_items=items,
            aggregation_type='generic',
            confidence=0.6
        )
    
    def _create_llm_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context using LLM analysis."""
        if not self.model_client:
            return None
        
        try:
            contexts = [item.get('context', '') for item in items]
            data_source = items[0].get('data_source', 'unknown')
            
            # Create prompt for LLM analysis
            context_text = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])
            
            prompt = f"""Analyze the following context items and create a coherent, aggregated summary that:
1. Preserves all important information
2. Removes redundancy
3. Organizes information logically
4. Maintains critical details (numbers, names, dates, URLs, etc.)

Context items:
{context_text}

Provide a well-organized summary that combines this information effectively."""

            response = self.model_client.chat.completions.create(
                model="claude-3-haiku-20240307",  # Use a fast model for aggregation
                messages=[
                    {"role": "system", "content": "You are an expert at organizing and summarizing information. Create clear, comprehensive summaries that preserve all important details."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=256
            )
            
            aggregated_content = response.choices[0].message.content.strip()
            
            return AggregatedContext(
                content=aggregated_content,
                metadata={
                    'data_source': data_source,
                    'aggregation_type': 'llm_analyzed',
                    'group_key': group_key,
                    'item_count': len(items)
                },
                source_items=items,
                aggregation_type='llm_analyzed',
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Error in LLM aggregation: {str(e)}")
            return None
    
    def _create_aggregated_contexts(
        self, 
        identifier_groups: Dict[str, List[Dict[str, Any]]], 
        semantic_groups: Dict[str, List[Dict[str, Any]]]
    ) -> List[AggregatedContext]:
        """Create aggregated context objects from groups."""
        aggregated_contexts = []
        
        # Process identifier groups (e.g., git commits)
        for group_key, items in identifier_groups.items():
            aggregated_context = self._create_commit_aggregation(group_key, items)
            if aggregated_context:
                aggregated_contexts.append(aggregated_context)
        
        # Process semantic groups
        for group_key, items in semantic_groups.items():
            aggregated_context = self._create_semantic_aggregation(group_key, items)
            if aggregated_context:
                aggregated_contexts.append(aggregated_context)
        
        return aggregated_contexts
    
    def _create_commit_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context for git commit information."""
        if not items:
            return None
        
        # Extract commit information
        commit_info = {
            'repo_name': None,
            'author_name': None,
            'author_email': None,
            'date': None,
            'message': None,
            'html_url': None,
            'data_source': items[0].get('data_source', 'unknown')
        }
        
        # Parse each item to extract commit details
        for item in items:
            context = item.get('context', '')
            
            # Extract repo name
            if 'repo_name is ' in context:
                match = re.search(r'repo_name is ([^.]+)\.', context)
                if match:
                    commit_info['repo_name'] = match.group(1)
            
            # Extract author name
            if 'author_name is ' in context:
                match = re.search(r'author_name is ([^.]+)\.', context)
                if match:
                    commit_info['author_name'] = match.group(1).strip()
            
            # Extract author email
            if 'author_email is ' in context:
                match = re.search(r'author_email is ([^.]+)\.', context)
                if match:
                    commit_info['author_email'] = match.group(1).strip()
            
            # Extract date and message
            if 'date is ' in context and 'message is ' in context:
                match = re.search(r'date is ([^.]+)\. message is ([^.]+)\.', context)
                if match:
                    commit_info['date'] = match.group(1)
                    commit_info['message'] = match.group(2)
            
            # Extract URL
            if 'html_url is ' in context:
                match = re.search(r'html_url is ([^.]+)\.', context)
                if match:
                    commit_info['html_url'] = match.group(1)
        
        # Create aggregated content
        content_parts = []
        
        if commit_info['repo_name']:
            content_parts.append(f"Repository: {commit_info['repo_name']}")
        
        if commit_info['author_name'] and commit_info['author_email']:
            content_parts.append(f"Author: {commit_info['author_name']} ({commit_info['author_email']})")
        elif commit_info['author_name']:
            content_parts.append(f"Author: {commit_info['author_name']}")
        elif commit_info['author_email']:
            content_parts.append(f"Author: {commit_info['author_email']}")
        
        if commit_info['date'] and commit_info['message']:
            content_parts.append(f"Date: {commit_info['date']}")
            content_parts.append(f"Message: {commit_info['message']}")
        elif commit_info['date']:
            content_parts.append(f"Date: {commit_info['date']}")
        elif commit_info['message']:
            content_parts.append(f"Message: {commit_info['message']}")
        
        if commit_info['html_url']:
            content_parts.append(f"URL: {commit_info['html_url']}")
        
        if not content_parts:
            return None
        
        aggregated_content = "\n".join(content_parts)
        
        return AggregatedContext(
            content=aggregated_content,
            metadata={
                'data_source': commit_info['data_source'],
                'aggregation_type': 'commit',
                'group_key': group_key,
                'item_count': len(items)
            },
            source_items=items,
            aggregation_type='commit',
            confidence=0.9
        )
    
    def _create_semantic_aggregation(
        self, 
        group_key: str, 
        items: List[Dict[str, Any]]
    ) -> Optional[AggregatedContext]:
        """Create aggregated context for semantically similar items."""
        if not items:
            return None
        
        # Combine contexts intelligently
        contexts = [item.get('context', '') for item in items]
        
        # Remove duplicates and combine
        unique_contexts = []
        seen = set()
        
        for context in contexts:
            context_lower = context.lower()
            if context_lower not in seen:
                unique_contexts.append(context)
                seen.add(context_lower)
        
        # Create aggregated content
        if len(unique_contexts) == 1:
            aggregated_content = unique_contexts[0]
        else:
            # Combine multiple contexts with clear separation
            aggregated_content = "\n".join([f"- {ctx}" for ctx in unique_contexts])
        
        return AggregatedContext(
            content=aggregated_content,
            metadata={
                'data_source': items[0].get('data_source', 'unknown'),
                'aggregation_type': 'semantic',
                'group_key': group_key,
                'item_count': len(items)
            },
            source_items=items,
            aggregation_type='semantic',
            confidence=0.7
        )
    
    def _sort_contexts_by_relevance(
        self, 
        contexts: List[AggregatedContext], 
        query: str = None
    ) -> List[AggregatedContext]:
        """Sort aggregated contexts by relevance and recency."""
        def sort_key(context):
            # Prioritize commits over semantic groups
            type_priority = {'commit': 0, 'semantic': 1}.get(context.aggregation_type, 2)
            
            # Use confidence as secondary sort
            confidence = context.confidence
            
            # Use item count as tertiary sort (more items = more comprehensive)
            item_count = context.metadata.get('item_count', 1)
            
            return (type_priority, -confidence, -item_count)
        
        return sorted(contexts, key=sort_key)
    
    def format_aggregated_context(self, aggregated_contexts: List[AggregatedContext]) -> str:
        """Format aggregated contexts into readable text for the LLM."""
        if not aggregated_contexts:
            return ""
        
        formatted_parts = []
        
        for i, context in enumerate(aggregated_contexts):
            # Add header with aggregation info
            header = f"[{context.aggregation_type.upper()}"
            if context.metadata.get('item_count', 1) > 1:
                header += f" - {context.metadata['item_count']} items"
            header += f" - Source: {context.metadata.get('data_source', 'unknown')}]"
            
            formatted_parts.append(header)
            formatted_parts.append(context.content)
            formatted_parts.append("")  # Add blank line between contexts
        
        return "\n".join(formatted_parts)
