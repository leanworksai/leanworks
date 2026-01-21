from typing import Any, Dict, Optional, Tuple
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Response type classification"""
    STRUCTURED_SIMPLE = "structured_simple"    # Simple JSON → DuckDB
    STRUCTURED_COMPLEX = "structured_complex"  # Complex JSON → jq + file
    UNSTRUCTURED = "unstructured"              # Text → hybrid file + RAG
    MIXED = "mixed"                            # Contains both types
    SMALL = "small"                            # Small enough to include directly

class LargeResponseHandler:
    """Handles large tool responses with automatic storage routing"""
    
    # Default thresholds (can be overridden via config)
    MAX_DIRECT_TOKENS = 2000
    MAX_DIRECT_ITEMS = 50
    MAX_DIRECT_CHARS = 8000
    MIN_UNSTRUCTURED_CHARS = 1000  # Minimum to consider for RAG
    
    @classmethod
    def configure(cls, config: Dict[str, Any]):
        """Update thresholds from configuration"""
        cls.MAX_DIRECT_TOKENS = config.get("max_direct_tokens", cls.MAX_DIRECT_TOKENS)
        cls.MAX_DIRECT_ITEMS = config.get("max_direct_items", cls.MAX_DIRECT_ITEMS)
        cls.MAX_DIRECT_CHARS = config.get("max_direct_chars", cls.MAX_DIRECT_CHARS)
        cls.MIN_UNSTRUCTURED_CHARS = config.get("min_unstructured_chars", cls.MIN_UNSTRUCTURED_CHARS)
        logger.info(
            "LargeResponseHandler configured with thresholds: "
            "max_direct_tokens=%d, max_direct_items=%d, max_direct_chars=%d, min_unstructured_chars=%d",
            cls.MAX_DIRECT_TOKENS, cls.MAX_DIRECT_ITEMS, cls.MAX_DIRECT_CHARS, cls.MIN_UNSTRUCTURED_CHARS
        )
    
    @classmethod
    def classify_response(cls, result: Any) -> Tuple[ResponseType, bool]:
        """
        Classify response type and determine if it's large.
        
        Args:
            result: The tool response result
        
        Returns:
            (ResponseType, is_large) tuple
        """
        # Check size first
        is_large = cls._is_large(result)
        
        if not is_large:
            logger.info("Response classified as SMALL, size check passed")
            return (ResponseType.SMALL, False)
        
        # Classify type
        if isinstance(result, list):
            if not result:
                return (ResponseType.STRUCTURED, False)  # Empty list is small
            
            # Check first item type
            first_item = result[0]
            if isinstance(first_item, dict):
                # List of dicts - analyze complexity of first item
                from leanworks.agent.json_complexity_analyzer import JSONComplexityAnalyzer
                analysis = JSONComplexityAnalyzer.analyze(first_item)

                if analysis["level"].value == "simple":
                    logger.info("Response classified as STRUCTURED_SIMPLE (list of dicts), list size: %d", len(result))
                    return (ResponseType.STRUCTURED_SIMPLE, True)
                else:  # complex
                    logger.info("Response classified as STRUCTURED_COMPLEX (list of dicts), list size: %d, complexity: %s", 
                               len(result), analysis["level"].value)
                    return (ResponseType.STRUCTURED_COMPLEX, True)
            elif isinstance(first_item, str):
                # List of strings - could be unstructured
                total_text = ' '.join(str(item) for item in result)
                if len(total_text) > cls.MIN_UNSTRUCTURED_CHARS:
                    logger.info("Response classified as UNSTRUCTURED (list of strings), total chars: %d", len(total_text))
                    return (ResponseType.UNSTRUCTURED, True)
                logger.info("Response classified as STRUCTURED_SIMPLE (list of strings), total chars: %d", len(total_text))
                return (ResponseType.STRUCTURED_SIMPLE, True)
            else:
                # List of other types (numbers, etc.)
                logger.info("Response classified as STRUCTURED_SIMPLE (list of other types), list size: %d", len(result))
                return (ResponseType.STRUCTURED_SIMPLE, True)
        
        if isinstance(result, dict):
            # Use JSON complexity analyzer to classify dicts
            from leanworks.agent.json_complexity_analyzer import JSONComplexityAnalyzer
            analysis = JSONComplexityAnalyzer.analyze(result)

            if analysis["level"].value == "simple":
                logger.info("Response classified as STRUCTURED_SIMPLE (dict), complexity: %s", analysis["level"].value)
                return (ResponseType.STRUCTURED_SIMPLE, True)
            elif analysis["level"].value == "complex":
                logger.info("Response classified as STRUCTURED_COMPLEX (dict), complexity: %s, depth: %d", 
                           analysis["level"].value, analysis.get("depth", 0))
                return (ResponseType.STRUCTURED_COMPLEX, True)
            else:
                # Not JSON or other cases - treat as unstructured
                logger.info("Response classified as UNSTRUCTURED (dict, non-JSON), complexity: %s", analysis["level"].value)
                return (ResponseType.UNSTRUCTURED, True)
        
        if isinstance(result, str):
            # HTML content from doc tools should be treated as unstructured text
            # Check for common HTML patterns
            html_patterns = ['<p', '<div', '<h1', '<h2', '<h3', '<ul', '<ol', '<li', '<br', '<strong', '<em', '<a href']
            is_html = any(pattern in result for pattern in html_patterns)
            if is_html:
                # HTML content - definitely unstructured
                logger.info("Response classified as UNSTRUCTURED (HTML content), size: %d chars", len(result))
                return (ResponseType.UNSTRUCTURED, True)
            else:
                # Plain text - unstructured
                logger.info("Response classified as UNSTRUCTURED (plain text), size: %d chars", len(result))
                return (ResponseType.UNSTRUCTURED, True)
        
        # Default to structured for other types
        return (ResponseType.STRUCTURED, True)
    
    @classmethod
    def _is_large(cls, result: Any) -> bool:
        """Check if response exceeds size thresholds"""
        if isinstance(result, list):
            if len(result) > cls.MAX_DIRECT_ITEMS:
                logger.info("Response is large: list size %d exceeds max_direct_items %d", len(result), cls.MAX_DIRECT_ITEMS)
                return True
            # Estimate total size from first 10 items
            sample_size = min(10, len(result))
            total_chars = sum(len(str(item)) for item in result[:sample_size])
            avg_chars = total_chars / sample_size if sample_size > 0 else 0
            estimated_total = avg_chars * len(result)
            if estimated_total > cls.MAX_DIRECT_CHARS:
                logger.info("Response is large: estimated size %.0f chars exceeds max_direct_chars %d", estimated_total, cls.MAX_DIRECT_CHARS)
                return True
        
        if isinstance(result, str):
            is_large = len(result) > cls.MAX_DIRECT_CHARS
            if is_large:
                logger.info("Response is large: string size %d exceeds max_direct_chars %d", len(result), cls.MAX_DIRECT_CHARS)
            return is_large
        
        if isinstance(result, dict):
            try:
                json_str = json.dumps(result)
                is_large = len(json_str) > cls.MAX_DIRECT_CHARS
                if is_large:
                    logger.info("Response is large: dict size %d chars (as JSON) exceeds max_direct_chars %d", len(json_str), cls.MAX_DIRECT_CHARS)
                return is_large
            except (TypeError, ValueError):
                # If can't serialize, estimate from string representation
                str_len = len(str(result))
                is_large = str_len > cls.MAX_DIRECT_CHARS
                if is_large:
                    logger.info("Response is large: dict size %d chars (as string, non-serializable) exceeds max_direct_chars %d", str_len, cls.MAX_DIRECT_CHARS)
                return is_large
        
        # For other types, check string representation
        str_repr = str(result)
        is_large = len(str_repr) > cls.MAX_DIRECT_CHARS
        if is_large:
            logger.info("Response is large: %s size %d chars exceeds max_direct_chars %d", type(result).__name__, len(str_repr), cls.MAX_DIRECT_CHARS)
        return is_large
    
    @classmethod
    def _calculate_text_ratio(cls, data: Dict) -> float:
        """Calculate ratio of text content in a dict"""
        total_chars = 0
        text_chars = 0
        
        def traverse(obj):
            nonlocal total_chars, text_chars
            if isinstance(obj, str):
                total_chars += len(obj)
                text_chars += len(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)
            else:
                total_chars += len(str(obj))
        
        traverse(data)
        return text_chars / total_chars if total_chars > 0 else 0.0
    
    @classmethod
    def split_mixed_response(cls, result: Any) -> Tuple[Optional[Any], Optional[str]]:
        """
        Split mixed response into structured and unstructured parts.
        
        Args:
            result: Mixed response (dict with both structured and text data)
        
        Returns:
            (structured_part, unstructured_part) tuple
        """
        if isinstance(result, dict):
            structured_part = {}
            unstructured_parts = []
            
            for key, value in result.items():
                if isinstance(value, str) and len(value) > cls.MIN_UNSTRUCTURED_CHARS:
                    unstructured_parts.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    structured_part[key] = value
                else:
                    structured_part[key] = value
            
            unstructured_text = '\n\n'.join(unstructured_parts) if unstructured_parts else None
            
            if unstructured_parts or structured_part:
                logger.info("Split mixed response: structured_keys=%d, unstructured_parts=%d", 
                           len(structured_part), len(unstructured_parts))
            
            return (structured_part if structured_part else None, unstructured_text)
        
        # For other types, can't split meaningfully
        return (result, None)
    
    @classmethod
    def estimate_tokens(cls, data: Any) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        try:
            if isinstance(data, str):
                json_str = data
            else:
                json_str = json.dumps(data)
            tokens = len(json_str) // 4
            logger.info("Estimated tokens: %d (from %d chars)", tokens, len(json_str))
            return tokens
        except (TypeError, ValueError):
            str_len = len(str(data))
            tokens = str_len // 4
            logger.info("Estimated tokens: %d (from %d chars, non-serializable)", tokens, str_len)
            return tokens

