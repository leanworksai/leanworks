"""
FactExtractor extracts structured facts from conversation turns.

This utility identifies and extracts important operational details like file paths,
resource IDs, and storage references that should be preserved during summarization.
"""
import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FactExtractor:
    """
    Extracts structured facts from conversation turns to preserve during summarization.
    """

    # Regex patterns for different types of facts
    PATTERNS = {
        "file_paths": r'(/[\w/.-]+\.\w+|\.{0,2}/[\w/.-]+)',  # File paths (absolute/relative)
        "uuids": r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',  # UUIDs
        "doc_ids": r'\b(doc-|file-|task-|proj-)[a-zA-Z0-9]+\b',  # Document/resource IDs
        "storage_refs": r'\b(response_id|query_id|index_id):\s*([a-zA-Z0-9-]+)',  # Storage references
        "temp_markers": r'/tmp/[\w.-]+|\btemp_[\w.-]+',  # Temp file markers
        "api_endpoints": r'\bhttps?://[^\s<>"{}|\\^`]+',  # API endpoints/URLs
        "resource_patterns": r'\b(id|ref|key):\s*([a-zA-Z0-9_-]+)',  # Generic resource patterns
    }

    @classmethod
    def extract_facts(cls, conversation_turns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract structured facts from conversation turns.

        Args:
            conversation_turns: List of conversation turn dictionaries

        Returns:
            Dictionary mapping fact types to lists of extracted values
        """
        facts = {
            "file_paths": [],
            "uuids": [],
            "doc_ids": [],
            "storage_refs": [],
            "temp_markers": [],
            "api_endpoints": [],
            "resource_patterns": []
        }

        for turn in conversation_turns:
            # Extract from user message
            user_message = turn.get("user_message", {})
            cls._extract_from_message(user_message, facts)

            # Extract from assistant message
            assistant_message = turn.get("assistant_message")
            if assistant_message:
                cls._extract_from_message(assistant_message, facts)

        # Remove duplicates while preserving order
        for fact_type in facts:
            facts[fact_type] = cls._deduplicate_preserve_order(facts[fact_type])

        logger.debug(f"Extracted facts: {dict((k, len(v)) for k, v in facts.items())}")
        return facts

    @classmethod
    def _extract_from_message(cls, message: Dict[str, Any], facts: Dict[str, List[str]]) -> None:
        """
        Extract facts from a single message.

        Args:
            message: Message dictionary in Claude format
            facts: Facts dictionary to update
        """
        if not isinstance(message, dict):
            return

        content = message.get("content", [])

        # Handle different content formats
        if isinstance(content, str):
            cls._extract_from_text(content, facts)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        cls._extract_from_text(block.get("text", ""), facts)
                    elif block_type in ["tool_use", "tool_result"]:
                        # Extract from tool-related content
                        tool_content = str(block.get("content", ""))
                        cls._extract_from_text(tool_content, facts)

                        # Also check for specific tool fields
                        if block_type == "tool_use":
                            tool_name = block.get("name", "")
                            if tool_name:
                                facts["resource_patterns"].append(f"tool:{tool_name}")

    @classmethod
    def _extract_from_text(cls, text: str, facts: Dict[str, List[str]]) -> None:
        """
        Extract facts from text using regex patterns.

        Args:
            text: Text to extract facts from
            facts: Facts dictionary to update
        """
        if not text or not isinstance(text, str):
            return

        for pattern_name, pattern in cls.PATTERNS.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Handle different pattern formats
                    if pattern_name == "storage_refs":
                        # storage_refs pattern captures tuples (type, value)
                        for match in matches:
                            if isinstance(match, tuple):
                                facts[pattern_name].append(f"{match[0]}:{match[1]}")
                            else:
                                facts[pattern_name].append(match)
                    elif pattern_name == "resource_patterns":
                        # resource_patterns captures tuples (type, value)
                        for match in matches:
                            if isinstance(match, tuple):
                                facts[pattern_name].append(f"{match[0]}:{match[1]}")
                            else:
                                facts[pattern_name].append(match)
                    else:
                        # Other patterns return strings directly
                        facts[pattern_name].extend(matches)
            except re.error as e:
                logger.warning(f"Regex error in pattern {pattern_name}: {e}")

    @classmethod
    def _deduplicate_preserve_order(cls, items: List[str]) -> List[str]:
        """
        Remove duplicates from list while preserving order.

        Args:
            items: List of items

        Returns:
            Deduplicated list
        """
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @classmethod
    def format_facts_for_prompt(cls, facts: Dict[str, List[str]]) -> str:
        """
        Format extracted facts for inclusion in summarization prompt.

        Args:
            facts: Facts dictionary

        Returns:
            Formatted string for prompt inclusion
        """
        if not facts or all(not v for v in facts.values()):
            return ""

        lines = ["Extracted Technical Facts:"]
        for fact_type, values in facts.items():
            if values:
                lines.append(f"  {fact_type.replace('_', ' ').title()}:")
                for value in values[:10]:  # Limit to prevent prompt bloat
                    lines.append(f"    - {value}")
                if len(values) > 10:
                    lines.append(f"    ... and {len(values) - 10} more")

        return "\n".join(lines)

    @classmethod
    def get_fact_summary(cls, facts: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Get summary of extracted facts (counts by type).

        Args:
            facts: Facts dictionary

        Returns:
            Dictionary with fact type counts
        """
        return {fact_type: len(values) for fact_type, values in facts.items()}