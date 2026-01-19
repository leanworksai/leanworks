"""
Tests for FactExtractor class
"""
import pytest
from leanworks.agent.fact_extractor import FactExtractor


class TestFactExtractor:
    """Test cases for FactExtractor functionality"""

    def test_extract_file_paths(self):
        """Test extracting file paths from conversation turns"""
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Please analyze the data in /tmp/analysis.json"}]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I found results in ./output/results.csv"}]
                }
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "/tmp/analysis.json" in facts["file_paths"]
        assert "./output/results.csv" in facts["file_paths"]

    def test_extract_uuids(self):
        """Test extracting UUIDs"""
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Check document a1b2c3d4-e5f6-1234-5678-123456789abc"}]
                },
                "assistant_message": None
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "a1b2c3d4-e5f6-1234-5678-123456789abc" in facts["uuids"]

    def test_extract_document_ids(self):
        """Test extracting document IDs with common prefixes"""
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Look at doc-12345 and file-67890"}]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Also check task-abc123 and proj-xyz789"}]
                }
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "doc-12345" in facts["doc_ids"]
        assert "file-67890" in facts["doc_ids"]
        assert "task-abc123" in facts["doc_ids"]
        assert "proj-xyz789" in facts["doc_ids"]

    def test_extract_storage_refs(self):
        """Test extracting storage references like response_id:xxx"""
        conversation_turns = [
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Results saved with response_id: abc123def"}]
                },
                "user_message": None
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "response_id:abc123def" in facts["storage_refs"]

    def test_extract_temp_markers(self):
        """Test extracting temp file markers"""
        conversation_turns = [
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Created temp_analysis_2024.json in /tmp/ directory"}]
                },
                "user_message": None
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "/tmp/" in facts["temp_markers"] or any("/tmp/" in marker for marker in facts["temp_markers"])
        assert any("temp_analysis" in marker for marker in facts["temp_markers"])

    def test_extract_from_tool_content(self):
        """Test extracting facts from tool_use and tool_result blocks"""
        conversation_turns = [
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "save_data", "content": "response_id: xyz789"},
                        {"type": "tool_result", "content": "File saved to /tmp/results.json"}
                    ]
                },
                "user_message": None
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        assert "response_id:xyz789" in facts["storage_refs"]
        assert "/tmp/results.json" in facts["file_paths"]

    def test_deduplication(self):
        """Test that duplicate facts are removed while preserving order"""
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "/tmp/test.json /tmp/test.json ./other.json"}]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "/tmp/test.json"}]
                }
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        # Should only appear once
        assert facts["file_paths"].count("/tmp/test.json") == 1
        assert len(facts["file_paths"]) == 2  # /tmp/test.json and ./other.json

    def test_format_facts_for_prompt(self):
        """Test formatting facts for summarization prompt"""
        facts = {
            "file_paths": ["/tmp/data.json", "/output/results.csv"],
            "doc_ids": ["doc-123", "file-456"],
            "storage_refs": ["response_id:abc"]
        }

        formatted = FactExtractor.format_facts_for_prompt(facts)

        assert "Extracted Technical Facts:" in formatted
        assert "File Paths:" in formatted
        assert "/tmp/data.json" in formatted
        assert "doc-123" in formatted
        assert "response_id:abc" in formatted

    def test_empty_facts(self):
        """Test handling of conversations with no extractable facts"""
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, how are you?"}]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'm doing well, thank you!"}]
                }
            }
        ]

        facts = FactExtractor.extract_facts(conversation_turns)

        # Should have empty lists for all fact types
        for fact_list in facts.values():
            assert fact_list == []

    def test_get_fact_summary(self):
        """Test getting summary counts of extracted facts"""
        facts = {
            "file_paths": ["/tmp/a.json", "/tmp/b.json"],
            "doc_ids": ["doc-1"],
            "uuids": []
        }

        summary = FactExtractor.get_fact_summary(facts)

        assert summary["file_paths"] == 2
        assert summary["doc_ids"] == 1
        assert summary["uuids"] == 0

    def test_regex_patterns_are_safe(self):
        """Test that regex patterns don't cause issues with edge cases"""
        # Test with text that might cause regex issues
        conversation_turns = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "File: /tmp/test[file].json and (special) chars"}]
                },
                "assistant_message": None
            }
        ]

        # Should not raise exceptions
        facts = FactExtractor.extract_facts(conversation_turns)

        # Should still extract the valid file path
        assert "/tmp/test[file].json" in facts["file_paths"]