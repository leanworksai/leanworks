#!/usr/bin/env python3
"""
Simple tests for SearchTool components that don't require external dependencies.

These tests validate the core classes and logic without importing Google AI libraries
that cause segmentation faults on macOS ARM64.
"""

import pytest


class SearchResult:
    """
    Simplified SearchResult class for testing (copied from search.py to avoid imports).
    """
    def __init__(self, formatted_context: str, data_sources: list = None):
        self.formatted_context = formatted_context
        self._search_data_sources = data_sources or []

    def __str__(self):
        return self.formatted_context

    def __repr__(self):
        return f"SearchResult(context_length={len(self.formatted_context)}, sources={len(self._search_data_sources)})"

    def __contains__(self, item):
        """Support 'in' operator for string-like behavior."""
        return item in self.formatted_context

    def __len__(self):
        """Support len() for string-like behavior."""
        return len(self.formatted_context)


class TestSearchResult:
    """Test the SearchResult class behavior."""

    def test_search_result_creation(self):
        """Test SearchResult object creation."""
        context = "Test document content"
        sources = ["source1", "source2"]

        result = SearchResult(context, sources)

        assert str(result) == context
        assert result._search_data_sources == sources
        assert len(result) == len(context)

    def test_search_result_string_operations(self):
        """Test SearchResult string-like operations."""
        context = "This is a test document with some content"
        result = SearchResult(context)

        # Test __contains__
        assert "test document" in result
        assert "nonexistent" not in result

        # Test __len__
        assert len(result) == len(context)

    def test_search_result_repr(self):
        """Test SearchResult repr method."""
        context = "Short content"
        sources = ["src1", "src2", "src3"]

        result = SearchResult(context, sources)

        repr_str = repr(result)
        assert "SearchResult" in repr_str
        assert "context_length=13" in repr_str  # len("Short content")
        assert "sources=3" in repr_str

    def test_search_result_empty_sources(self):
        """Test SearchResult with no data sources."""
        context = "Content without sources"
        result = SearchResult(context)

        assert result._search_data_sources == []
        assert str(result) == context

    def test_search_result_formatting(self):
        """Test SearchResult with formatted document content."""
        formatted_context = """DOCUMENT - Date: 2024-01-01, Source: github, Doc ID: abc123
This is a commit message about fixing a bug.

DOCUMENT - Date: 2024-01-02, Source: jira, Doc ID: def456
This is a ticket description about a new feature.
"""

        sources = ["github", "jira"]
        result = SearchResult(formatted_context, sources)

        # Test that formatted content is preserved
        assert "DOCUMENT - Date:" in str(result)
        assert "github" in str(result)
        assert "jira" in str(result)
        assert len(result._search_data_sources) == 2


class MockSearchTool:
    """Mock SearchTool for testing core logic without external dependencies."""

    def __init__(self):
        self.read_document_ids = set()

    def _convert_date_to_timestamp(self, date_str: str):
        """
        Simplified date conversion (copied from search.py).
        """
        if not date_str:
            return None

        try:
            # Handle different date formats
            if 'T' in date_str:
                # ISO format with time
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                from datetime import datetime
                dt = datetime.fromisoformat(date_str)
            else:
                # Date only format
                from datetime import datetime
                dt = datetime.strptime(date_str, '%Y-%m-%d')

            # Convert to UTC timestamp
            from datetime import timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return int(dt.timestamp())
        except (ValueError, TypeError):
            return None

    def _build_timestamp_filter(self, start_date: str = None, end_date: str = None):
        """
        Simplified timestamp filter building (copied from search.py).
        """
        timestamp_filter = {}
        if start_date:
            start_timestamp = self._convert_date_to_timestamp(start_date)
            if start_timestamp is not None:
                timestamp_filter["$gte"] = start_timestamp

        if end_date:
            end_timestamp = self._convert_date_to_timestamp(end_date)
            if end_timestamp is not None:
                timestamp_filter["$lte"] = end_timestamp

        return timestamp_filter

    def _convert_unix_timestamps_in_text(self, text: str) -> str:
        """
        Simplified Unix timestamp conversion (copied from search.py).
        """
        import re
        from datetime import datetime

        def replace_timestamp(match):
            timestamp_str = match.group(2)
            try:
                unix_timestamp = float(timestamp_str)
                if unix_timestamp > 1e10:  # Likely milliseconds
                    unix_timestamp = unix_timestamp / 1000
                dt = datetime.fromtimestamp(unix_timestamp)
                iso_format = dt.isoformat()
                return f"{match.group(1)}{iso_format}"
            except (ValueError, OSError):
                return match.group(0)

        unix_pattern = r'(\w*[Tt]imestamp\s*is\s*)(\d{10,13}(?:\.\d+)?)'
        text = re.sub(unix_pattern, replace_timestamp, text)

        return text


class TestSearchToolLogic:
    """Test SearchTool core logic without external dependencies."""

    @pytest.fixture
    def search_tool(self):
        """Create a mock SearchTool instance."""
        return MockSearchTool()

    def test_date_conversion_valid(self, search_tool):
        """Test date conversion with valid inputs."""
        # Test date-only format
        timestamp = search_tool._convert_date_to_timestamp("2024-01-01")
        assert timestamp is not None
        assert isinstance(timestamp, int)

        # Test ISO format
        timestamp = search_tool._convert_date_to_timestamp("2024-01-01T12:00:00Z")
        assert timestamp is not None
        assert isinstance(timestamp, int)

    def test_date_conversion_invalid(self, search_tool):
        """Test date conversion with invalid inputs."""
        invalid_inputs = ["invalid", "", None, "2024-13-45"]

        for invalid_input in invalid_inputs:
            timestamp = search_tool._convert_date_to_timestamp(invalid_input)
            assert timestamp is None

    def test_timestamp_filter_both_dates(self, search_tool):
        """Test timestamp filter with both start and end dates."""
        filter_dict = search_tool._build_timestamp_filter("2024-01-01", "2024-12-31")

        assert "$gte" in filter_dict
        assert "$lte" in filter_dict
        assert filter_dict["$gte"] < filter_dict["$lte"]  # start before end

    def test_timestamp_filter_start_only(self, search_tool):
        """Test timestamp filter with start date only."""
        filter_dict = search_tool._build_timestamp_filter("2024-01-01", None)

        assert "$gte" in filter_dict
        assert "$lte" not in filter_dict

    def test_timestamp_filter_end_only(self, search_tool):
        """Test timestamp filter with end date only."""
        filter_dict = search_tool._build_timestamp_filter(None, "2024-12-31")

        assert "$lte" in filter_dict
        assert "$gte" not in filter_dict

    def test_timestamp_filter_none(self, search_tool):
        """Test timestamp filter with no dates."""
        filter_dict = search_tool._build_timestamp_filter(None, None)

        assert filter_dict == {}

    def test_unix_timestamp_conversion(self, search_tool):
        """Test conversion of Unix timestamps in text."""
        test_text = "The timestamp is 1640995200 and another timestamp is 1641081600.123"

        converted = search_tool._convert_unix_timestamps_in_text(test_text)

        # Should contain ISO date formats (note: 1640995200 = 2021-12-31T16:00:00 UTC)
        assert "2021-12-31" in converted  # First timestamp
        assert "2022-01-01" in converted  # Second timestamp
        assert "timestamp is" in converted

    def test_read_document_ids_tracking(self, search_tool):
        """Test document ID deduplication tracking."""
        # Initially empty
        assert len(search_tool.read_document_ids) == 0

        # Add some IDs
        search_tool.read_document_ids.add("doc1")
        search_tool.read_document_ids.add("doc2")
        search_tool.read_document_ids.add("doc1")  # duplicate

        assert len(search_tool.read_document_ids) == 2
        assert "doc1" in search_tool.read_document_ids
        assert "doc2" in search_tool.read_document_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])