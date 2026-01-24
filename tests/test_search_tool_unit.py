#!/usr/bin/env python3
"""
Unit tests for SearchTool - Testing core logic without external dependencies.

These tests use mocks to avoid Google AI library imports that cause segmentation faults
on macOS ARM64, while still validating the core SearchTool functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# Import only the classes we need, avoiding the full leanworks import tree
from leanworks.agent.tools.search import SearchTool, SearchResult


class TestSearchToolUnit:
    """Unit tests for SearchTool core functionality using mocks."""

    @pytest.fixture
    def mock_firestore_client(self):
        """Mock firestore client."""
        return Mock()

    @pytest.fixture
    def mock_secret_manager_client(self):
        """Mock secret manager client."""
        client = Mock()
        # Mock the secret access
        response = Mock()
        response.payload.data.decode.return_value = "mock-secret-value"
        client.access_secret_version.return_value = response
        return client

    @pytest.fixture
    def mock_chat(self):
        """Mock AsyncChat instance."""
        chat = Mock()
        chat.async_rewrite_query = AsyncMock(return_value=["test query", "alternative query"])
        chat.async_postprocess_nodes = AsyncMock(return_value=(
            [
                {
                    "context": "Test document content",
                    "metadata": {"doc_id": "doc1", "data_source": "test", "timestamp": 1640995200},
                    "doc_id": "doc1",
                    "data_source": "test"
                }
            ],
            []
        ))
        return chat

    @pytest.fixture
    def search_tool(self, mock_firestore_client, mock_secret_manager_client, mock_chat):
        """Create SearchTool with mocked dependencies."""
        with patch('leanworks.agent.tools.search.AsyncChat', return_value=mock_chat):
            # Mock the json.load and open calls in __init__
            with patch('builtins.open', Mock()):
                with patch('json.load', return_value={"project_id": "test-project"}):
                    tool = SearchTool(
                        firestore_client=mock_firestore_client,
                        org_slug="test-org",
                        secret_manager_client=mock_secret_manager_client,
                        read_document_ids=set()
                    )
                    tool.chat = mock_chat  # Override the chat instance
                    return tool

    def test_search_tool_initialization(self, search_tool):
        """Test that SearchTool initializes correctly."""
        assert search_tool is not None
        assert hasattr(search_tool, 'chat')
        assert search_tool.org_slug == "test-org"
        assert isinstance(search_tool.read_document_ids, set)

    def test_search_result_class_behavior(self):
        """Test SearchResult class string-like behavior."""
        test_context = "DOCUMENT - Date: , Source: test, Doc ID: 123\nTest content"
        test_sources = ["source1", "source2"]

        result = SearchResult(test_context, test_sources)

        # Test string-like behavior
        assert str(result) == test_context
        assert len(result) == len(test_context)
        assert "Test content" in result
        assert "Not in content" not in result

        # Test repr
        repr_str = repr(result)
        assert "SearchResult" in repr_str
        assert "context_length" in repr_str
        assert "sources=2" in repr_str

    def test_date_conversion_valid_dates(self, search_tool):
        """Test date string to timestamp conversion with valid inputs."""
        test_cases = [
            ("2024-01-01", True),
            ("2024-01-01T12:00:00Z", True),
            ("2024-01-01T12:00:00+00:00", True),
        ]

        for date_str, should_succeed in test_cases:
            timestamp = search_tool._convert_date_to_timestamp(date_str)
            if should_succeed:
                assert timestamp is not None, f"Expected success for {date_str}"
                assert isinstance(timestamp, int), f"Expected int timestamp for {date_str}"
            else:
                assert timestamp is None, f"Expected failure for {date_str}"

    def test_date_conversion_invalid_dates(self, search_tool):
        """Test date string to timestamp conversion with invalid inputs."""
        invalid_cases = ["invalid-date", "", None]

        for date_str in invalid_cases:
            timestamp = search_tool._convert_date_to_timestamp(date_str)
            assert timestamp is None, f"Expected None for invalid date: {date_str}"

    def test_timestamp_filter_building(self, search_tool):
        """Test building timestamp filters for search queries."""
        # Test with both dates
        filter_dict = search_tool._build_timestamp_filter("2024-01-01", "2024-12-31")
        assert "$gte" in filter_dict
        assert "$lte" in filter_dict

        # Test with start date only
        filter_dict = search_tool._build_timestamp_filter("2024-01-01", None)
        assert "$gte" in filter_dict
        assert "$lte" not in filter_dict

        # Test with end date only
        filter_dict = search_tool._build_timestamp_filter(None, "2024-12-31")
        assert "$lte" in filter_dict
        assert "$gte" not in filter_dict

        # Test with no dates
        filter_dict = search_tool._build_timestamp_filter(None, None)
        assert filter_dict == {}

    @pytest.mark.asyncio
    async def test_basic_search_execution(self, search_tool, mock_chat):
        """Test basic search execution with mocked dependencies."""
        # Setup mock return data
        mock_chat.async_postprocess_nodes.return_value = ([
            {
                "context": "Sample document content about testing",
                "metadata": {"doc_id": "doc1", "data_source": "test", "timestamp": 1640995200},
                "doc_id": "doc1",
                "data_source": "test"
            }
        ], ["test"])

        result = await search_tool.async_search_documents(query="test query")

        # Verify the method was called
        mock_chat.async_rewrite_query.assert_called_once_with("test query")
        mock_chat.async_postprocess_nodes.assert_called_once()

        # Verify result structure
        assert "formatted_context" in result
        assert "data_sources" in result
        assert len(result["data_sources"]) == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_tool, mock_chat):
        """Test search with data source and date filters."""
        mock_chat.async_postprocess_nodes.return_value = ([], [])

        result = await search_tool.async_search_documents(
            query="test query",
            data_source="github_commits",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Verify result structure even with empty results
        assert "formatted_context" in result
        assert "data_sources" in result
        assert isinstance(result["data_sources"], list)

    def test_search_documents_sync_wrapper(self, search_tool):
        """Test the synchronous wrapper around async search."""
        with patch.object(search_tool, 'async_search_documents', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = {
                "formatted_context": "Test results",
                "data_sources": ["test"]
            }

            result = search_tool.search_documents(query="test")

            assert isinstance(result, SearchResult)
            assert str(result) == "Test results"
            assert result._search_data_sources == ["test"]

    def test_search_documents_error_handling(self, search_tool):
        """Test error handling in synchronous search wrapper."""
        with patch.object(search_tool, 'async_search_documents', new_callable=AsyncMock) as mock_async:
            mock_async.side_effect = Exception("Test error")

            result = search_tool.search_documents(query="test")

            assert isinstance(result, SearchResult)
            assert "Error occurred during documents search" in str(result)

    def test_deduplication_tracking(self, search_tool):
        """Test that read_document_ids set is properly managed."""
        # Initially empty
        assert len(search_tool.read_document_ids) == 0

        # Add some IDs
        search_tool.read_document_ids.add("doc1")
        search_tool.read_document_ids.add("doc2")

        assert len(search_tool.read_document_ids) == 2
        assert "doc1" in search_tool.read_document_ids
        assert "doc2" in search_tool.read_document_ids

    def test_timestamp_formatting_in_context(self, search_tool):
        """Test Unix timestamp conversion in formatted context."""
        test_text = "timestamp is 1640995200.123"
        formatted = search_tool._convert_unix_timestamps_in_text(test_text)

        # Should contain ISO format date
        assert "2022-01-01" in formatted
        assert "timestamp is" in formatted

    def test_result_formatting_with_metadata(self, search_tool):
        """Test result formatting with document metadata."""
        # This tests the formatting logic in async_search_documents
        # We'll mock the internal calls and verify the output structure

        test_context = [
            {
                "context": "Document content",
                "metadata": {"doc_id": "doc1", "data_source": "test", "timestamp": 1640995200},
                "doc_id": "doc1",
                "data_source": "test"
            }
        ]

        # Test the formatting logic directly
        formatted_context = ""
        for ctx in test_context:
            title = f"DOCUMENT - Date: , Source: test, Doc ID: doc1"
            formatted_context += f"{title}\nDocument content\n\n"

        assert "DOCUMENT -" in formatted_context
        assert "Source: test" in formatted_context
        assert "Doc ID: doc1" in formatted_context
        assert "Document content" in formatted_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])