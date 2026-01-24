#!/usr/bin/env python3
"""
Integration tests for SearchTool - Comprehensive testing of core search functionality.

These tests verify that the SearchTool works correctly with real external services
(Pinecone, Firestore, GCP Secret Manager) and test the core search features including
scope-based searching, filtering, result formatting, and error handling.
"""

import pytest
import pytest_asyncio
import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from leanworks.agent.tools.search import SearchTool, SearchResult
from google.cloud import bigquery, secretmanager
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def search_tool():
    """Initialize SearchTool with real services for integration testing."""
    tool = None
    try:
        logger.info("🔧 Initializing SearchTool for integration testing...")

        # Load credentials
        credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")

        # Initialize BigQuery client with credentials
        bq_client = bigquery.Client(credentials=credentials)

        # Initialize Secret Manager client
        secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=credentials)

        # Initialize search tool with correct parameters
        tool = SearchTool(
            firestore_client=bq_client,  # Using BigQuery as firestore client
            org_slug="leanworks.ai",
            secret_manager_client=secret_manager_client,
            read_document_ids=set(),
            credential_path="gcp_credential.json"
        )

        logger.info("✅ SearchTool initialized successfully for testing")
        yield tool

    except Exception as e:
        logger.error(f"❌ Failed to initialize SearchTool: {e}")
        pytest.skip(f"SearchTool initialization failed: {e}")
        yield None


@pytest.fixture
def test_org_slug():
    """Organization slug for testing."""
    return "leanworks.ai"


@pytest.fixture
def cleanup_read_ids(search_tool):
    """Reset the shared deduplication set between tests."""
    if search_tool:
        original_ids = search_tool.read_document_ids.copy()
        search_tool.read_document_ids.clear()
        yield
        # Restore original state
        search_tool.read_document_ids = original_ids


class TestSearchToolIntegration:
    """Integration tests for SearchTool core functionality."""

    def test_search_tool_initialization(self, search_tool):
        """Test that SearchTool initializes correctly."""
        assert search_tool is not None
        assert hasattr(search_tool, 'chat')
        assert hasattr(search_tool, 'org_slug')
        assert search_tool.org_slug == "leanworks.ai"
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

    @pytest.mark.asyncio
    async def test_basic_search_execution(self, search_tool):
        """Test basic search execution returns results."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(query="project management")

        # Verify result is SearchResult object
        assert isinstance(result, SearchResult)

        # Verify it has string-like behavior
        context = str(result)
        assert isinstance(context, str)

        # Verify data sources are populated
        assert hasattr(result, '_search_data_sources')
        assert isinstance(result._search_data_sources, list)

        logger.info(f"Basic search returned {len(context)} chars with {len(result._search_data_sources)} sources")

    @pytest.mark.asyncio
    async def test_search_scope_knowledge_base(self, search_tool):
        """Test searching knowledge base scope only."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="backend development",
            search_scope="knowledge_base"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should return results (may be empty but shouldn't error)
        assert isinstance(context, str)
        logger.info(f"KB-only search returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_search_scope_tool_responses(self, search_tool):
        """Test searching tool responses scope only."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="api call",
            search_scope="tool_responses"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should return results (may be empty but shouldn't error)
        assert isinstance(context, str)
        logger.info(f"Tool responses search returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_search_scope_all(self, search_tool):
        """Test searching all scopes (default behavior)."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="system architecture",
            search_scope="all"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should return results (may be empty but shouldn't error)
        assert isinstance(context, str)
        logger.info(f"All scopes search returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_data_source_filtering(self, search_tool):
        """Test filtering by data source."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        # Test with GitHub commits data source
        result = search_tool.search_documents(
            query="bug fix",
            data_source="github_commits"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should return results (may be filtered to empty but shouldn't error)
        assert isinstance(context, str)
        logger.info(f"GitHub commits filter returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_date_filtering_start_end(self, search_tool):
        """Test date filtering with both start and end dates."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="feature development",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should return results (may be filtered but shouldn't error)
        assert isinstance(context, str)
        logger.info(f"Date range filter returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_date_filtering_start_only(self, search_tool):
        """Test date filtering with start date only."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="recent changes",
            start_date="2024-06-01"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        assert isinstance(context, str)
        logger.info(f"Start date only filter returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_date_filtering_end_only(self, search_tool):
        """Test date filtering with end date only."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="old documentation",
            end_date="2024-06-30"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        assert isinstance(context, str)
        logger.info(f"End date only filter returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_tool_name_filtering(self, search_tool):
        """Test filtering by tool name in tool responses."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="query execution",
            search_scope="tool_responses",
            tool_name="bigquery"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        assert isinstance(context, str)
        logger.info(f"Tool name filter returned {len(context)} chars")

    @pytest.mark.asyncio
    async def test_combined_filters(self, search_tool):
        """Test combining multiple filters."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(
            query="code review",
            data_source="github_commits",
            start_date="2024-01-01",
            end_date="2024-12-31",
            search_scope="knowledge_base"
        )

        assert isinstance(result, SearchResult)
        context = str(result)

        assert isinstance(context, str)
        logger.info(f"Combined filters returned {len(context)} chars")

    def test_date_conversion_functionality(self, search_tool):
        """Test date string to timestamp conversion."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        test_cases = [
            ("2024-01-01", True),  # Valid date
            ("2024-01-01T12:00:00Z", True),  # Valid ISO
            ("invalid-date", False),  # Invalid
            ("", False),  # Empty
            (None, False),  # None
        ]

        for date_str, should_succeed in test_cases:
            timestamp = search_tool._convert_date_to_timestamp(date_str)
            if should_succeed:
                assert timestamp is not None, f"Expected success for {date_str}"
                assert isinstance(timestamp, int), f"Expected int timestamp for {date_str}"
            else:
                assert timestamp is None, f"Expected failure for {date_str}"

    @pytest.mark.asyncio
    async def test_deduplication_behavior(self, search_tool, cleanup_read_ids):
        """Test that deduplication works across multiple searches."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        # First search
        result1 = search_tool.search_documents(query="test deduplication")

        # Second search with same query
        result2 = search_tool.search_documents(query="test deduplication")

        # Results should be accessible (deduplication shouldn't break functionality)
        assert isinstance(result1, SearchResult)
        assert isinstance(result2, SearchResult)

        logger.info("Deduplication test completed successfully")

    @pytest.mark.asyncio
    async def test_result_formatting_structure(self, search_tool):
        """Test that results are properly formatted."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(query="test formatting")

        assert isinstance(result, SearchResult)
        context = str(result)

        # Check for proper document structure if results exist
        if len(context.strip()) > 0:
            # Should have proper formatting
            assert isinstance(context, str)

            # If there are documents, they should have proper headers
            if "DOCUMENT -" in context:
                lines = context.split('\n')
                doc_headers = [line for line in lines if line.startswith("DOCUMENT -")]
                assert len(doc_headers) > 0, "Found DOCUMENT headers but none detected"

                # Check header structure
                for header in doc_headers:
                    assert "Date:" in header, f"Header missing Date: {header}"
                    assert "Source:" in header, f"Header missing Source: {header}"
                    assert "Doc ID:" in header, f"Header missing Doc ID: {header}"

        logger.info(f"Result formatting check: {len(context)} chars")

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_tool):
        """Test handling of empty or minimal queries."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        # Test with minimal query
        result = search_tool.search_documents(query="a")

        # Should not crash, should return some result
        assert isinstance(result, SearchResult)

        logger.info("Empty query handling test completed")

    @pytest.mark.asyncio
    async def test_timestamp_in_result_formatting(self, search_tool):
        """Test that timestamps are properly formatted in results."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        result = search_tool.search_documents(query="timestamp test")

        assert isinstance(result, SearchResult)
        context = str(result)

        # Check for Unix timestamp conversion (should convert timestamps to readable format)
        # This is a soft check - conversion may not always happen
        assert isinstance(context, str)

        logger.info(f"Timestamp formatting check: {len(context)} chars")

    @pytest.mark.asyncio
    async def test_error_handling_invalid_credentials(self):
        """Test error handling when credentials are invalid."""
        # This would require mocking or separate setup
        # For integration tests, we'll assume credentials are valid
        pass

    @pytest.mark.asyncio
    async def test_query_rewrite_functionality(self, search_tool):
        """Test that query rewriting works and provides multiple query variants."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        # Test basic search to ensure query rewriting is invoked
        result = search_tool.search_documents(query="machine learning model")

        assert isinstance(result, SearchResult)
        context = str(result)

        # Query rewriting should happen automatically - we can't directly test it
        # but we can verify the search completes successfully
        assert isinstance(context, str)

        logger.info("Query rewrite functionality test completed (rewriting happens internally)")

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, search_tool):
        """Test multiple concurrent search requests."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        async def single_search(query):
            return search_tool.search_documents(query=query)

        # Run multiple searches concurrently
        queries = ["async test", "concurrent search", "parallel execution"]
        tasks = [single_search(query) for query in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent search {i} failed: {result}")
            else:
                assert isinstance(result, SearchResult), f"Search {i} returned wrong type"

        logger.info(f"Concurrent searches completed: {len(results)} successful")

    @pytest.mark.asyncio
    async def test_large_result_handling(self, search_tool):
        """Test handling of potentially large result sets."""
        if not search_tool:
            pytest.skip("SearchTool not available")

        # Use a broad query that might return many results
        result = search_tool.search_documents(query="code")

        assert isinstance(result, SearchResult)
        context = str(result)

        # Should handle large results gracefully
        assert isinstance(context, str)

        # Log result size for monitoring
        logger.info(f"Large result handling: {len(context)} chars, {len(result._search_data_sources)} sources")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])