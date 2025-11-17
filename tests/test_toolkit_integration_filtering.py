"""
Test suite for ToolUse integration filtering functionality.

This test verifies that:
1. Internal tools (postgres, search, duckdb) are always enabled
2. External tools are filtered based on PostgreSQL integrations table
3. Tools not in the integration list are properly disabled
"""

import logging
from unittest.mock import Mock, patch, MagicMock
from leanworks.agent.tools.toolkit import ToolUse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_integration_filtering_with_outlook():
    """Test that outlook tool is enabled when it's in the integration list"""
    
    # Mock PostgreSQL client wrapper
    postgres_wrapper = Mock()
    postgres_wrapper.domain = "test.leanworks.ai"
    
    # Mock PostgresTool.query_postgres method (patch where it's imported inside _get_available_integrations)
    with patch('leanworks.agent.tools.postgres.PostgresTool') as mock_postgres_class:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.query_postgres.return_value = [
            {'integration_name': 'outlook', 'connected': True},
            {'integration_name': 'duckdb', 'connected': True}
        ]
        mock_postgres_class.return_value = mock_postgres_instance
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            postgres_client_wrapper=postgres_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify that outlook is in requested_tools
        assert 'outlook' in tool_use.requested_tools, "Outlook should be enabled (in integration list)"
        
        # Verify internal tools are always included
        assert 'postgres' in tool_use.requested_tools, "PostgreSQL should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        
        logger.info(f"✅ Test passed: outlook enabled with integrations: {tool_use.requested_tools}")


def test_integration_filtering_without_outlook():
    """Test that outlook tool is disabled when it's NOT in the integration list"""
    
    # Mock PostgreSQL client wrapper
    postgres_wrapper = Mock()
    postgres_wrapper.domain = "test.leanworks.ai"
    
    # Mock PostgresTool.query_postgres method (patch where it's imported inside _get_available_integrations)
    with patch('leanworks.agent.tools.postgres.PostgresTool') as mock_postgres_class:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.query_postgres.return_value = [
            {'integration_name': 'duckdb', 'connected': True}
        ]
        mock_postgres_class.return_value = mock_postgres_instance
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            postgres_client_wrapper=postgres_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify that outlook is NOT in requested_tools
        assert 'outlook' not in tool_use.requested_tools, "Outlook should be disabled (not in integration list)"
        
        # Verify internal tools are always included
        assert 'postgres' in tool_use.requested_tools, "PostgreSQL should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        
        # Verify duckdb is included
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should be enabled (in integration list)"
        
        logger.info(f"✅ Test passed: outlook disabled without integration: {tool_use.requested_tools}")


def test_internal_tools_always_enabled():
    """Test that internal tools (postgres, search, duckdb) are always enabled even with empty integrations"""
    
    # Mock PostgreSQL client wrapper
    postgres_wrapper = Mock()
    postgres_wrapper.domain = "test.leanworks.ai"
    
    # Mock PostgresTool.query_postgres method (patch where it's imported inside _get_available_integrations)
    with patch('leanworks.agent.tools.postgres.PostgresTool') as mock_postgres_class:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.query_postgres.return_value = []  # No integrations
        mock_postgres_class.return_value = mock_postgres_instance
        # Initialize ToolUse without any additional tools
        tool_use = ToolUse(
            postgres_client_wrapper=postgres_wrapper,
            storage_client=Mock(),
            secret_client=Mock()
        )
        
        # Verify internal tools are always included
        assert 'postgres' in tool_use.requested_tools, "PostgreSQL should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should always be enabled (internal tool)"
        
        # Verify external tools are not included
        assert 'outlook' not in tool_use.requested_tools, "Outlook should not be enabled (no integration)"
        
        logger.info(f"✅ Test passed: internal tools always enabled: {tool_use.requested_tools}")


def test_integration_filtering_error_handling():
    """Test that integration filtering handles PostgreSQL errors gracefully"""
    
    # Mock PostgreSQL client wrapper
    postgres_wrapper = Mock()
    postgres_wrapper.domain = "test.leanworks.ai"
    
    # Mock PostgresTool.query_postgres method to raise an exception
    with patch('leanworks.agent.tools.toolkit.PostgresTool') as mock_postgres_class:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.query_postgres.side_effect = Exception("PostgreSQL error")
        mock_postgres_class.return_value = mock_postgres_instance
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            postgres_client_wrapper=postgres_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify internal tools are still enabled
        assert 'postgres' in tool_use.requested_tools, "PostgreSQL should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should always be enabled (internal tool)"
        
        # Verify external tools are disabled (safe default)
        assert 'outlook' not in tool_use.requested_tools, "Outlook should be disabled on error (safe default)"
        
        logger.info(f"✅ Test passed: error handling works correctly: {tool_use.requested_tools}")


def test_integration_type_field_variations():
    """Test that integration filtering works with integration_name and integration_id fields"""
    
    # Mock PostgreSQL client wrapper
    postgres_wrapper = Mock()
    postgres_wrapper.domain = "test.leanworks.ai"
    
    # Mock PostgresTool.query_postgres method (patch where it's imported inside _get_available_integrations)
    with patch('leanworks.agent.tools.postgres.PostgresTool') as mock_postgres_class:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.query_postgres.return_value = [
            {'integration_name': 'outlook', 'connected': True},
            {'integration_id': 'gitlab', 'connected': True}  # No integration_name, fallback to integration_id
        ]
        mock_postgres_class.return_value = mock_postgres_instance
        # Initialize ToolUse
        tool_use = ToolUse(
            postgres_client_wrapper=postgres_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook', 'gitlab']
        )
        
        # Verify both integrations are recognized
        assert 'outlook' in tool_use.requested_tools, "Outlook should be enabled (integration_name field)"
        assert 'gitlab' in tool_use.requested_tools, "GitLab should be enabled (integration_id fallback)"
        
        logger.info(f"✅ Test passed: field variations handled: {tool_use.requested_tools}")


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING INTEGRATION FILTERING")
    print("=" * 80)
    print()
    
    try:
        test_integration_filtering_with_outlook()
        print()
        
        test_integration_filtering_without_outlook()
        print()
        
        test_internal_tools_always_enabled()
        print()
        
        test_integration_filtering_error_handling()
        print()
        
        test_integration_type_field_variations()
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print("=" * 80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 80)
        raise
    except Exception as e:
        print("=" * 80)
        print(f"❌ TEST ERROR: {str(e)}")
        print("=" * 80)
        raise

