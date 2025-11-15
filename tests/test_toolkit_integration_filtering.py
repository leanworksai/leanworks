"""
Test suite for ToolUse integration filtering functionality.

This test verifies that:
1. Internal tools (firestore, search) are always enabled
2. External tools are filtered based on Firestore integrations collection
3. Tools not in the integration list are properly disabled
"""

import logging
from unittest.mock import Mock, patch, MagicMock
from leanworks.agent.tools.toolkit import ToolUse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_integration_filtering_with_outlook():
    """Test that outlook tool is enabled when it's in the integration list"""
    
    # Mock Firestore client wrapper
    firestore_wrapper = Mock()
    firestore_wrapper.domain = "test.leanworks.ai"
    
    # Mock Firestore client and integration documents
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    
    # Create mock integration documents
    outlook_doc = Mock()
    outlook_doc.to_dict.return_value = {'type': 'outlook', 'enabled': True}
    
    duckdb_doc = Mock()
    duckdb_doc.to_dict.return_value = {'type': 'duckdb', 'enabled': True}
    
    mock_collection.stream.return_value = [outlook_doc, duckdb_doc]
    
    # Patch _get_firestore_client to return our mock
    with patch('leanworks.setting._get_firestore_client', return_value=mock_db):
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            firestore_client_wrapper=firestore_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify that outlook is in requested_tools
        assert 'outlook' in tool_use.requested_tools, "Outlook should be enabled (in integration list)"
        
        # Verify internal tools are always included
        assert 'firestore' in tool_use.requested_tools, "Firestore should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        
        logger.info(f"✅ Test passed: outlook enabled with integrations: {tool_use.requested_tools}")


def test_integration_filtering_without_outlook():
    """Test that outlook tool is disabled when it's NOT in the integration list"""
    
    # Mock Firestore client wrapper
    firestore_wrapper = Mock()
    firestore_wrapper.domain = "test.leanworks.ai"
    
    # Mock Firestore client with no outlook integration
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    
    # Only duckdb integration, no outlook
    duckdb_doc = Mock()
    duckdb_doc.to_dict.return_value = {'type': 'duckdb', 'enabled': True}
    
    mock_collection.stream.return_value = [duckdb_doc]
    
    # Patch _get_firestore_client to return our mock
    with patch('leanworks.setting._get_firestore_client', return_value=mock_db):
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            firestore_client_wrapper=firestore_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify that outlook is NOT in requested_tools
        assert 'outlook' not in tool_use.requested_tools, "Outlook should be disabled (not in integration list)"
        
        # Verify internal tools are always included
        assert 'firestore' in tool_use.requested_tools, "Firestore should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        
        # Verify duckdb is included
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should be enabled (in integration list)"
        
        logger.info(f"✅ Test passed: outlook disabled without integration: {tool_use.requested_tools}")


def test_internal_tools_always_enabled():
    """Test that internal tools (firestore, search, duckdb) are always enabled even with empty integrations"""
    
    # Mock Firestore client wrapper
    firestore_wrapper = Mock()
    firestore_wrapper.domain = "test.leanworks.ai"
    
    # Mock Firestore client with NO integrations
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_collection.stream.return_value = []  # No integrations
    
    # Patch _get_firestore_client to return our mock
    with patch('leanworks.setting._get_firestore_client', return_value=mock_db):
        # Initialize ToolUse without any additional tools
        tool_use = ToolUse(
            firestore_client_wrapper=firestore_wrapper,
            storage_client=Mock(),
            secret_client=Mock()
        )
        
        # Verify internal tools are always included
        assert 'firestore' in tool_use.requested_tools, "Firestore should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should always be enabled (internal tool)"
        
        # Verify external tools are not included
        assert 'outlook' not in tool_use.requested_tools, "Outlook should not be enabled (no integration)"
        
        logger.info(f"✅ Test passed: internal tools always enabled: {tool_use.requested_tools}")


def test_integration_filtering_error_handling():
    """Test that integration filtering handles Firestore errors gracefully"""
    
    # Mock Firestore client wrapper
    firestore_wrapper = Mock()
    firestore_wrapper.domain = "test.leanworks.ai"
    
    # Patch _get_firestore_client to raise an exception
    with patch('leanworks.setting._get_firestore_client', side_effect=Exception("Firestore error")):
        # Initialize ToolUse with outlook requested
        tool_use = ToolUse(
            firestore_client_wrapper=firestore_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook']
        )
        
        # Verify internal tools are still enabled
        assert 'firestore' in tool_use.requested_tools, "Firestore should always be enabled (internal tool)"
        assert 'search' in tool_use.requested_tools, "Search should always be enabled (internal tool)"
        assert 'duckdb' in tool_use.requested_tools, "DuckDB should always be enabled (internal tool)"
        
        # Verify external tools are disabled (safe default)
        assert 'outlook' not in tool_use.requested_tools, "Outlook should be disabled on error (safe default)"
        
        logger.info(f"✅ Test passed: error handling works correctly: {tool_use.requested_tools}")


def test_integration_type_field_variations():
    """Test that integration filtering works with different field names (type vs integrationType)"""
    
    # Mock Firestore client wrapper
    firestore_wrapper = Mock()
    firestore_wrapper.domain = "test.leanworks.ai"
    
    # Mock Firestore client with different field names
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    
    # One doc uses 'type', another uses 'integrationType'
    outlook_doc = Mock()
    outlook_doc.to_dict.return_value = {'type': 'outlook', 'enabled': True}
    
    gitlab_doc = Mock()
    gitlab_doc.to_dict.return_value = {'integrationType': 'gitlab', 'enabled': True}
    
    mock_collection.stream.return_value = [outlook_doc, gitlab_doc]
    
    # Patch _get_firestore_client to return our mock
    with patch('leanworks.setting._get_firestore_client', return_value=mock_db):
        # Initialize ToolUse
        tool_use = ToolUse(
            firestore_client_wrapper=firestore_wrapper,
            storage_client=Mock(),
            secret_client=Mock(),
            tools=['outlook', 'gitlab']
        )
        
        # Verify both integrations are recognized
        assert 'outlook' in tool_use.requested_tools, "Outlook should be enabled (type field)"
        assert 'gitlab' in tool_use.requested_tools, "GitLab should be enabled (integrationType field)"
        
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

