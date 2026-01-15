"""
Integration tests for ChatManagementTool.
Tests the domain-specific API-based chat/message management tool.
"""
import pytest
from leanworks.agent.tools.chat_management import ChatManagementTool


def test_chat_management_tool_initialization():
    """Test ChatManagementTool initialization."""
    tool = ChatManagementTool(org_slug='test.ai', user_id='test@test.ai')
    assert tool.org_slug == 'test.ai'
    assert tool.user_id == 'test@test.ai'
    assert tool.base_url is not None


def test_query_messages_property():
    """Test query_messages tool property structure."""
    tool = ChatManagementTool(org_slug='test.ai')
    prop = tool.query_messages_property
    
    assert prop['type'] == 'custom'
    assert prop['name'] == 'query_messages'
    assert 'description' in prop
    assert 'input_schema' in prop
    assert 'required' in prop['input_schema']
    assert 'chatId' in prop['input_schema']['required']
    
    # Check expected parameters
    props = prop['input_schema']['properties']
    assert 'chatId' in props
    assert 'role' in props
    assert 'limit' in props
    assert 'orderBy' in props


@pytest.mark.skip(reason="Requires running leanworks-hub server and valid API key")
def test_query_messages_integration():
    """Integration test for query_messages (requires hub server)."""
    import os
    if not os.getenv('LEANWORKS_API_KEY'):
        pytest.skip("LEANWORKS_API_KEY not set")
    
    tool = ChatManagementTool(org_slug='test.ai', user_id='test@test.ai')
    result = tool.query_messages(
        chatId='ai-assistant-test@test.ai',
        limit=20
    )
    
    assert isinstance(result, (list, dict))
    if isinstance(result, list):
        assert len(result) <= 20
