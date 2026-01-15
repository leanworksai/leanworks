"""
Integration tests for all domain-specific management tools.
Tests basic initialization and property structure.
"""
import pytest
from leanworks.agent.tools.task_management import TaskManagementTool
from leanworks.agent.tools.project_management import ProjectManagementTool
from leanworks.agent.tools.event_management import EventManagementTool
from leanworks.agent.tools.user_management import UserManagementTool
from leanworks.agent.tools.chat_management import ChatManagementTool
from leanworks.agent.tools.base_api_client import BaseAPIClient


def test_base_api_client():
    """Test BaseAPIClient initialization and methods."""
    client = BaseAPIClient(org_slug='test.ai', user_id='test@test.ai')
    assert client.org_slug == 'test.ai'
    assert client.user_id == 'test@test.ai'
    
    headers = client._get_headers()
    assert 'Content-Type' in headers
    assert headers['Content-Type'] == 'application/json'
    assert 'X-Org-Id' in headers
    assert headers['X-Org-Id'] == 'test.ai'


def test_all_tools_initialization():
    """Test that all domain tools can be initialized."""
    org_slug = 'test.ai'
    user_id = 'test@test.ai'
    
    tools = [
        TaskManagementTool(org_slug, user_id),
        ProjectManagementTool(org_slug, user_id),
        EventManagementTool(org_slug, user_id),
        UserManagementTool(org_slug, user_id),
        ChatManagementTool(org_slug, user_id),
    ]
    
    for tool in tools:
        assert tool.org_slug == org_slug
        assert tool.user_id == user_id
        assert hasattr(tool, '_get_headers')
        assert hasattr(tool, '_make_request')


def test_tool_properties_structure():
    """Test that all tools have properly structured properties."""
    task_tool = TaskManagementTool('test.ai')
    project_tool = ProjectManagementTool('test.ai')
    event_tool = EventManagementTool('test.ai')
    user_tool = UserManagementTool('test.ai')
    chat_tool = ChatManagementTool('test.ai')
    
    # Check query_tasks property
    prop = task_tool.query_tasks_property
    assert prop['type'] == 'custom'
    assert 'name' in prop
    assert 'description' in prop
    assert 'input_schema' in prop
    
    # Check query_projects property
    prop = project_tool.query_projects_property
    assert prop['type'] == 'custom'
    assert 'name' in prop
    
    # Check query_events property
    prop = event_tool.query_events_property
    assert prop['type'] == 'custom'
    assert 'name' in prop
    
    # Check query_users property
    prop = user_tool.query_users_property
    assert prop['type'] == 'custom'
    assert 'name' in prop
    
    # Check query_messages property
    prop = chat_tool.query_messages_property
    assert prop['type'] == 'custom'
    assert 'name' in prop
    assert 'required' in prop['input_schema']
    assert 'chatId' in prop['input_schema']['required']


def test_environment_variable_handling():
    """Test that tools correctly handle environment variables."""
    import os
    
    # Save original values
    original_url = os.getenv('LEANWORKS_HUB_URL')
    original_key = os.getenv('LEANWORKS_API_KEY')
    
    try:
        # Test with custom values
        os.environ['LEANWORKS_HUB_URL'] = 'http://custom:3001'
        os.environ['LEANWORKS_API_KEY'] = 'test_key_123'
        
        tool = TaskManagementTool('test.ai')
        assert tool.base_url == 'http://custom:3001'
        assert tool.api_key == 'test_key_123'
        
        headers = tool._get_headers()
        assert 'X-API-Key' in headers
        assert headers['X-API-Key'] == 'test_key_123'
        
    finally:
        # Restore original values
        if original_url:
            os.environ['LEANWORKS_HUB_URL'] = original_url
        else:
            os.environ.pop('LEANWORKS_HUB_URL', None)
        
        if original_key:
            os.environ['LEANWORKS_API_KEY'] = original_key
        else:
            os.environ.pop('LEANWORKS_API_KEY', None)


def test_toolkit_integration():
    """Test that tools can be integrated into ToolUse toolkit."""
    from leanworks.agent.tools.toolkit import ToolUse
    
    # Create ToolUse instance with new domain tools
    tool_use = ToolUse(
        org_slug='test.ai',
        firestore_client=None,  # Not needed for domain tools
        secret_manager_client=None,
        tools=['task_management', 'project_management', 'event_management'],
        user_id='test@test.ai'
    )
    
    assert 'task_management' in tool_use.requested_tools
    assert 'project_management' in tool_use.requested_tools
    assert 'event_management' in tool_use.requested_tools
    
    # Check lazy loading works
    assert hasattr(tool_use, 'task_management_tool')
    assert hasattr(tool_use, 'project_management_tool')
    assert hasattr(tool_use, 'event_management_tool')
