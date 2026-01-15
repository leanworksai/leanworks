"""
Integration tests for TaskManagementTool.
Tests the domain-specific API-based task management tool.
"""
import pytest
from leanworks.agent.tools.task_management import TaskManagementTool


def test_task_management_tool_initialization():
    """Test TaskManagementTool initialization."""
    tool = TaskManagementTool(org_slug='test.ai', user_id='test@test.ai')
    assert tool.org_slug == 'test.ai'
    assert tool.user_id == 'test@test.ai'
    assert tool.base_url is not None


def test_query_tasks_property():
    """Test query_tasks tool property structure."""
    tool = TaskManagementTool(org_slug='test.ai')
    prop = tool.query_tasks_property
    
    assert prop['type'] == 'custom'
    assert prop['name'] == 'query_tasks'
    assert 'description' in prop
    assert 'input_schema' in prop
    assert 'properties' in prop['input_schema']
    
    # Check expected parameters
    props = prop['input_schema']['properties']
    assert 'status' in props
    assert 'priority' in props
    assert 'assignee' in props
    assert 'limit' in props


def test_create_task_property():
    """Test create_task tool property structure."""
    tool = TaskManagementTool(org_slug='test.ai', user_id='test@test.ai')
    prop = tool.create_task_property
    
    assert prop['type'] == 'custom'
    assert prop['name'] == 'create_task'
    assert 'description' in prop
    assert 'input_schema' in prop
    assert 'required' in prop['input_schema']
    assert 'title' in prop['input_schema']['required']


def test_update_task_property():
    """Test update_task tool property structure."""
    tool = TaskManagementTool(org_slug='test.ai', user_id='test@test.ai')
    prop = tool.update_task_property
    
    assert prop['type'] == 'custom'
    assert prop['name'] == 'update_task'
    assert 'description' in prop
    assert 'input_schema' in prop
    assert 'required' in prop['input_schema']
    assert 'taskId' in prop['input_schema']['required']


@pytest.mark.skip(reason="Requires running leanworks-hub server and valid API key")
def test_query_tasks_integration():
    """Integration test for query_tasks (requires hub server)."""
    import os
    if not os.getenv('LEANWORKS_API_KEY'):
        pytest.skip("LEANWORKS_API_KEY not set")
    
    tool = TaskManagementTool(org_slug='test.ai', user_id='test@test.ai')
    result = tool.query_tasks(status='completed', limit=10)
    
    assert isinstance(result, (list, dict))
    if isinstance(result, list):
        assert len(result) <= 10


@pytest.mark.skip(reason="Requires running leanworks-hub server and valid API key")
def test_create_task_integration():
    """Integration test for create_task (requires hub server)."""
    import os
    if not os.getenv('LEANWORKS_API_KEY'):
        pytest.skip("LEANWORKS_API_KEY not set")
    
    tool = TaskManagementTool(org_slug='test.ai', user_id='test@test.ai')
    result = tool.create_task(
        title='Test Task from Integration Test',
        status='todo',
        priority='low'
    )
    
    assert isinstance(result, dict)
    assert 'id' in result or 'error' in result
