"""
Tests for Plans AI API: resource planning endpoint, validation, and response parsing.
"""
import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Disable API key verification for tests
os.environ["VERIFY_API_KEY"] = "false"

# Stub leanworks.agent.core.chat so importing app.api.plans_ai does not load ChatAgent
# (and thus the full leanworks stack). Unit tests only need _get_planning_tools and
# _parse_resource_plan_response; integration tests patch ChatAgent anyway.
if "leanworks.agent.core.chat" not in sys.modules:
    _mock_chat_module = MagicMock()
    sys.modules["leanworks.agent.core.chat"] = _mock_chat_module

from app.api.plans_ai import (
    _get_planning_tools,
    _parse_resource_plan_response,
)


# --- Unit tests: _get_planning_tools (all tools enabled) ---


def test_get_planning_tools_empty():
    assert _get_planning_tools([]) == []
    assert _get_planning_tools(None) == []


def test_get_planning_tools_returns_all_enabled_tools():
    assert _get_planning_tools(["project_management", "outlook", "jira"]) == [
        "project_management",
        "outlook",
        "jira",
    ]


def test_get_planning_tools_single_tool():
    assert _get_planning_tools(["project_management"]) == ["project_management"]


# --- Unit tests: _parse_resource_plan_response ---


def _valid_strategy(strategy_name="cost-optimized"):
    return {
        "strategy": strategy_name,
        "rationale": "Saves money",
        "total_cost": 100000,
        "estimated_duration_weeks": 12,
        "team_size": 3,
        "risk_level": "low",
        "resource_allocations": [
            {
                "user_email": "a@example.com",
                "user_name": "Alice",
                "role": "Engineer",
                "allocation_percentage": 50,
                "hourly_rate": 100,
                "estimated_hours": 200,
            }
        ],
        "expected_outcomes": ["Deliver on time"],
        "trade_offs": ["Less buffer"],
    }


def test_parse_resource_plan_response_valid_json_array():
    strategies = [_valid_strategy("cost-optimized"), _valid_strategy("time-optimized"), _valid_strategy("quality-optimized")]
    raw = json.dumps(strategies)
    result = _parse_resource_plan_response(raw)
    assert result is not None
    assert len(result) == 3
    assert result[0]["strategy"] == "cost-optimized"
    assert result[1]["strategy"] == "time-optimized"
    assert result[2]["strategy"] == "quality-optimized"


def test_parse_resource_plan_response_with_markdown_code_block():
    strategies = [_valid_strategy("cost-optimized"), _valid_strategy("time-optimized"), _valid_strategy("quality-optimized")]
    raw = "```json\n" + json.dumps(strategies) + "\n```"
    result = _parse_resource_plan_response(raw)
    assert result is not None
    assert len(result) == 3


def test_parse_resource_plan_response_empty_string():
    assert _parse_resource_plan_response("") is None
    assert _parse_resource_plan_response("   ") is None


def test_parse_resource_plan_response_invalid_json():
    assert _parse_resource_plan_response("not json at all") is None
    assert _parse_resource_plan_response("[invalid") is None


def test_parse_resource_plan_response_missing_required_keys():
    s = _valid_strategy("cost-optimized")
    del s["rationale"]
    raw = json.dumps([s])
    result = _parse_resource_plan_response(raw)
    assert result is None


def test_parse_resource_plan_response_invalid_strategy_name():
    s = _valid_strategy("cost-optimized")
    s["strategy"] = "custom-strategy"
    raw = json.dumps([s])
    result = _parse_resource_plan_response(raw)
    assert result is None


def test_parse_resource_plan_response_resource_allocations_not_list():
    s = _valid_strategy("cost-optimized")
    s["resource_allocations"] = "not-a-list"
    raw = json.dumps([s])
    result = _parse_resource_plan_response(raw)
    assert result is None


def test_parse_resource_plan_response_extra_text_before_array():
    strategies = [_valid_strategy("cost-optimized"), _valid_strategy("time-optimized"), _valid_strategy("quality-optimized")]
    raw = "Here are the strategies:\n" + json.dumps(strategies)
    result = _parse_resource_plan_response(raw)
    assert result is not None
    assert len(result) == 3


# --- Integration tests: generate_resource_plan endpoint ---


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_resource_plan_validation_missing_user_id():
    from quart import Quart
    from app.api.plans_ai import setup_plans_ai_endpoints

    app = Quart(__name__)
    setup_plans_ai_endpoints(app)
    client = app.test_client()

    res = await client.post(
        "/api/plans/generate-resource-plan",
        json={"org_slug": "test-org"},
    )
    assert res.status_code == 400
    data = await res.get_json()
    assert "user_id" in data.get("error", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_resource_plan_validation_missing_org_slug():
    from quart import Quart
    from app.api.plans_ai import setup_plans_ai_endpoints

    app = Quart(__name__)
    setup_plans_ai_endpoints(app)
    client = app.test_client()

    res = await client.post(
        "/api/plans/generate-resource-plan",
        json={"user_id": "user@example.com"},
    )
    assert res.status_code == 400
    data = await res.get_json()
    assert "org_slug" in data.get("error", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_resource_plan_validation_invalid_dates():
    from quart import Quart
    from app.api.plans_ai import setup_plans_ai_endpoints

    app = Quart(__name__)
    setup_plans_ai_endpoints(app)
    client = app.test_client()

    res = await client.post(
        "/api/plans/generate-resource-plan",
        json={
            "user_id": "user@example.com",
            "org_slug": "test-org",
            "start_date": "2026-03-31",
            "end_date": "2026-01-01",
        },
    )
    assert res.status_code == 400
    data = await res.get_json()
    assert "end_date" in data.get("error", "").lower() or "date" in data.get("error", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_resource_plan_validation_negative_budget():
    from quart import Quart
    from app.api.plans_ai import setup_plans_ai_endpoints

    app = Quart(__name__)
    setup_plans_ai_endpoints(app)
    client = app.test_client()

    res = await client.post(
        "/api/plans/generate-resource-plan",
        json={
            "user_id": "user@example.com",
            "org_slug": "test-org",
            "total_budget": -100,
        },
    )
    assert res.status_code == 400
    data = await res.get_json()
    assert "budget" in data.get("error", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_resource_plan_returns_strategies_when_mocked():
    from quart import Quart
    from app.api.plans_ai import setup_plans_ai_endpoints

    def _valid_strategy(strategy_name="cost-optimized"):
        return {
            "strategy": strategy_name,
            "rationale": "Test",
            "total_cost": 100000,
            "estimated_duration_weeks": 12,
            "team_size": 3,
            "risk_level": "low",
            "resource_allocations": [],
            "expected_outcomes": [],
            "trade_offs": [],
        }

    mock_response_content = json.dumps([
        _valid_strategy("cost-optimized"),
        _valid_strategy("time-optimized"),
        _valid_strategy("quality-optimized"),
    ])

    mock_firestore = MagicMock()
    mock_secret = MagicMock()
    mock_model = MagicMock()
    mock_tools = ["project_management", "outlook", "search"]

    with patch("app.api.plans_ai.initialize_clients_async", new_callable=AsyncMock) as mock_init:
        mock_init.return_value = (mock_firestore, mock_secret, mock_model, mock_tools)
        with patch("app.api.plans_ai.ChatAgent") as MockChatAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.process_message.return_value = {
                "content": mock_response_content,
                "data_sources": [],
            }
            MockChatAgent.return_value = mock_agent_instance

            app = Quart(__name__)
            setup_plans_ai_endpoints(app)
            client = app.test_client()

            res = await client.post(
                "/api/plans/generate-resource-plan",
                json={
                    "user_id": "user@example.com",
                    "org_slug": "test-org",
                    "plan_name": "Q1 Plan",
                    "total_budget": 250000,
                    "start_date": "2026-01-01",
                    "end_date": "2026-03-31",
                },
            )

    assert res.status_code == 200
    data = await res.get_json()
    assert "strategies" in data
    strategies = data["strategies"]
    assert len(strategies) == 3
    assert strategies[0]["strategy"] == "cost-optimized"
    assert strategies[1]["strategy"] == "time-optimized"
    assert strategies[2]["strategy"] == "quality-optimized"
    mock_init.assert_called_once_with("user@example.com", "test-org")
    mock_agent_instance.process_message.assert_called_once()
    call_args = mock_agent_instance.process_message.call_args
    assert "GATHER REAL DATA" in call_args[0][0] or "resource planning" in call_args[0][0].lower()
