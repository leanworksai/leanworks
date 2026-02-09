"""
Agent Trigger Tool

Provides three tools for Lean (the AI TPM) to discover and trigger registered AI agents:
  1. list_registered_agents - Lightweight registry scan (name + summary)
  2. read_agent_skill - Full SKILL.md on-demand read
  3. trigger_ai_agent - Invoke an agent on a platform entity

Design: SKILL.md content is NOT loaded into the system prompt. Lean discovers
agents on-demand via tools, keeping the context window small and scalable.
"""

import json
import logging
import os
import requests
from typing import Optional

from leanworks.agent.skills.skill_loader import list_agent_registry, get_agent_skill_md

logger = logging.getLogger(__name__)


class AgentTriggerTool:
    """Tools for Lean to discover and trigger registered AI agents."""

    def __init__(self, org_slug: str, hub_api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.org_slug = org_slug
        self.hub_api_base = hub_api_base or os.environ.get(
            'HUB_API_BASE',
            'http://localhost:3001' if os.environ.get('NODE_ENV') != 'production' else 'http://hub-api:80'
        )
        self.api_key = api_key or os.environ.get('HUB_INTERNAL_API_KEY', '')

    # ========================================================================
    # Tool 1: List Registered Agents (lightweight, returns name + summary)
    # ========================================================================

    @property
    def list_registered_agents_property(self):
        return {
            "name": "list_registered_agents",
            "description": (
                "List all registered AI agents with their name and one-line summary. "
                "Use to discover which agents are available before reading their full SKILL.md. "
                "Returns a compact list — does not include full skill definitions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    def list_registered_agents(self, **kwargs) -> str:
        """Return lightweight agent registry from the org database."""
        try:
            rows = list_agent_registry(self.org_slug)
            if not rows:
                return json.dumps({
                    "agents": [],
                    "message": "No AI agents with SKILL.md are registered in this organization."
                })
            agents = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "summary": r.get("skill_summary", "No summary available"),
                    "type": r.get("agent_type", "unknown"),
                }
                for r in rows
            ]
            return json.dumps({"agents": agents, "count": len(agents)})
        except Exception as e:
            logger.error(f"Error listing registered agents: {e}")
            return json.dumps({"error": str(e), "agents": []})

    # ========================================================================
    # Tool 2: Read Agent Skill (full SKILL.md, on-demand)
    # ========================================================================

    @property
    def read_agent_skill_property(self):
        return {
            "name": "read_agent_skill",
            "description": (
                "Read the full SKILL.md for a specific agent. Returns the complete skill "
                "definition including trigger conditions, requirements, and behavior. "
                "Use after listing agents to understand a candidate in detail before triggering."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent's ID from list_registered_agents"
                    }
                },
                "required": ["agent_id"]
            }
        }

    def read_agent_skill(self, agent_id: str, **kwargs) -> str:
        """Fetch the full SKILL.md from the org database."""
        try:
            skill_md = get_agent_skill_md(self.org_slug, agent_id)
            if skill_md:
                return skill_md
            return f"Agent '{agent_id}' not found or has no SKILL.md defined."
        except Exception as e:
            logger.error(f"Error reading agent skill for {agent_id}: {e}")
            return f"Error reading agent skill: {str(e)}"

    # ========================================================================
    # Tool 3: Trigger Agent
    # ========================================================================

    @property
    def trigger_ai_agent_property(self):
        return {
            "name": "trigger_ai_agent",
            "description": (
                "Trigger a registered AI agent to perform work on a platform entity "
                "(task, project, or plan). Only use after reading the agent's SKILL.md "
                "to confirm it is the right fit for the job."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent's ID"
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["task", "project", "plan"],
                        "description": "The type of platform entity to trigger the agent on"
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "The entity's ID (e.g., task ID)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Context or instructions to pass to the agent"
                    }
                },
                "required": ["agent_id", "entity_type", "entity_id"]
            }
        }

    def trigger_ai_agent(self, agent_id: str, entity_type: str, entity_id: str, prompt: str = None, **kwargs) -> str:
        """Call leanworks-hub API to trigger the agent on an entity."""
        try:
            # Map entity_type to the hub API endpoint path
            entity_type_plural = {
                "task": "tasks",
                "project": "projects",
                "plan": "plans"
            }.get(entity_type)

            if not entity_type_plural:
                return json.dumps({"error": f"Unsupported entity type: {entity_type}"})

            url = f"{self.hub_api_base}/api/{entity_type_plural}/{entity_id}/trigger-agent"
            payload = {"agentId": agent_id}
            if prompt:
                payload["prompt"] = prompt

            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.ok:
                result = response.json()
                trigger_id = result.get("triggerId", "unknown")
                return json.dumps({
                    "success": True,
                    "triggerId": trigger_id,
                    "message": f"Agent '{agent_id}' triggered on {entity_type} '{entity_id}'."
                })
            else:
                error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
                return json.dumps({
                    "success": False,
                    "error": f"Failed to trigger agent: {error_msg}"
                })
        except requests.exceptions.ConnectionError:
            return json.dumps({
                "success": False,
                "error": "Cannot connect to leanworks-hub API. The platform may be unavailable."
            })
        except Exception as e:
            logger.error(f"Error triggering agent {agent_id}: {e}")
            return json.dumps({"success": False, "error": str(e)})
