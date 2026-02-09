"""
Skill Loader

Provides lightweight database queries for the agent skill registry.
Used by:
  - AgentTriggerTool (list_registered_agents, read_agent_skill) for chat-mode discovery
  - Lean routing endpoint for event-driven routing
"""

import logging
from typing import Optional

from app.services.database import query_org

logger = logging.getLogger(__name__)


def list_agent_registry(org_slug: str) -> list[dict]:
    """
    Load lightweight agent registry (id, name, one-line summary only).
    This is cheap and can be called frequently.
    
    Returns list of dicts with keys: id, name, skill_summary, agent_type, status
    """
    try:
        rows = query_org(
            org_slug,
            """SELECT id, name, skill_summary, agent_type, status
               FROM ai_agents
               WHERE status = 'active' AND skill_md IS NOT NULL
               ORDER BY name"""
        )
        return rows
    except Exception as e:
        # Table might not exist yet (migration not run)
        error_str = str(e)
        if '42P01' in error_str or 'does not exist' in error_str.lower():
            logger.debug(f"ai_agents table not found for org {org_slug}, returning empty registry")
            return []
        logger.error(f"Error loading agent registry for org {org_slug}: {e}")
        return []


def get_agent_skill_md(org_slug: str, agent_id: str) -> Optional[str]:
    """
    Load the full SKILL.md for a specific agent. Called on-demand.
    
    Returns the SKILL.md content string or None if not found.
    """
    try:
        rows = query_org(
            org_slug,
            "SELECT skill_md FROM ai_agents WHERE id = %s AND status = 'active'",
            (agent_id,)
        )
        return rows[0]['skill_md'] if rows else None
    except Exception as e:
        error_str = str(e)
        if '42P01' in error_str or 'does not exist' in error_str.lower():
            return None
        logger.error(f"Error loading skill_md for agent {agent_id} in org {org_slug}: {e}")
        return None


def load_agent_skills_with_md(org_slug: str) -> list[dict]:
    """
    Load all active agents with their full SKILL.md content.
    Used by the event-driven routing endpoint (not chat mode).
    
    Returns list of dicts with keys: id, name, skill_md, skill_summary, agent_type, config
    """
    try:
        rows = query_org(
            org_slug,
            """SELECT id, name, skill_md, skill_summary, agent_type, config
               FROM ai_agents
               WHERE status = 'active' AND skill_md IS NOT NULL
               ORDER BY name"""
        )
        return rows
    except Exception as e:
        error_str = str(e)
        if '42P01' in error_str or 'does not exist' in error_str.lower():
            return []
        logger.error(f"Error loading agent skills for org {org_slug}: {e}")
        return []
