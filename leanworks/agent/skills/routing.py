"""
Lean Routing Engine

Two-stage event routing for the Lean orchestrator:
  Stage 1: Cheap keyword-based pre-filtering to select top candidate agents
  Stage 2: LLM-based decision using Anthropic Claude to decide which agents to trigger

This is used by the /api/lean-route endpoint for event-driven routing,
NOT for chat-mode (which uses the on-demand tool-based approach).
"""

import json
import logging
import os
import re
import time
from typing import Any

import anthropic

from leanworks.agent.skills.skill_loader import load_agent_skills_with_md

logger = logging.getLogger(__name__)

# Default model for routing decisions (fast + smart enough)
ROUTING_MODEL = os.environ.get('LEAN_ROUTING_MODEL', 'claude-sonnet-4-20250514')
MAX_CANDIDATES = int(os.environ.get('LEAN_MAX_CANDIDATES', '5'))

ROUTING_SYSTEM_PROMPT = """You are Lean, an AI Technical Program Manager for a software organization.

Your job is to analyze a platform event and decide which AI agents should be triggered 
based on their SKILL.md descriptions.

For each candidate agent provided, decide:
1. Should it be triggered? (yes/no)
2. Why or why not? (1-2 sentences)
3. What context/prompt should be passed to the agent?
4. Confidence level (0.0 to 1.0)

Rules:
- Only trigger agents whose SKILL.md clearly matches the event context.
- Pay attention to "Don't trigger me when" sections — respect exclusions.
- If the event actor is the agent itself, do NOT trigger that agent (avoid loops).
- When in doubt, skip. False positives waste agent resources.

Respond ONLY with a JSON array (no markdown fences, no extra text):
[{"agent_id": "...", "trigger": true, "reasoning": "...", "context": "...", "confidence": 0.9}]
"""


def extract_event_keywords(event: dict) -> set[str]:
    """Extract searchable keywords from a platform event."""
    keywords = set()

    # Event type parts (e.g. "task.status_changed" -> {"task", "status", "changed"})
    event_type = event.get('type', '')
    for part in re.split(r'[._]', event_type):
        if part:
            keywords.add(part.lower())

    # Entity type
    entity_type = event.get('entityType', '')
    if entity_type:
        keywords.add(entity_type.lower())

    # Payload keywords
    payload = event.get('payload', {})
    if isinstance(payload, dict):
        for key in ['status', 'title', 'name', 'project_name', 'projectName']:
            val = payload.get(key, '')
            if isinstance(val, str) and val:
                for word in val.lower().split():
                    if len(word) > 2:
                        keywords.add(word)

        # Comment or description text (extract meaningful words)
        for key in ['comment', 'description', 'content', 'message']:
            val = payload.get(key, '')
            if isinstance(val, str) and val:
                for word in val.lower().split():
                    cleaned = re.sub(r'[^a-z0-9]', '', word)
                    if len(cleaned) > 3:
                        keywords.add(cleaned)

    return keywords


def keyword_overlap_score(event_keywords: set[str], skill_summary: str | None) -> float:
    """Score how well an agent's skill_summary matches event keywords."""
    if not skill_summary:
        return 0.0

    summary_words = set()
    for word in skill_summary.lower().split():
        cleaned = re.sub(r'[^a-z0-9]', '', word)
        if len(cleaned) > 2:
            summary_words.add(cleaned)

    if not summary_words:
        return 0.0

    overlap = event_keywords & summary_words
    # Normalize by the smaller set to avoid penalizing short summaries
    return len(overlap) / min(len(event_keywords), len(summary_words)) if event_keywords else 0.0


def pre_filter_candidates(event: dict, skills: list[dict], max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """
    Stage 1: Cheap keyword-based filtering.
    Returns the top N candidate agents sorted by relevance score.
    """
    event_keywords = extract_event_keywords(event)
    if not event_keywords:
        # If no keywords could be extracted, return top agents by name
        return skills[:max_candidates]

    scored = []
    for skill in skills:
        score = keyword_overlap_score(event_keywords, skill.get('skill_summary'))
        scored.append((skill, score))

    # Sort by score descending, then by name for stability
    scored.sort(key=lambda x: (-x[1], x[0].get('name', '')))

    # Return candidates with any positive score, capped at max
    candidates = [s for s, score in scored[:max_candidates] if score > 0]

    # If no positive-scoring candidates, include top agents anyway (LLM can decide)
    if not candidates and skills:
        candidates = [s for s, _ in scored[:max_candidates]]

    return candidates


def format_routing_prompt(event: dict, candidates: list[dict]) -> str:
    """Build the user message for the LLM routing decision."""
    parts = ["## Platform Event\n"]
    parts.append(f"- **Type**: {event.get('type', 'unknown')}")
    parts.append(f"- **Entity Type**: {event.get('entityType', 'unknown')}")
    parts.append(f"- **Entity ID**: {event.get('entityId', 'unknown')}")
    parts.append(f"- **Actor**: {event.get('actorId', 'unknown')} ({event.get('actorType', 'unknown')})")

    payload = event.get('payload', {})
    if payload:
        # Include relevant payload fields (truncated)
        payload_str = json.dumps(payload, default=str)
        if len(payload_str) > 2000:
            payload_str = payload_str[:2000] + '... (truncated)'
        parts.append(f"- **Payload**: {payload_str}")

    mentions = event.get('mentions', [])
    if mentions:
        parts.append(f"- **Mentions**: {', '.join(mentions)}")

    parts.append(f"\n## Candidate Agents ({len(candidates)})\n")

    for i, candidate in enumerate(candidates, 1):
        parts.append(f"### Agent {i}: {candidate.get('name', 'Unknown')} (ID: {candidate.get('id', '?')})")
        parts.append(f"Type: {candidate.get('agent_type', 'unknown')}")
        skill_md = candidate.get('skill_md', '')
        if skill_md:
            # Include full SKILL.md (already filtered to top candidates)
            if len(skill_md) > 3000:
                skill_md = skill_md[:3000] + '\n... (truncated)'
            parts.append(f"\n```markdown\n{skill_md}\n```\n")
        else:
            summary = candidate.get('skill_summary', 'No skill description available')
            parts.append(f"Summary: {summary}\n")

    parts.append("\n## Your Decision\n")
    parts.append("For each agent above, decide whether to trigger it based on the event context and their SKILL.md.")

    return '\n'.join(parts)


async def llm_route_decision(event: dict, candidates: list[dict], api_key: str | None = None) -> list[dict]:
    """
    Stage 2: LLM-based routing decision.
    Calls Anthropic Claude to decide which agents to trigger.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'))

    user_message = format_routing_prompt(event, candidates)

    start_time = time.time()
    try:
        response = client.messages.create(
            model=ROUTING_MODEL,
            system=ROUTING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2048,
            temperature=0.0,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract text content
        text = ''
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text

        # Parse JSON response
        decisions = parse_routing_response(text, candidates)

        # Attach latency to each decision
        for d in decisions:
            d['latency_ms'] = latency_ms

        return decisions

    except Exception as e:
        logger.error(f"LLM routing decision failed: {e}")
        latency_ms = int((time.time() - start_time) * 1000)
        # Return all candidates as skipped on error
        return [
            {
                "agent_id": c.get("id", ""),
                "trigger": False,
                "reasoning": f"LLM routing error: {str(e)}",
                "context": "",
                "confidence": 0.0,
                "latency_ms": latency_ms,
            }
            for c in candidates
        ]


def parse_routing_response(text: str, candidates: list[dict]) -> list[dict]:
    """Parse the LLM JSON response into structured routing decisions."""
    # Try to extract JSON from the response (handle markdown fences)
    text = text.strip()
    if text.startswith('```'):
        # Remove markdown code fences
        lines = text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        text = '\n'.join(lines).strip()

    try:
        decisions = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                decisions = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse routing response (chars=%d)", len(text))
                return []
        else:
            logger.error("No JSON array found in routing response (chars=%d)", len(text))
            return []

    if not isinstance(decisions, list):
        decisions = [decisions]

    # Validate and normalize each decision
    valid_agent_ids = {c.get('id') for c in candidates}
    normalized = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        agent_id = d.get('agent_id', '')
        if agent_id not in valid_agent_ids:
            continue
        normalized.append({
            "agent_id": agent_id,
            "trigger": bool(d.get('trigger', False)),
            "reasoning": str(d.get('reasoning', ''))[:1000],
            "context": str(d.get('context', ''))[:2000],
            "confidence": min(1.0, max(0.0, float(d.get('confidence', 0.5)))),
        })

    return normalized
