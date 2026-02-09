"""
Lean Routing API Endpoint

Provides POST /api/lean-route for event-driven two-stage agent routing.
Called by leanworks-hub's event consumer worker when platform events occur.
"""

import logging
from quart import request, jsonify
from app import app
from app.auth.middleware import require_api_key
from leanworks.agent.skills.skill_loader import load_agent_skills_with_md
from leanworks.agent.skills.routing import (
    pre_filter_candidates,
    llm_route_decision,
    MAX_CANDIDATES,
)

logger = logging.getLogger(__name__)


def setup_lean_routing_endpoints():
    """Register Lean routing endpoints on the Quart app."""

    @app.route('/api/lean-route', methods=['POST'])
    @require_api_key
    async def lean_route():
        """
        Two-stage routing: given a platform event, decide which agents to trigger.
        
        Request body:
        {
            "event": { PlatformEvent object },
            "org_slug": "string",
            "max_candidates": 5  (optional)
        }
        
        Response:
        {
            "decisions": [
                {
                    "agent_id": "agent_xxx",
                    "trigger": true/false,
                    "reasoning": "...",
                    "context": "...",
                    "confidence": 0.9,
                    "latency_ms": 450
                }
            ]
        }
        """
        try:
            data = await request.get_json()
            if not data:
                return jsonify({"error": "Request body is required"}), 400

            event = data.get('event')
            org_slug = data.get('org_slug')
            max_candidates = data.get('max_candidates', MAX_CANDIDATES)

            if not event:
                return jsonify({"error": "event field is required"}), 400
            if not org_slug:
                return jsonify({"error": "org_slug field is required"}), 400

            # Stage 1: Load all agents with skills and pre-filter
            all_skills = load_agent_skills_with_md(org_slug)

            if not all_skills:
                return jsonify({
                    "decisions": [],
                    "reasoning": "No agents with SKILL.md found in this organization"
                })

            candidates = pre_filter_candidates(event, all_skills, max_candidates=max_candidates)

            if not candidates:
                return jsonify({
                    "decisions": [],
                    "reasoning": "No candidate agents matched the event after pre-filtering"
                })

            logger.info(
                f"[Lean Route] org={org_slug} event={event.get('type')} "
                f"total_skills={len(all_skills)} candidates={len(candidates)}"
            )

            # Stage 2: LLM decision
            decisions = await llm_route_decision(event, candidates)

            triggered_count = sum(1 for d in decisions if d.get('trigger'))
            logger.info(
                f"[Lean Route] Routing complete: {triggered_count}/{len(decisions)} agents triggered"
            )

            return jsonify({"decisions": decisions})

        except Exception as e:
            logger.error(f"[Lean Route] Error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
