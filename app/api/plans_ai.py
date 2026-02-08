"""
Plans AI API Endpoints
Handles AI-powered resource planning and insights generation for plans
"""

import json
import asyncio
import logging
import time
import traceback
from quart import request, Response
from leanworks.agent.core.chat import ChatAgent
from anthropic import Anthropic
from app import get_firestore_client, get_secret_manager_client
from app.auth.middleware import require_api_key
from app.services.client import initialize_clients_async, get_client_info
from app.services.database import query_org_one

logger = logging.getLogger(__name__)


async def stream_ai_response(user_id, org_slug, session_id, query, firestore_client, 
                             secret_manager_client, model_client):
    """
    Async generator for SSE streaming of AI responses.
    Yields formatted SSE events.
    """
    try:
        # Initialize ChatAgent
        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id=session_id,
            clear_conversation=True,
            tools=None  # No external tools for AI planning
        )
        
        # Stream events from agent
        async for event in agent.process_message_stream(query, None, None):
            yield f"data: {json.dumps(event)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream_ai_response: {str(e)}", exc_info=True)
        error_event = {"type": "error", "error": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"


def setup_plans_ai_endpoints(app):
    """
    Register Plans AI endpoints with the app.
    """

    @app.route('/api/plans/generate-resource-plan', methods=['POST'])
    @require_api_key
    async def generate_resource_plan():
        """
        Generate AI-powered resource allocation plans.
        
        Request body:
        {
            "user_id": "user@example.com",
            "org_slug": "org-slug",
            "session_id": "session-id",
            "plan_name": "Q1 Product Launch",
            "plan_objectives": [...],
            "total_budget": 250000,
            "budget_categories": [...],
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
            "team_members": [...]
        }
        """
        request_start_time = time.time()
        logger.info("Generate resource plan endpoint accessed")
        
        try:
            data = await request.get_json()
            user_id = data.get("user_id")
            org_slug = data.get("org_slug")
            session_id = data.get("session_id")
            stream_enabled = data.get("stream", False)
            
            # Validate required fields
            if not user_id:
                return {"error": "user_id is required"}, 400
            
            if not org_slug:
                return {"error": "org_slug is required"}, 400
            
            logger.info(f"Generating resource plan for user_id: {user_id}, org_slug: {org_slug}")
            
            # Initialize clients
            try:
                firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
                logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
            except Exception as e:
                logger.error(f"Error initializing clients: {str(e)}")
                traceback.print_exc()
                return {"error": f"Failed to initialize clients: {str(e)}"}, 500
            
            # Build prompt for resource planning
            plan_data = {
                "name": data.get("plan_name", "Plan"),
                "objectives": data.get("plan_objectives", []),
                "total_budget": data.get("total_budget", 0),
                "budget_categories": data.get("budget_categories", []),
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
                "team_members": data.get("team_members", []),
            }
            
            prompt = f"""You are an expert project planner and resource allocation specialist.

Analyze the following plan and generate 3 alternative resource allocation strategies:

Plan Details:
- Name: {plan_data['name']}
- Start Date: {plan_data['start_date']}
- End Date: {plan_data['end_date']}
- Total Budget: ${plan_data['total_budget']:,.2f}
- Objectives: {json.dumps(plan_data['objectives'], indent=2)}
- Budget Categories: {json.dumps(plan_data['budget_categories'], indent=2)}
- Available Team Members: {json.dumps(plan_data['team_members'], indent=2)}

Generate 3 optimization strategies:
1. Cost-Optimized: Minimize total cost while meeting objectives
2. Time-Optimized: Minimize timeline while staying within budget
3. Quality-Optimized: Maximize quality with best resources

For each strategy, return a JSON object with:
{{
  "strategy": "cost-optimized|time-optimized|quality-optimized",
  "rationale": "Why this strategy works for this plan",
  "total_cost": number,
  "estimated_duration_weeks": number,
  "team_size": number,
  "risk_level": "low|medium|high",
  "resource_allocations": [
    {{
      "user_email": "user@example.com",
      "user_name": "Name",
      "role": "role",
      "allocation_percentage": number (0-100),
      "hourly_rate": number,
      "estimated_hours": number
    }}
  ],
  "expected_outcomes": ["outcome1", "outcome2"],
  "trade_offs": ["trade-off1", "trade-off2"]
}}

Return a JSON array with exactly 3 strategy objects. Return ONLY valid JSON."""

            if stream_enabled:
                logger.info("Streaming mode enabled for resource plan generation")
                return Response(
                    stream_ai_response(user_id, org_slug, session_id or f"resource-plan-{int(time.time())}", 
                                     prompt, firestore_client, secret_manager_client, model_client),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    }
                )
            
            # Non-streaming mode
            agent = ChatAgent(
                firestore_client=firestore_client,
                secret_manager_client=secret_manager_client,
                model_client=model_client,
                user_id=user_id,
                org_slug=org_slug,
                session_id=session_id or f"resource-plan-{int(time.time())}",
                clear_conversation=True,
                tools=None
            )
            
            processing_start_time = time.time()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, agent.process_message, prompt, None)
            processing_time = time.time() - processing_start_time
            logger.info(f"Resource plan generation completed in {processing_time:.3f}s")
            
            response_content = response.get("content", "")
            
            # Parse JSON from response
            strategies = []
            try:
                # Try to extract JSON array from response
                import re
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_content, re.DOTALL)
                if json_match:
                    strategies = json.loads(json_match.group(0))
                else:
                    # Try to parse entire response
                    strategies = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON from response: {str(e)}")
                # Return the raw response
                return {
                    "strategies": [{
                        "strategy": "error",
                        "raw_response": response_content[:1000]
                    }]
                }, 200
            
            total_time = time.time() - request_start_time
            logger.info(f"Total resource plan generation time: {total_time:.3f}s")
            logger.info(f"Successfully generated {len(strategies)} resource plan strategies")
            
            return {"strategies": strategies}, 200
            
        except Exception as e:
            error_msg = f"Error generating resource plan: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}, 500

    @app.route('/api/plans/generate-insights', methods=['POST'])
    @require_api_key
    async def generate_insights():
        """
        Generate AI insights for a plan (risks, recommendations, predictions).
        
        Request body:
        {
            "user_id": "user@example.com",
            "org_slug": "org-slug",
            "session_id": "session-id",
            "plan_id": "plan-id",
            "plan_name": "Q1 Product Launch",
            "plan_status": "active",
            "health_score": 68,
            "total_budget": 250000,
            "spent_to_date": 160000,
            "objectives": [...],
            "resource_allocations": [...],
            "milestones": [...],
            "team_size": 12
        }
        """
        request_start_time = time.time()
        logger.info("Generate insights endpoint accessed")
        
        try:
            data = await request.get_json()
            user_id = data.get("user_id")
            org_slug = data.get("org_slug")
            session_id = data.get("session_id")
            stream_enabled = data.get("stream", False)
            
            # Validate required fields
            if not user_id:
                return {"error": "user_id is required"}, 400
            
            if not org_slug:
                return {"error": "org_slug is required"}, 400
            
            logger.info(f"Generating insights for user_id: {user_id}, org_slug: {org_slug}")
            
            # Initialize clients
            try:
                firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
                logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
            except Exception as e:
                logger.error(f"Error initializing clients: {str(e)}")
                traceback.print_exc()
                return {"error": f"Failed to initialize clients: {str(e)}"}, 500
            
            # Build prompt for insights generation
            plan_data = {
                "name": data.get("plan_name", "Plan"),
                "status": data.get("plan_status", "active"),
                "health_score": data.get("health_score", 50),
                "total_budget": data.get("total_budget", 0),
                "spent_to_date": data.get("spent_to_date", 0),
                "budget_percentage": (data.get("spent_to_date", 0) / max(data.get("total_budget", 1), 1)) * 100,
                "objectives": data.get("objectives", []),
                "resource_allocations": data.get("resource_allocations", []),
                "milestones": data.get("milestones", []),
                "team_size": data.get("team_size", 0),
            }
            
            spent_pct = plan_data['budget_percentage']
            burn_rate = "HIGH" if spent_pct > 80 else "MEDIUM" if spent_pct > 60 else "LOW"
            
            prompt = f"""You are an expert project analyst and strategic planner.

Analyze the following plan and provide comprehensive insights:

Plan Status:
- Name: {plan_data['name']}
- Status: {plan_data['status']}
- Health Score: {plan_data['health_score']}/100
- Budget: ${plan_data['total_budget']:,.2f} ({spent_pct:.1f}% spent: ${plan_data['spent_to_date']:,.2f})
- Burn Rate: {burn_rate}
- Team Size: {plan_data['team_size']}

Objectives Status:
{json.dumps(plan_data['objectives'], indent=2)}

Resource Allocations:
{json.dumps(plan_data['resource_allocations'], indent=2)}

Milestones:
{json.dumps(plan_data['milestones'], indent=2)}

Generate a comprehensive insights JSON object with:
{{
  "summary": "Executive summary of plan health (2-3 sentences)",
  "risks": [
    {{
      "title": "Risk title",
      "severity": "low|medium|high",
      "description": "Detailed description",
      "impact": "What happens if this occurs"
    }}
  ],
  "recommendations": [
    {{
      "title": "Recommendation title",
      "priority": "low|medium|high",
      "description": "Detailed action item",
      "expected_impact": "Expected outcome"
    }}
  ],
  "predictions": {{
    "budget_trend": "on_track|at_risk|over_budget",
    "timeline_trend": "on_track|at_risk|delayed",
    "estimated_completion_date": "YYYY-MM-DD",
    "confidence_level": "low|medium|high"
  }},
  "quick_insight": "One-liner summary for list view"
}}

Return ONLY valid JSON."""

            if stream_enabled:
                logger.info("Streaming mode enabled for insights generation")
                return Response(
                    stream_ai_response(user_id, org_slug, session_id or f"insights-{int(time.time())}", 
                                     prompt, firestore_client, secret_manager_client, model_client),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    }
                )
            
            # Non-streaming mode
            agent = ChatAgent(
                firestore_client=firestore_client,
                secret_manager_client=secret_manager_client,
                model_client=model_client,
                user_id=user_id,
                org_slug=org_slug,
                session_id=session_id or f"insights-{int(time.time())}",
                clear_conversation=True,
                tools=None
            )
            
            processing_start_time = time.time()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, agent.process_message, prompt, None)
            processing_time = time.time() - processing_start_time
            logger.info(f"Insights generation completed in {processing_time:.3f}s")
            
            response_content = response.get("content", "")
            
            # Parse JSON from response
            insights = {}
            try:
                # Try to extract JSON object from response
                import re
                json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group(0))
                else:
                    # Try to parse entire response
                    insights = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON from response: {str(e)}")
                # Return basic insights structure
                insights = {
                    "summary": response_content[:200],
                    "risks": [],
                    "recommendations": [],
                    "predictions": {"budget_trend": "on_track"},
                    "quick_insight": response_content[:100]
                }
            
            total_time = time.time() - request_start_time
            logger.info(f"Total insights generation time: {total_time:.3f}s")
            logger.info(f"Successfully generated insights")
            
            return insights, 200
            
        except Exception as e:
            error_msg = f"Error generating insights: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}, 500
