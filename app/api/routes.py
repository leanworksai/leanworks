"""
API routes for the Leanworks API
"""
import json
import asyncio
import datetime
import logging
import os
import re
import time
import traceback
import uuid
import base64
import requests
from quart import request, Response
from leanworks.agent.chat import ChatAgent
from anthropic import Anthropic
from app import app, get_firestore_client, get_secret_manager_client
from app.auth.middleware import require_api_key
from app.services.client import (
    get_client_info, 
    initialize_clients_async, 
    get_cached_storage_client
)
from app.services.database import query_org_one, get_domain_from_email
from app.utils.cache import clear_cache
from leanworks.setting import MAX_IMAGES_PER_REQUEST, MAX_IMAGE_SIZE_MB, VISION_SUPPORTED_IMAGE_TYPES

logger = logging.getLogger(__name__)


def validate_vision_images(images: list) -> tuple:
    """
    Validate images for vision API.
    
    Args:
        images: List of image objects with type (base64 or url), and appropriate data
        
    Returns:
        Tuple of (is_valid: bool, error_message: str, processed_images: list)
    """
    if not images:
        return True, "", []
    
    if not isinstance(images, list):
        return False, "images must be a list", []
    
    if len(images) > MAX_IMAGES_PER_REQUEST:
        return False, f"Maximum {MAX_IMAGES_PER_REQUEST} images per request", []
    
    processed_images = []
    
    for idx, img in enumerate(images):
        if not isinstance(img, dict):
            return False, f"Image {idx} must be an object", []
        
        img_type = img.get("type")
        
        if img_type == "base64":
            # Validate base64 image
            media_type = img.get("media_type")
            data = img.get("data")
            
            if not media_type:
                return False, f"Image {idx}: media_type is required for base64 images", []
            
            if media_type not in VISION_SUPPORTED_IMAGE_TYPES:
                return False, f"Image {idx}: Unsupported image type '{media_type}'. Supported types: {', '.join(VISION_SUPPORTED_IMAGE_TYPES)}", []
            
            if not data:
                return False, f"Image {idx}: data is required for base64 images", []
            
            if not isinstance(data, str):
                return False, f"Image {idx}: base64 data must be a string", []
            
            # Validate base64 format and size
            try:
                decoded = base64.b64decode(data, validate=True)
                size_mb = len(decoded) / (1024 * 1024)
                if size_mb > MAX_IMAGE_SIZE_MB:
                    return False, f"Image {idx}: Size {size_mb:.2f}MB exceeds maximum allowed size of {MAX_IMAGE_SIZE_MB}MB", []
            except Exception as e:
                return False, f"Image {idx}: Invalid base64 data - {str(e)}", []
            
            processed_images.append({
                "type": "base64",
                "media_type": media_type,
                "data": data
            })
        
        elif img_type == "url":
            # Validate URL image
            url = img.get("url")
            
            if not url:
                return False, f"Image {idx}: url is required for URL images", []
            
            if not isinstance(url, str):
                return False, f"Image {idx}: url must be a string", []
            
            # Basic URL validation
            if not (url.startswith("http://") or url.startswith("https://")):
                return False, f"Image {idx}: URL must start with http:// or https://", []
            
            processed_images.append({
                "type": "url",
                "url": url
            })
        
        else:
            return False, f"Image {idx}: type must be 'base64' or 'url', got '{img_type}'", []
    
    return True, "", processed_images


async def stream_ask_response(user_id, org_slug, session_id, query, cited_context, file_references,
                              tools, firestore_client, secret_manager_client, 
                              model_client, available_tools):
    """
    Async generator for SSE streaming of ask API responses.
    Yields formatted SSE events.
    
    Args:
        user_id: User identifier
        org_slug: Organization slug
        session_id: Session ID for conversation
        query: User query
        cited_context: Cited context for the query
        file_references: List of file references for vision API (base64 or URL images)
        tools: Tools to use (already filtered)
        firestore_client: Firestore client
        secret_manager_client: Secret manager client
        model_client: Model client (Anthropic)
        available_tools: Available tools from configuration
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
            clear_conversation=False,
            tools=tools
        )
        
        # Stream events from agent
        async for event in agent.process_message_stream(query, cited_context, file_references if file_references else None):
            # Format as SSE
            yield f"data: {json.dumps(event)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream_ask_response: {str(e)}", exc_info=True)
        error_event = {"type": "error", "error": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"


def get_user_details_from_postgres(user_email: str):
    """
    Get user details from PostgreSQL shared database.
    
    Args:
        user_email: User email address
        
    Returns:
        dict: User details with first_name, last_name, avatar, timezone, or None if not found
    """
    try:
        # Query users table in the shared database
        from app.services.database import query_shared_one
        
        user_data = query_shared_one(
            "SELECT first_name, last_name, job_title, timezone FROM users WHERE email = %s",
            (user_email.lower(),)
        )
        
        if user_data:
            # Generate avatar from initials if not stored
            first_name = user_data.get("first_name", "")
            last_name = user_data.get("last_name", "")
            avatar = ""
            if first_name and last_name:
                avatar = (first_name[0] + last_name[0]).upper()
            
            return {
                "first_name": first_name,
                "last_name": last_name,
                "avatar": avatar,
                "timezone": user_data.get("timezone", "UTC") or "UTC"
            }
    except Exception as e:
        logger.warning(f"Could not fetch user details from PostgreSQL for {user_email}: {str(e)}")
    
    return None


def get_project_details_from_postgres(user_email: str, project_id: str):
    """
    Get project details from PostgreSQL.
    
    Args:
        user_email: User email address (to determine org database)
        project_id: Project ID
        
    Returns:
        dict: Project details with name, or None if not found
    """
    try:
        # Derive org_slug from user email domain
        org_slug = get_domain_from_email(user_email)
        # Query projects table in the org's database
        project_data = query_org_one(
            org_slug,
            "SELECT name FROM projects WHERE id = %s",
            (project_id,)
        )
        
        if project_data:
            return {
                "name": project_data.get("name", "")
            }
    except Exception as e:
        logger.warning(f"Could not fetch project details from PostgreSQL for {project_id}: {str(e)}")
    
    return None


async def async_log_interaction(storage_client, payload, response, client_domain: str):
    """
    Asynchronously log user interactions with the API to Google Cloud Storage.
    Logs are partitioned by date with all logs from the same date appended to the same file.
    
    Args:
        storage_client: Initialized CloudStorage client
        payload (dict): The complete request payload
        response (dict): The RAG response object
        client_domain (str): The client domain for organizing logs
    """
    try:
        timestamp = datetime.datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Create log entry
        log_entry = {
            "payload": payload,
            "response": response,
            "timestamp": timestamp.isoformat()
        }
        
        log_filename = f"domains/{client_domain}/{date_str}.json"
        
        # Try to load existing logs for today
        existing_logs = []
        try:
            loop = asyncio.get_event_loop()
            log_data = await loop.run_in_executor(None, storage_client.download_blob_to_memory, log_filename)
            existing_logs = json.loads(log_data)
            # If the existing logs are not a list, initialize with an empty list
            if not isinstance(existing_logs, list):
                existing_logs = []
        except Exception as e:
            # If file doesn't exist or there's an error loading it, start with empty list
            logger.warning(f"Could not load existing logs: {str(e)}")
            existing_logs = []
        
        # Append new log entry
        existing_logs.append(log_entry)
        
        # Upload updated logs to GCS
        try:
            await loop.run_in_executor(
                None,
                storage_client.upload_blob_from_memory,
                json.dumps(existing_logs, indent=2),
                log_filename
            )
            logger.info(f"Successfully logged interaction to {log_filename}")
        except Exception as e:
            logger.error(f"Failed to upload logs to GCS: {str(e)}")
    except Exception as e:
        logger.error(f"Error in async_log_interaction: {str(e)}")


@app.route("/", methods=["GET"])
async def root():
    logger.info("Root endpoint accessed")
    return "Hello from /!", 200


@app.route("/api/verify", methods=["GET"])
@require_api_key
async def verify():
    """Endpoint to verify API key is working"""
    logger.info("API key verification successful")
    return {"status": "success", "message": "API key is valid"}, 200


@app.route('/api/ask', methods=['POST'])
@require_api_key
async def ask():
    request_start_time = time.time()
    logger.info("Ask endpoint accessed")
    
    try:
        # Handle JSON request
        data = await request.get_json()
        user_id = data.get("user_id")
        org_slug = data.get("org_slug")
        session_id = data.get("session_id")
        query = data.get("query")
        cited_context = data.get("cited_context")
        tools = data.get("tools")
        stream_enabled = data.get("stream", False)
        images = data.get("images", [])
        
        # Validate required fields
        if not user_id:
            return {"error": "user_id is required"}, 400
        
        if not org_slug:
            return {"error": "org_slug is required"}, 400
        
        if not query:
            return {"error": "query is required"}, 400
        
        # Validate and process images
        if images:
            is_valid, error_msg, vision_images = validate_vision_images(images)
            if not is_valid:
                logger.warning(f"Image validation failed: {error_msg}")
                return {"error": error_msg, "code": "INVALID_IMAGES"}, 400
            logger.info(f"Validated {len(vision_images)} images for vision request")
        else:
            vision_images = []
        
        # Process tools parameter
        if tools:
            if isinstance(tools, str):
                tools = tools.split(",")
            elif isinstance(tools, list):
                tools = tools
            else:
                tools = None
        else:
            tools = None
        
        # Log API payload
        payload = {
            "user_id": user_id,
            "org_slug": org_slug,
            "session_id": session_id,
            "query": query,
            "cited_context": cited_context,
            "tools": tools
        }
        if vision_images:
            payload["images"] = [{"type": img.get("type"), "media_type": img.get("media_type")} if img.get("type") == "base64" else {"type": img.get("type"), "url": img.get("url")} for img in vision_images]
        logger.info(f"Ask API payload: {json.dumps(payload, default=str)}")
            
        logger.info(f"Request from user_id: {user_id}, org_slug: {org_slug}, session_id: {session_id}")
        if vision_images:
            logger.info(f"Processing {len(vision_images)} images for vision analysis")
        
        # Performance optimization: Initialize clients asynchronously with caching
        try:
            if not user_id:
                logger.error("user_id is None or empty in ask endpoint")
                return {"error": "user_id is required"}, 400
                
            logger.info(f"Initializing clients for user: {user_id} in org slug: {org_slug}")
            firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
            
            logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
        except Exception as e:
            logger.error(f"Error initializing clients for user {user_id} in org slug {org_slug}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to initialize clients: {str(e)}"}, 500

        # Filter tools based on integrations table in PostgreSQL if tools are provided
        if tools is not None:
            try:
                logger.info(f"Available tools from PostgreSQL integrations: {available_tools}")
                
                # Filter tools to only include those enabled in PostgreSQL integrations table
                filtered_tools = []
                for tool in tools:
                    if tool in available_tools:
                        filtered_tools.append(tool)
                        logger.info(f"Tool '{tool}' is enabled in PostgreSQL integrations")
                    else:
                        logger.warning(f"Tool '{tool}' is not found in PostgreSQL integrations table, skipping")
                
                logger.info(f"Filtered tools: {filtered_tools}")
            except Exception as e:
                logger.error(f"Error filtering tools against PostgreSQL integrations: {str(e)}")
                # If there's an error filtering, use original tools
                filtered_tools = tools
        else:
            filtered_tools = None
        
        # Log tools being used
        logger.info(f"Ask API - Tools being used: {json.dumps(filtered_tools if filtered_tools else available_tools, default=str)}")
        print(f"Filtered tools: {filtered_tools}")
        
        # Web app now sends HTML positions directly - no conversion needed
        
        # Check if streaming is requested
        if stream_enabled:
            logger.info(f"Streaming mode enabled for request from user_id: {user_id}")
            # Return SSE streaming response
            return Response(
                stream_ask_response(user_id, org_slug, session_id, query, cited_context, vision_images,
                                   filtered_tools, firestore_client, 
                                   secret_manager_client, model_client, available_tools),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        # Performance optimization: Initialize Agent with pre-initialized clients (using keyword arguments like test file)
        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id=session_id,
            clear_conversation=False,
            tools=filtered_tools
        )
        
        # Generate response
        logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Performance optimization: Process message with timing
        processing_start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            agent.process_message, 
            query, 
            cited_context,
            vision_images if vision_images else None
        )
        processing_time = time.time() - processing_start_time
        logger.info(f"Message processing completed in {processing_time:.3f}s")
        
        # Performance optimization: Log the interaction in the background without blocking
        # Initialize CloudStorage client for logging separately
        try:
            client_name, _ = get_client_info(org_slug)
            if client_name:
                loop = asyncio.get_event_loop()
                storage_client = await loop.run_in_executor(None, get_cached_storage_client, client_name)
                # Prepare payload for logging
                payload = {
                    "user_id": user_id,
                    "org_slug": org_slug,
                    "session_id": session_id,
                    "query": query,
                    "cited_context": cited_context,
                    "tools": tools
                }
                asyncio.create_task(async_log_interaction(
                    storage_client=storage_client,
                    payload=payload,
                    response=response,
                    client_domain=client_name
                ))
        except Exception as e:
            logger.warning(f"Failed to initialize storage client for logging: {str(e)}")
        
        total_time = time.time() - request_start_time
        logger.info(f"Total request processing time: {total_time:.3f}s")
        
        # Log final response with image info
        response_log = {
            "content": response.get("content", "")[:500] + "..." if len(response.get("content", "")) > 500 else response.get("content", "")
        }
        if vision_images:
            response_log["images_count"] = len(vision_images)
            response_log["images_types"] = [img.get("type") for img in vision_images]
        logger.info(f"Ask API final response: {json.dumps(response_log, default=str)}")
        
        logger.info("Successfully generated and returned response")
        return response
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}, 500


@app.route('/api/generate-task', methods=['POST'])
@require_api_key
async def generate_task():
    """
    Generate task details based on task name using the ask API.
    Users can provide additional task details that will override or supplement AI-generated values.
    Returns a task object formatted according to the tasks table schema.
    """
    request_start_time = time.time()
    logger.info("Generate task endpoint accessed")
    
    try:
        data = await request.get_json()
        task_name = data.get("task_name")
        user_id = data.get("user_id")
        org_slug = data.get("org_slug")
        session_id = data.get("session_id")
        
        # User-provided task details (optional)
        user_description = data.get("description")
        user_status = data.get("status")
        user_priority = data.get("priority")
        user_due_date = data.get("due_date")
        user_tags = data.get("tags")
        user_reason = data.get("reason")
        
        # Project and assignee details (optional)
        project_id = data.get("project_id")
        project_name = data.get("project_name")
        assignee_id = data.get("assignee_id")
        assignee_name = data.get("assignee_name")
        assignee_avatar = data.get("assignee_avatar")
        created_by = data.get("created_by")
        
        if not task_name:
            return {"error": "task_name is required"}, 400
        
        if not user_id:
            return {"error": "user_id is required"}, 400
        
        if not org_slug:
            return {"error": "org_slug is required"}, 400
        
        logger.info(f"Generating task details for task_name: {task_name}, user_id: {user_id}, org_slug: {org_slug}")
        
        # Initialize clients
        try:
            firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
            logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
        except Exception as e:
            logger.error(f"Error initializing clients for user {user_id} in org slug {org_slug}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to initialize clients: {str(e)}"}, 500
        
        # Build prompt with user-provided context
        user_context_parts = []
        if user_description:
            user_context_parts.append(f"Description: {user_description}")
        if user_status:
            user_context_parts.append(f"Status: {user_status}")
        if user_priority:
            user_context_parts.append(f"Priority: {user_priority}")
        if user_due_date:
            user_context_parts.append(f"Due date: {user_due_date}")
        if user_tags:
            user_context_parts.append(f"Tags: {', '.join(user_tags) if isinstance(user_tags, list) else user_tags}")
        if user_reason:
            user_context_parts.append(f"Reason/Context: {user_reason}")
        if project_name:
            user_context_parts.append(f"Project name: {project_name}")
        if assignee_id:
            user_context_parts.append(f"Assignee ID: {assignee_id}")
        
        user_context = "\n".join(user_context_parts) if user_context_parts else None
        
        # Get client name for database queries
        client_name, _ = get_client_info(org_slug)
        
        # Build the prompt that instructs the agent to gather context before generating
        user_provided_section = ""
        if user_context:
            user_provided_section = f"""
## User-Provided Details
{user_context}
"""

        project_context_hint = ""
        if project_id:
            project_context_hint = f"Project ID: {project_id}"
        if project_name:
            project_context_hint += f"\nProject Name: {project_name}"

        prompt = f"""You are a senior software engineer and project manager helping to create a well-defined task.

## Task to Generate
**Task Name**: "{task_name}"
{user_provided_section}
{f"## Project Reference" + chr(10) + project_context_hint if project_context_hint else ""}

## Instructions

Use your tools to gather relevant context before generating each field. Return a JSON object with these fields:

- **title**: The task title
- **description**: Markdown description including User Story, Acceptance Criteria, Definition of Done, and Dependencies. Reference related tasks and progress you find.
- **status**: One of 'todo', 'in-progress', 'review', 'completed', 'blocked'. Consider what's currently blocked or in progress.
- **priority**: One of 'low', 'medium', 'high', 'urgent'. Consider current blockers and in-progress work.
- **due_date**: YYYY-MM-DD format. Consider project timeline and related task due dates.
- **tags**: Array of tags. Be consistent with tags used in existing project tasks.
- **reason**: Why this task is needed. Consider current project progress and gaps.
- **assignee_id**: Email of team member. Consider their role, expertise, and current workload.
- **project_id**: Use "{project_id}" if provided, otherwise find the most relevant project.
- **project_name**: Derive from project context.

Return ONLY valid JSON."""

        # Initialize ChatAgent with tools enabled for context gathering
        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id=session_id or f"generate-task-{int(time.time())}",
            clear_conversation=True,  # Use fresh conversation for task generation
            tools=available_tools  # Enable tools so agent can gather context
        )
        
        # Generate response using ChatAgent
        processing_start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.process_message, prompt, None)
        processing_time = time.time() - processing_start_time
        logger.info(f"Task generation completed in {processing_time:.3f}s")
        
        # Extract JSON from response
        response_content = response.get("content", "")
        
        # Try to parse JSON from the response
        task_data = {}
        try:
            # Try to extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON object directly (handle nested braces)
                # Find the first { and then match balanced braces
                start_idx = response_content.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(response_content)):
                        if response_content[i] == '{':
                            brace_count += 1
                        elif response_content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if brace_count == 0:
                        json_str = response_content[start_idx:end_idx]
                        task_data = json.loads(json_str)
                    else:
                        # If braces don't balance, try parsing the whole response
                        task_data = json.loads(response_content)
                else:
                    # If no JSON found, try parsing the whole response
                    task_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from response: {str(e)}")
            logger.warning(f"Response content: {response_content[:500]}")
            # Fallback: create basic task structure from response
            task_data = {
                "title": task_name,
                "description": response_content,
                "status": "todo",
                "priority": "medium"
            }
        
        # Generate task ID (you might want to use a proper ID generator)
        task_id = str(uuid.uuid4())[:50]  # Limit to 50 chars as per schema
        
        # Build task object according to tasks table schema
        # Merge AI-generated values with user-provided values (user values take precedence)
        current_time = int(time.time() * 1000)  # milliseconds timestamp
        current_date = datetime.date.today()
        
        # Merge tags: combine user tags with AI tags, removing duplicates
        ai_tags = task_data.get("tags", [])
        if not isinstance(ai_tags, list):
            ai_tags = []
        if user_tags:
            if isinstance(user_tags, list):
                combined_tags = list(set(user_tags + ai_tags))  # Remove duplicates
            else:
                combined_tags = [user_tags] + ai_tags if ai_tags else [user_tags]
        else:
            combined_tags = ai_tags
        
        # Determine assignee information (user-provided > AI-suggested)
        final_assignee_id = assignee_id or task_data.get("assignee_id")
        final_assignee_name = assignee_name
        final_assignee_avatar = assignee_avatar
        
        # Fetch assignee details from PostgreSQL if we have an assignee_id but missing name/avatar
        if final_assignee_id and (not final_assignee_name or not final_assignee_avatar):
            try:
                loop = asyncio.get_event_loop()
                user_details = await loop.run_in_executor(
                    None,
                    get_user_details_from_postgres,
                    final_assignee_id
                )
                if user_details:
                    if not final_assignee_name:
                        final_assignee_name = f"{user_details.get('first_name', '')} {user_details.get('last_name', '')}".strip()
                    if not final_assignee_avatar:
                        final_assignee_avatar = user_details.get("avatar", "")
            except Exception as e:
                logger.warning(f"Could not fetch assignee details: {str(e)}")
        
        # Determine project information (user-provided > AI-suggested)
        final_project_id = project_id or task_data.get("project_id")
        final_project_name = project_name or task_data.get("project_name")
        
        # Fetch project details from PostgreSQL if we have a project_id but missing name
        if final_project_id and not final_project_name:
            try:
                loop = asyncio.get_event_loop()
                project_details = await loop.run_in_executor(
                    None,
                    get_project_details_from_postgres,
                    user_id,
                    final_project_id
                )
                if project_details:
                    final_project_name = project_details.get("name", "")
            except Exception as e:
                logger.warning(f"Could not fetch project details: {str(e)}")
        
        task = {
            "id": task_id,
            "title": task_data.get("title", task_name),
            "description": user_description or task_data.get("description", ""),  # User description takes precedence
            "status": user_status or task_data.get("status", "todo"),  # User status takes precedence
            "priority": user_priority or task_data.get("priority", "medium"),  # User priority takes precedence
            "assignee_id": final_assignee_id,
            "assignee_name": final_assignee_name,
            "assignee_avatar": final_assignee_avatar,
            "project_id": final_project_id,
            "project_name": final_project_name,
            "created_by": created_by or user_id,
            "due_date": user_due_date or task_data.get("due_date"),  # User due_date takes precedence
            "created_date": current_date.isoformat(),
            "created_at": current_time,
            "tags": combined_tags,  # Merged tags
            "reason": user_reason or task_data.get("reason"),  # User reason takes precedence
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Validate status and priority values
        valid_statuses = ['todo', 'in-progress', 'review', 'completed', 'blocked']
        if task["status"] not in valid_statuses:
            task["status"] = "todo"
        
        valid_priorities = ['low', 'medium', 'high', 'urgent']
        if task["priority"] not in valid_priorities:
            task["priority"] = "medium"
        
        # Remove None values to match database defaults
        task = {k: v for k, v in task.items() if v is not None}
        
        total_time = time.time() - request_start_time
        logger.info(f"Total task generation time: {total_time:.3f}s")
        logger.info(f"Successfully generated task: {task_id}")
        
        return {"task": task}, 200
        
    except Exception as e:
        error_msg = f"Error generating task: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}, 500


def _markdown_to_html(markdown: str) -> str:
    """
    Convert markdown to HTML with improved formatting for meeting summaries.
    
    Supports:
    - Headers (#, ##, ###)
    - Bold (**text**)
    - Italic (*text*)
    - Bullet lists (-, *)
    - Numbered lists (1., 2., etc.)
    - Proper spacing and structure
    
    Args:
        markdown: Markdown content
        
    Returns:
        HTML formatted content
    """
    if not markdown:
        return ""
    
    lines = markdown.split('\n')
    html_lines = []
    in_ul = False
    in_ol = False
    in_paragraph = False
    
    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append('</ul>')
            in_ul = False
        if in_ol:
            html_lines.append('</ol>')
            in_ol = False
    
    def close_paragraph():
        nonlocal in_paragraph
        if in_paragraph:
            html_lines.append('</p>')
            in_paragraph = False
    
    def process_inline_formatting(text: str) -> str:
        """Process bold, italic, and other inline formatting"""
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="font-weight: 600;">\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong style="font-weight: 600;">\1</strong>', text)
        # Italic: *text* or _text_ (but not if it's part of **text**)
        # First handle bold, then italic
        text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<em style="font-style: italic;">\1</em>', text)
        text = re.sub(r'(?<!_)_(?!_)([^_]+?)(?<!_)_(?!_)', r'<em style="font-style: italic;">\1</em>', text)
        return text
    
    for i, line in enumerate(lines):
        original_line = line
        line = line.rstrip()
        
        # Empty line
        if not line:
            close_lists()
            close_paragraph()
            continue
        
        # Headers
        if line.startswith('###'):
            close_lists()
            close_paragraph()
            header_text = process_inline_formatting(line[3:].strip())
            html_lines.append(f'<h5 style="font-size: 1rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.75rem; color: #4b5563;">{header_text}</h5>')
            continue
        elif line.startswith('##'):
            close_lists()
            close_paragraph()
            header_text = process_inline_formatting(line[2:].strip())
            html_lines.append(f'<h4 style="font-size: 1.125rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.75rem; color: #374151;">{header_text}</h4>')
            continue
        elif line.startswith('#'):
            close_lists()
            close_paragraph()
            header_text = process_inline_formatting(line[1:].strip())
            html_lines.append(f'<h3 style="font-size: 1.25rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; color: #1f2937;">{header_text}</h3>')
            continue
        
        # Numbered list (1., 2., etc.)
        ol_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if ol_match:
            close_paragraph()
            if not in_ol:
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                html_lines.append('<ol style="margin: 0.75rem 0; padding-left: 1.5rem; list-style-type: decimal;">')
                in_ol = True
            item_text = process_inline_formatting(ol_match.group(2).strip())
            html_lines.append(f'<li style="margin: 0.5rem 0; line-height: 1.6;">{item_text}</li>')
            continue
        
        # Bullet list (- or *)
        if line.startswith('-') or line.startswith('*'):
            close_paragraph()
            if not in_ul:
                if in_ol:
                    html_lines.append('</ol>')
                    in_ol = False
                html_lines.append('<ul style="margin: 0.75rem 0; padding-left: 1.5rem; list-style-type: disc;">')
                in_ul = True
            # Remove the bullet and process formatting
            item_text = line[1:].strip()
            item_text = process_inline_formatting(item_text)
            html_lines.append(f'<li style="margin: 0.5rem 0; line-height: 1.6;">{item_text}</li>')
            continue
        
        # Regular paragraph text
        close_lists()
        if not in_paragraph:
            html_lines.append('<p style="margin: 0.75rem 0; line-height: 1.6;">')
            in_paragraph = True
        else:
            # Add space for line breaks within paragraph
            html_lines.append(' ')
        
        processed_text = process_inline_formatting(line)
        html_lines.append(processed_text)
    
    # Close any open tags
    close_lists()
    close_paragraph()
    
    # Join and clean up extra whitespace
    html = '\n'.join(html_lines)
    # Clean up multiple spaces
    html = re.sub(r' +', ' ', html)
    # Clean up spaces around tags
    html = re.sub(r'>\s+<', '><', html)
    # Add proper spacing between elements
    html = re.sub(r'(</(?:h[3-5]|p|ul|ol|li)>)', r'\1\n', html)
    html = re.sub(r'(<(?:h[3-5]|ul|ol|p)[^>]*>)', r'\n\1', html)
    html = re.sub(r'\n\n+', '\n', html)
    
    return html.strip()


def _clean_summary_content(content: str) -> str:
    """
    Clean up summary content by removing model's introductory text.
    Extracts only the actual meeting summary.
    
    Args:
        content: Raw summary content from the model
        
    Returns:
        Cleaned summary content with only the actual summary
    """
    if not content:
        return content
    
    # Look for where the actual summary starts
    # Common indicators: "Meeting Summary", "# Meeting Summary", "## Meeting Summary", etc.
    summary_start_patterns = [
        r"(?:^|\n)#+\s*Meeting\s+Summary",
        r"(?:^|\n)Meeting\s+Summary",
        r"(?:^|\n)#+\s*Summary",
    ]
    
    # Find the earliest occurrence of a summary header
    earliest_pos = None
    for pattern in summary_start_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            if earliest_pos is None or match.start() < earliest_pos:
                earliest_pos = match.start()
    
    # If we found a summary header, start from there
    if earliest_pos is not None:
        cleaned = content[earliest_pos:].lstrip()
        # Remove the "Meeting Summary" header if it's just plain text (not markdown)
        cleaned = re.sub(r"^(?:Meeting\s+Summary|Summary)\s*:?\s*\n+", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned.strip()
    
    # If no explicit header found, remove introductory sentences
    # Look for common intro patterns at the start
    lines = content.split('\n')
    start_idx = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Skip empty lines
        if not line_stripped:
            continue
        
        # If we hit a markdown header (#) or list item (-, *), we're in the summary
        if line_stripped.startswith('#') or line_stripped.startswith('-') or line_stripped.startswith('*'):
            start_idx = i
            break
        
        # If the line looks like an intro sentence, skip it
        intro_keywords = r"\b(I'll|I will|Let me|I'm going to|I can|I should|I'll analyze|I'll generate|I'll provide|Based on|Here is)"
        if re.search(intro_keywords, line_stripped, re.IGNORECASE):
            # Check if this is the last intro line (next non-empty line should be the summary)
            continue
        
        # If we get here and the line doesn't look like an intro, start from here
        start_idx = i
        break
    
    cleaned = '\n'.join(lines[start_idx:]).strip()
    
    # Final cleanup: remove any remaining intro patterns at the very start
    intro_patterns = [
        r"^I'll\s+(?:analyze|generate|provide|create).*?summary.*?\.\s*\n",
        r"^I'll\s+.*?comprehensive\s+meeting\s+summary.*?\.\s*\n",
        r"^Based\s+on\s+the\s+transcript.*?\.\s*\n",
        r"^Here\s+is\s+(?:a|the)\s+.*?summary.*?\.\s*\n",
    ]
    
    for pattern in intro_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    return cleaned.strip()


@app.route('/api/doc-summary', methods=['POST'])
@require_api_key
async def generate_doc_summary():
    """
    Generate a meeting doc summary for a doc.
    Reads the doc content, extracts the transcript, generates a summary using AI,
    and updates the doc with the summary shown before the transcript.
    
    Request body:
    {
        "doc_id": "doc-id-here",
        "user_id": "user@example.com",
        "org_slug": "org-slug"
    }
    """
    request_start_time = time.time()
    logger.info("Generate doc summary endpoint accessed")
    
    try:
        data = await request.get_json() or {}
        doc_id = data.get("doc_id")
        user_id = data.get("user_id")
        org_slug = data.get("org_slug")
        
        if not doc_id:
            return {"error": "doc_id is required"}, 400
        
        if not user_id:
            return {"error": "user_id is required"}, 400
        
        if not org_slug:
            return {"error": "org_slug is required"}, 400
        
        logger.info(f"Generating summary for doc_id: {doc_id}, user_id: {user_id}, org_slug: {org_slug}")
        
        # Query the doc from the database
        from app.services.database import query_org_one, execute_org
        
        doc = query_org_one(
            org_slug,
            "SELECT id, title, content, owner_email FROM docs WHERE id = %s",
            (doc_id,)
        )
        
        if not doc:
            return {"error": "Doc not found"}, 404
        
        # Verify the user has access to this doc (owner or in visible_to_members)
        owner_email = doc.get("owner_email", "").lower()
        user_id_lower = user_id.lower()
        
        # Check if user is owner
        if owner_email != user_id_lower:
            # Check if user is in visible_to_members
            doc_with_visibility = query_org_one(
                org_slug,
                "SELECT visibility, visible_to_members FROM docs WHERE id = %s",
                (doc_id,)
            )
            
            if doc_with_visibility:
                visibility = doc_with_visibility.get("visibility", "private")
                visible_to_members = doc_with_visibility.get("visible_to_members", [])
                
                if visibility == "private" or (visibility == "specific_members" and user_id_lower not in [m.lower() for m in visible_to_members]):
                    return {"error": "Access denied"}, 403
        
        # Extract transcript from doc content
        import re
        from html import unescape
        
        doc_content = doc.get("content", "")
        
        # Extract transcript section from HTML
        # The doc format from transcription worker has:
        # <div>
        #   <h2>Voice Call Transcript</h2>
        #   <p><strong>Date:</strong> ...</p>
        #   <p><strong>Participants:</strong> ...</p>
        #   <hr>
        #   <div>
        #     ${transcriptContent}
        #   </div>
        # </div>
        
        # Try to find content after <hr> tag (which separates metadata from transcript)
        hr_match = re.search(r'<hr[^>]*>(.*?)$', doc_content, re.DOTALL | re.IGNORECASE)
        if hr_match:
            transcript_html = hr_match.group(1).strip()
            # Remove the closing </div> tags at the end
            transcript_html = re.sub(r'</div>\s*$', '', transcript_html, flags=re.DOTALL)
        else:
            # Fallback: try to find content after "Voice Call Transcript" header
            h2_match = re.search(r'<h2[^>]*>Voice Call Transcript</h2>(.*?)$', doc_content, re.DOTALL | re.IGNORECASE)
            if h2_match:
                transcript_html = h2_match.group(1).strip()
                # Remove metadata lines (Date, Participants) and hr
                transcript_html = re.sub(r'<p><strong>(Date|Participants):</strong>.*?</p>', '', transcript_html, flags=re.IGNORECASE)
                transcript_html = re.sub(r'<hr[^>]*>', '', transcript_html, flags=re.IGNORECASE)
            else:
                # Last resort: use everything after first div or use entire content
                transcript_html = doc_content
        
        # Extract plain text from HTML for AI processing
        # Remove HTML tags but preserve structure
        text_content = re.sub(r'<[^>]+>', ' ', transcript_html)
        text_content = unescape(text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        if not text_content or len(text_content) < 50:
            return {"error": "Doc does not contain sufficient transcript content to generate a summary"}, 400
        
        # Initialize clients for AI
        try:
            firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
            logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
        except Exception as e:
            logger.error(f"Error initializing clients for user {user_id} in org slug {org_slug}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to initialize clients: {str(e)}"}, 500
        
        # Generate summary using ChatAgent with tools enabled for context gathering
        session_id = f"doc-summary-{doc_id}-{int(time.time())}"
        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id=session_id,
            clear_conversation=True,
            tools=available_tools  # Enable tools so agent can gather context when needed
        )
        
        # Create prompt for summary generation
        # Limit transcript to 8000 chars to avoid token limits
        transcript_for_ai = text_content[:8000]
        summary_prompt = f"""Please analyze the following meeting transcript and generate a comprehensive meeting summary.

The summary should include:
1. Key Topics Discussed - main topics and themes
2. Important Decisions Made - key decisions and rationale
3. Action Items and Next Steps - specific tasks with owners
4. Key Participants and Contributions - who said what and their roles
5. Deadlines and Important Dates - any time-sensitive items

If any part of the transcript lacks sufficient context for a good summary, use your tools to search for and gather relevant information (projects, tasks, teams, people, etc.) that would help provide better context.

Format the summary using markdown with:
- Use ## for main section headings (e.g., ## Key Topics Discussed)
- Use ### for subsections if needed
- Use bullet points (-) for lists
- Use **bold** for emphasis on important items
- Keep each section concise and well-organized
- Start directly with the content (no introductory text)

Transcript:
{transcript_for_ai}

Generate a concise but comprehensive summary:"""
        
        # Generate summary
        processing_start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.process_message, summary_prompt, None)
        processing_time = time.time() - processing_start_time
        logger.info(f"Summary generation completed in {processing_time:.3f}s")
        
        summary_content = response.get("content", "").strip()
        
        if not summary_content:
            return {"error": "Failed to generate summary"}, 500
        
        # Clean up the summary content - remove model's introductory text
        summary_content = _clean_summary_content(summary_content)
        
        # Convert markdown summary to HTML with improved formatting
        summary_html = _markdown_to_html(summary_content)
        
        # Update doc content to include summary before transcript
        updated_content = doc_content
        
        # Check if summary already exists (to avoid duplicates)
        summary_pattern = r'<div[^>]*>\s*<h2>Meeting Summary</h2>.*?</div>\s*<hr[^>]*>'
        if re.search(summary_pattern, updated_content, flags=re.DOTALL | re.IGNORECASE):
            # Replace existing summary
            summary_section = f'''<div style="margin-bottom: 2rem;">
                <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; color: #1f2937;">Meeting Summary</h2>
                <div style="line-height: 1.6; color: #374151;">
                    {summary_html}
                </div>
              </div>
              <hr style="margin: 2rem 0; border: none; border-top: 1px solid #e5e7eb;">'''
            
            updated_content = re.sub(
                summary_pattern,
                summary_section,
                updated_content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            # Insert summary before the transcript section
            summary_section = f'''<div style="margin-bottom: 2rem;">
                <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; color: #1f2937;">Meeting Summary</h2>
                <div style="line-height: 1.6; color: #374151;">
                    {summary_html}
                </div>
              </div>
              <hr style="margin: 2rem 0; border: none; border-top: 1px solid #e5e7eb;">'''
            
            # Insert before the first <hr> tag (which separates metadata from transcript)
            hr_match = re.search(r'(<hr[^>]*>)', updated_content, re.IGNORECASE)
            if hr_match:
                hr_pos = hr_match.start()
                updated_content = updated_content[:hr_pos] + summary_section + '\n' + updated_content[hr_pos:]
            else:
                # Fallback: insert after the header section
                h2_match = re.search(r'(<h2[^>]*>Voice Call Transcript</h2>)', updated_content, re.IGNORECASE)
                if h2_match:
                    h2_end = h2_match.end()
                    # Find the end of the metadata section (after Participants line)
                    participants_match = re.search(r'<p><strong>Participants:</strong>.*?</p>', updated_content, re.IGNORECASE | re.DOTALL)
                    if participants_match:
                        insert_pos = participants_match.end()
                        updated_content = updated_content[:insert_pos] + '\n' + summary_section + '\n' + updated_content[insert_pos:]
                    else:
                        updated_content = updated_content[:h2_end] + '\n' + summary_section + '\n' + updated_content[h2_end:]
                else:
                    # Last resort: prepend to content
                    updated_content = summary_section + '\n' + updated_content
        
        # Update the doc in the database
        execute_org(
            org_slug,
            "UPDATE docs SET content = %s, updated_at = NOW() WHERE id = %s",
            (updated_content, doc_id)
        )
        
        total_time = time.time() - request_start_time
        logger.info(f"Total doc summary generation time: {total_time:.3f}s")
        logger.info(f"Successfully generated and saved summary for doc: {doc_id}")
        
        return {
            "status": "success",
            "message": "Summary generated and saved to doc",
            "doc_id": doc_id
        }, 200
        
    except Exception as e:
        error_msg = f"Error generating doc summary: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}, 500


@app.route('/api/messages/generate-response', methods=['POST'])
@require_api_key
async def generate_message_response():
    """
    Generate a response to a message using ChatAgent.
    Supports dual-mode: frontend mode (with chatId) and independent API mode (without chatId).
    
    Request body:
    {
        "message": "Hello",                    // Required: Current user message
        "user_id": "user@example.com",        // Required: User identifier
        "org_slug": "my_org",                  // Required: Organization slug
        "chatId": "ai-assistant-user@example.com",  // Optional: Chat ID (frontend mode)
        "session_id": "custom-session-123",     // Optional: Session ID (defaults to chatId if provided, or generated if neither provided)
        "cited_context": "Context text"         // Optional: Cited context to include with the message
    }
    
    Returns:
    {
        "response": "Generated response text",
        "content": "Generated response text (same as response for compatibility)"
    }
    """
    request_start_time = time.time()
    logger.info("Generate message response endpoint accessed")
    
    try:
        data = await request.get_json()
        user_id = data.get("user_id")
        org_slug = data.get("org_slug")
        current_message = data.get("message")
        session_id = data.get("session_id")
        chat_id = data.get("chatId")
        
        # Validate required parameters
        if not user_id:
            return {"error": "user_id is required"}, 400
        
        if not org_slug:
            return {"error": "org_slug is required"}, 400
        
        if not current_message:
            return {"error": "message is required"}, 400
        
        # Determine mode and session_id
        if chat_id:
            # Frontend mode: Use chatId as session_id, load from messages collection
            agent_session_id = session_id or chat_id
            load_from_messages = True
            logger.info(f"Frontend mode: chatId={chat_id}, session_id={agent_session_id}")
        else:
            # Independent API mode: Use provided session_id or generate new one
            if session_id:
                agent_session_id = session_id
            else:
                # Generate new session_id for fresh conversation
                agent_session_id = f"api-session-{user_id}-{int(time.time())}"
            load_from_messages = False
            logger.info(f"Independent API mode: session_id={agent_session_id}")
        
        logger.info(f"Generating response for user_id: {user_id}, org_slug: {org_slug}, chat_id: {chat_id}, session_id: {agent_session_id}")
        
        # Initialize clients
        try:
            firestore_client, secret_manager_client, model_client, available_tools = await initialize_clients_async(user_id, org_slug)
            logger.info(f"Successfully initialized clients for user: {user_id} in org slug: {org_slug}")
        except Exception as e:
            logger.error(f"Error initializing clients for user {user_id} in org slug {org_slug}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to initialize clients: {str(e)}"}, 500
        
        # Initialize ChatAgent
        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id=agent_session_id,
            clear_conversation=False,  # Preserve memory (MemoryManager state)
            tools=available_tools
        )
        
        # Load conversation based on mode
        if load_from_messages:
            # Load from messages collection (frontend mode)
            # Pass current_message to exclude it from loaded history if it's already saved
            agent.conversation.load_conversation_from_messages(chat_id, limit=10, current_message=current_message)
        else:
            # ConversationManager auto-loads from files collection on init
            # (via load_conversation() in __init__)
            pass
        
        # Pass the current message directly - conversation history is now in ConversationManager
        # The system prompt and conversation context will provide all necessary context
        prompt = current_message
        
        # Extract cited_context from request if provided
        cited_context = data.get("cited_context") if "cited_context" in data else None
        if cited_context:
            # Log cited_context appropriately (handle both string and dict formats)
            if isinstance(cited_context, dict):
                logger.info(f"Cited context received in request (structured): {json.dumps(cited_context, default=str)[:200]}{'...' if len(json.dumps(cited_context, default=str)) > 200 else ''}")
            else:
                cited_str = str(cited_context)
                logger.info(f"Cited context received in request: {cited_str[:200]}{'...' if len(cited_str) > 200 else ''}")
        
        # Web app now sends HTML positions directly - no conversion needed
        
        # Add chat context if available (for project/team channels) - this can be added to the message
        if chat_id:
            if chat_id.startswith("project-"):
                project_id = chat_id.replace("project-", "")
                prompt += f"\n\n[Context: This conversation is in a project channel (project ID: {project_id})]"
            elif chat_id.startswith("team-"):
                team_id = chat_id.replace("team-", "")
                prompt += f"\n\n[Context: This conversation is in a team channel (team ID: {team_id})]"
        
        # Generate response using ChatAgent
        processing_start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.process_message, prompt, cited_context)
        processing_time = time.time() - processing_start_time
        logger.info(f"Message response generation completed in {processing_time:.3f}s")
        
        # Extract response content
        response_content = response.get("content", "").strip()
        
        if not response_content:
            return {"error": "Failed to generate response"}, 500
        
        total_time = time.time() - request_start_time
        logger.info(f"Total message response generation time: {total_time:.3f}s")
        logger.info(f"Successfully generated response for chat_id: {chat_id}, session_id: {agent_session_id}")
        
        # Return response in format compatible with existing message structure
        return {
            "response": response_content,
            "content": response_content  # For compatibility with existing code
        }, 200
        
    except Exception as e:
        error_msg = f"Error generating message response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}, 500


# Optional: Cache management endpoint (remove if not needed)
@app.route("/api/cache/clear", methods=["POST"])
async def clear_cache_endpoint():
    """Clear all caches (useful for debugging and maintenance)"""
    clear_cache()
    logger.info("All caches cleared")
    return {"status": "success", "message": "All caches cleared"}, 200

