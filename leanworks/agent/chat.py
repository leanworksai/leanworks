from leanworks.agent.tools.toolkit import ToolUse
from leanworks.agent.tools.duckdb import cleanup_responses, clear_session_response_ids
from leanworks.agent.helpers import AgentHelpers
from datetime import datetime, timezone
from leanworks.agent.conversation import ConversationManager
from leanworks.agent.memory import MemoryManager
from leanworks.agent.tool_registry import ToolRegistry
from leanworks.agent.tool_response_handler import ToolResponseHandlerFactory
from leanworks.setting import AGENT_SYSTEM_PROMPT, SEARCH_KNOWLEDGE_QUERY, EVALUATION_PROMPT, CRITIQUE_MESSAGE, GENERATION_MODEL
from google.cloud import firestore, secretmanager
from typing import Dict, Any, List
import traceback
import logging
import pytz
logger = logging.getLogger(__name__)

class ChatAgent:
    """
    A class that handles the chat interaction with Claude, including
    tool calls, verification, and conversation management.
    """
    
    def __init__(self, 
                 firestore_client,
                 secret_manager_client,
                 model_client,
                 user_id,
                 org_slug,
                 session_id=None,
                 clear_conversation=True,
                 tools=None,
                 additional_context=None,
                 credential_path: str = "gcp_credential.json",
                 ):
        """
        Initialize the ChatAgent with necessary clients and settings.
        
        Args:
            firestore_client: The official Firestore client (google.cloud.firestore.Client)
            secret_manager_client: The official Secret Manager client (google.cloud.secretmanager.SecretManagerServiceClient)
            model_client: The Claude model client for main chat
            user_id (str): The user ID for conversation tracking (email address)
            org_slug (str): The organization name for data isolation (e.g., 'leanworks.ai')
            session_id (str): The session ID for conversation tracking
            clear_conversation (bool): Whether to clear conversation history on init
            tools (list): List of additional tools to enable. These will be added to the default tools ['search', 'postgres', 'duckdb']. ToolUse handles the processing and filtering.
            credential_path (str): Path to GCP credential JSON file (default: "gcp_credential.json")
        """
        self.org_slug = org_slug
        if not self.org_slug:
            raise ValueError("org_slug is required for ChatAgent initialization")
        
        # Read project_id from credential file
        self.project_id = AgentHelpers.get_project_id_from_credentials(credential_path)
        
        # Store the original clients
        self.firestore_client = firestore_client
        self.secret_manager_client = secret_manager_client
        self.model_client = model_client
        
        # Set parameters
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize data source tracking
        self.data_sources = []
        
        # Initialize document ID tracking for aggressive deduplication
        self.read_document_ids = set()
        
        # Initialize selected text context (for position-based document editing)
        self.selected_text_context = None
        
        # Initialize tool use with org_slug and tools (passes session context for tools that can persist large results)
        self.tool_use = ToolUse(org_slug=self.org_slug, firestore_client=firestore_client, secret_manager_client=secret_manager_client, model_client=model_client, read_document_ids=self.read_document_ids, tools=tools, user_id=self.user_id, session_id=self.session_id, credential_path=credential_path)
        

        
        # Initialize memory management (always enabled)
        try:
            # Use model-aware factory method for optimal defaults
            self.memory_manager = MemoryManager.create_for_model(
                model_name=GENERATION_MODEL,
                model_client=model_client,
                firestore_client=firestore_client,
                org_slug=self.org_slug,
                user_id=user_id,
                session_id=session_id
            )
            logger.info(f"MemoryManager initialized for model {GENERATION_MODEL}")
            logger.info(f"Memory settings: {self.memory_manager.get_memory_stats()}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            self.memory_manager = None
        
        # Initialize conversation manager
        self.conversation = ConversationManager(
            self.model_client,
            self.firestore_client,
            self.org_slug,
            self.user_id,
            self.session_id,
            memory_manager=self.memory_manager
        )
        
        if clear_conversation:
            self.conversation.clear_conversation()
            # Also clear read document IDs when clearing conversation
            self.read_document_ids.clear()
            # Also clear memory when starting fresh
            if self.memory_manager:
                self.memory_manager.clear_memory()
        # When clear_conversation=False, keep existing memory for context continuity
        
        # Load conversation from messages collection (source of truth for all chat types)
        # This includes AI assistant chats, project channels, team channels, etc.
        # Do this after clear_conversation check so we don't clear what we just loaded
        if self.session_id and not clear_conversation:
            logger.info(f"Loading conversation history from messages collection for chatId: {self.session_id}")
            self.conversation.load_conversation_from_messages(
                chat_id=self.session_id,
                limit=10,
                exclude_last=False  # Don't exclude last message during initialization
            )
        
        # Get user info from Firestore
        user_info = self._get_user_info()
        user_timezone_str = user_info.get("timezone", "UTC") or "UTC"
        # Validate and convert timezone string to pytz timezone object
        try:
            user_timezone = pytz.timezone(user_timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone '{user_timezone_str}' for user {self.user_id}, defaulting to UTC")
            user_timezone = pytz.UTC
        # Set up API parameters for main model
        
        self.system_prompt = AGENT_SYSTEM_PROMPT.format(
            USER_INFO=user_info, 
            CURRENT_DATE_LOCAL=datetime.now(user_timezone).isoformat(),
            USER_TIMEZONE=user_timezone_str
        )
        
        # Set the system prompt and user profile for memory manager
        if self.memory_manager:
            self.memory_manager.set_system_prompt(self.system_prompt)
            # Pass user_info as dict so it can be updated later
            self.memory_manager.set_user_profile(user_info)
        
        # Initialize tool response handler factory
        self.tool_response_handler_factory = ToolResponseHandlerFactory()
        
        # Add server tools (execute on Anthropic's servers)
        # These are schema-less Anthropic-defined tools
        # Note: Only web_search_20250305 is currently supported as a server tool
        # web_fetch and tool_search may not be available yet
        server_tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search"
            }
        ]
        
        # Combine client tools with server tools
        all_tools = list(self.tool_use.tools) + server_tools
        
        # Log tools for debugging
        logger.info(f"Total tools registered: {len(all_tools)}")
        tool_names = [tool.get("name", "unknown") for tool in all_tools]
        logger.info(f"Tool names: {tool_names}")
        
        self.api_params = {
            "model": GENERATION_MODEL,
            "system": self.system_prompt,
            "messages": self.conversation.conversation,
            "tools": all_tools,
            "max_tokens": 8192,  # Increased from 1024 to support longer responses
            "temperature": 0.1,
            "timeout": 60
        }
        

    def _get_user_info(self):
        """
        Get user information from the organization database (users table).
        Falls back to default values if user not found or query fails.
        
        Returns:
            dict: User information dictionary with user_id, org_slug, timezone, and other fields
        """
        default_info = {
            "user_id": self.user_id or "Unknown", 
            "first_name": "", 
            "last_name": "", 
            "job_title": "",
            "responsibilities": "",
            "org_slug": self.org_slug or "",
            "timezone": "UTC",
            "work_style": ""
        }
        
        # Try to fetch user info from organization database
        if not self.user_id or not self.org_slug:
            return default_info
        
        try:
            from app.services.database import query_org_one
            
            user_data = query_org_one(
                self.org_slug,
                "SELECT email, first_name, last_name, job_title, timezone, responsibilities FROM users WHERE email = %s",
                (self.user_id.lower(),)
            )
            
            if user_data:
                return {
                    "user_id": self.user_id or "Unknown",
                    "first_name": user_data.get("first_name", ""),
                    "last_name": user_data.get("last_name", ""),
                    "job_title": user_data.get("job_title", ""),
                    "responsibilities": user_data.get("responsibilities", ""),
                    "org_slug": self.org_slug or "",
                    "timezone": user_data.get("timezone", "UTC") or "UTC",  # Fallback to UTC if None
                    "work_style": ""
                }
        except Exception as e:
            logger.warning(f"Could not fetch user info from org database for {self.user_id} (org: {self.org_slug}): {str(e)}")
            # Fall back to default info
        
        return default_info


    def _extract_user_message_from_conversation_history(self, user_message: str) -> str:
        """
        Extract the actual user message from embedded conversation history.
        
        The frontend embeds the last 5 messages (most recent) in the user message.
        This method extracts the actual current user message, which should be the last
        user message in the conversation history array.
        
        Args:
            user_message (str): The user message that may contain embedded conversation history
            
        Returns:
            str: The extracted actual user message (the most recent user message in the history)
        """
        # Check if the message contains conversation history markers
        if "## Conversation History" in user_message or '"role": "user"' in user_message:
            try:
                import json
                import re
                
                # Try to find and parse the conversation history JSON array
                # Look for JSON array pattern: [ {...}, {...} ]
                json_match = re.search(r'\[[\s\S]*?\]', user_message)
                if json_match:
                    conversation_json = json_match.group(0)
                    messages = json.loads(conversation_json)
                    
                    # Find the last user message in the conversation history
                    # Messages are typically ordered chronologically, so the last user message is the current one
                    for msg in reversed(messages):
                        if msg.get("role") == "user" and msg.get("content"):
                            content = msg.get("content")
                            # Handle both string and dict content formats
                            if isinstance(content, str):
                                actual_message = content
                            elif isinstance(content, list) and len(content) > 0:
                                # Handle Anthropic message format with content blocks
                                text_block = next((item.get("text", "") for item in content if item.get("type") == "text"), "")
                                if text_block:
                                    actual_message = text_block
                                else:
                                    actual_message = str(content[0]) if content else ""
                            else:
                                actual_message = str(content)
                            
                            # Remove any conversation history preamble from the message
                            # Remove any leading conversation history text
                            if actual_message.startswith("You are participating"):
                                # Extract just the actual question/statement
                                lines = actual_message.split("\n")
                                # Find the line after "Write your response message now"
                                for i, line in enumerate(lines):
                                    if "Write your response message now" in line or "CRITICAL INSTRUCTIONS" in line:
                                        # Take everything after this line as the actual message
                                        actual_message = "\n".join(lines[i+1:]).strip()
                                        break
                            
                            logger.info(f"Extracted actual user message from conversation history: {actual_message[:100]}...")
                            return actual_message
                    
                    # If no user message found, log warning and return original
                    logger.warning("Found conversation history but no user message in it, using original message")
                    return user_message
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse conversation history from user message: {e}")
                # Fallback: try to extract the last meaningful line
                lines = user_message.split("\n")
                # Look for the actual question (usually after "Write your response message now" or similar)
                for i, line in enumerate(lines):
                    if "Write your response message now" in line or "CRITICAL INSTRUCTIONS" in line:
                        remaining = "\n".join(lines[i+1:]).strip()
                        if remaining:
                            logger.info(f"Extracted user message from text pattern: {remaining[:100]}...")
                            return remaining
        
        # No conversation history detected, return as-is
        return user_message

    def process_message(self, user_message, cited_context=None, file_references=None, thinking=False, streaming=False):
        """
        Process a user message and handle the conversation flow.
        
        Args:
            user_message (str): The user's message content (may contain embedded conversation history)
            cited_context (str): The cited context for the user message
            file_references (list): List of file references from Claude Files API
                [
                    {
                        "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
                        "filename": "document.pdf",
                        "mime_type": "application/pdf",
                        "size_bytes": 1024000
                    },
                    ...
                ]
            thinking (bool): When True, enable evaluation-and-critique loop. When False, skip evaluation and return the first direct response.
            streaming (bool): When True, show tools being used and print response in a streaming way.
        Returns:
            dict: Dictionary with 'content' (response text) and 'data_sources' (list of sources)
        """
        # Extract the actual user message from embedded conversation history (if present)
        actual_user_message = self._extract_user_message_from_conversation_history(user_message)
        
        # Log the user query with context
        logger.info(f"User query received - user_id: {self.user_id}, session_id: {self.session_id}, org_slug: {self.org_slug}")
        logger.info(f"User query: {actual_user_message}")
        
        # Log cited context if provided
        if cited_context:
            logger.info(f"Cited context provided: {cited_context}")
        
        # Log file references if provided
        if file_references:
            logger.info(f"File references: {[f.get('filename', 'unknown') for f in file_references]}")
        
        # Load conversation history from messages collection (source of truth for all chat types)
        # This ensures we have the latest context from the channel before processing the new message
        if self.session_id:
            logger.info(f"Loading conversation history from messages collection before processing message: {self.session_id}")
            self.conversation.load_conversation_from_messages(
                chat_id=self.session_id,
                limit=10,
                exclude_last=True,  # Exclude the current message being processed
                current_message=actual_user_message
            )
        
        # Reset data sources for new message
        self.data_sources = []
        
        # Clear any previously tracked response IDs for cleanup
        clear_session_response_ids()
        
        # Store the original user query for evaluation (before adding cited context)
        self.original_user_query = actual_user_message
        logger.info(f"Stored original user query for evaluation: {self.original_user_query[:200]}...")
        
        # Log current state of document deduplication
        logger.info(f"Processing message with {len(self.read_document_ids)} documents already read for deduplication")
        
        # Process cited_context - handle both string and dict formats
        # Store selected text context for tools to access
        self.selected_text_context = None
        cited_context_str = None
        
        if cited_context:
            if isinstance(cited_context, dict):
                # Structured format - extract selectedText and format for LLM
                selected_text = cited_context.get("selectedText")
                if selected_text:
                    # Store for tools to access
                    self.selected_text_context = selected_text
                    logger.info(f"Stored selected text context: docId={selected_text.get('docId')}, htmlFrom={selected_text.get('htmlFrom')}, htmlTo={selected_text.get('htmlTo')}")
                    doc_id = selected_text.get("docId")
                    html_from = selected_text.get("htmlFrom")
                    html_to = selected_text.get("htmlTo")
                    if doc_id and html_from is not None and html_to is not None:
                        try:
                            doc_tool = self.tool_use.doc_management_tool
                            if doc_tool:
                                doc_tool.set_selected_text_positions(doc_id, html_from, html_to)
                        except Exception as e:
                            logger.warning(f"Failed to store selected text positions for doc {doc_id}: {e}")
                
                # Format structured context for LLM
                context_parts = []
                
                # Add projects if present
                projects = cited_context.get("projects", [])
                if projects:
                    project_names = [p.get("name", p.get("id", "")) for p in projects]
                    context_parts.append(f"Cited projects: {', '.join(project_names)}")
                
                # Add tasks if present
                tasks = cited_context.get("tasks", [])
                if tasks:
                    task_titles = [t.get("title", t.get("id", "")) for t in tasks]
                    context_parts.append(f"Cited tasks: {', '.join(task_titles)}")
                
                # Add docs if present
                docs = cited_context.get("docs", [])
                if docs:
                    doc_titles = [d.get("title", d.get("id", "")) for d in docs]
                    context_parts.append(f"Cited documents: {', '.join(doc_titles)}")
                
                # Add selected text if present
                if selected_text:
                    text = selected_text.get("text", "")
                    doc_id = selected_text.get("docId", "")
                    if text:
                        context_parts.append(f"Selected text from document {doc_id}: {text[:200]}{'...' if len(text) > 200 else ''}")
                
                cited_context_str = "\n".join(context_parts) if context_parts else None
            else:
                # String format (legacy)
                cited_context_str = str(cited_context)
        
        # Prepare user message (use the extracted actual message)
        user_message = actual_user_message
        if cited_context_str:
            user_message = f"<cited_context>{cited_context_str}</cited_context>\n{user_message}"
            # Log the final message with cited context
            logger.info(f"Final user message with cited context: {user_message}")
        
        # Build multimodal message content
        content_blocks = [{"type": "text", "text": user_message}]
        
        # Add file references to message content
        if file_references:
            for file_ref in file_references:
                file_id = file_ref.get("file_id")
                mime_type = file_ref.get("mime_type", "")
                filename = file_ref.get("filename", "unknown")
                
                if not file_id:
                    logger.warning(f"Skipping file reference without file_id: {filename}")
                    continue
                
                # Determine content block type based on MIME type
                if mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                    # Image content block
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "file",
                            "file_id": file_id
                        }
                    })
                    logger.info(f"Added image reference: {filename} ({file_id})")
                    
                elif mime_type in ["application/pdf", "text/plain"]:
                    # Document content block with citations enabled
                    content_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id
                        },
                        "title": filename,
                        "citations": {"enabled": True}  # Enable citations for PDFs
                    })
                    logger.info(f"Added document reference: {filename} ({file_id})")
                    
                else:
                    # For unsupported types, log warning
                    logger.warning(f"Unsupported MIME type for Files API: {mime_type} (file: {filename})")
        
        # Create user message object with multimodal content
        user_message_obj = {
            "role": "user",
            "content": content_blocks
        }
        
        # Add to memory manager if enabled
        if self.memory_manager:
            self.memory_manager.add_turn(user_message_obj)
            logger.info(f"Added user message to memory manager. Stats: {self.memory_manager.get_memory_stats()}")
        
        # Add the user message to conversation (multimodal support)
        if file_references:
            self.conversation.add_user_message_multimodal(content_blocks, include_in_slim=True)
        else:
            self.conversation.add_user_message(user_message, include_in_slim=True)
        
        # If using memory manager, update API params with memory context
        if self.memory_manager:
            memory_context, _ = self.memory_manager.get_context_for_inference()
            
            # Update system prompt to include memory context in the optimal position
            if memory_context:
                # Find the split point before <communication> section
                communication_start = self.system_prompt.find('<communication>')
                if communication_start != -1:
                    # Insert memory context before <communication> for better logical flow
                    before_communication = self.system_prompt[:communication_start].rstrip()
                    after_communication = self.system_prompt[communication_start:]
                    enhanced_system_prompt = f"{before_communication}\n\n{memory_context}\n\n{after_communication}"
                    logger.info("Injected memory context before <communication> section")
                else:
                    # Fallback to appending if <communication> section not found
                    enhanced_system_prompt = f"{self.system_prompt}\n\n{memory_context}"
                    logger.info("Appended memory context at end (communication section not found)")
            else:
                enhanced_system_prompt = self.system_prompt
                logger.info("No memory context to add")
            
            # IMPORTANT: Don't replace messages with memory messages during processing
            # We'll use current conversation messages but with enhanced system prompt
            self.api_params.update({
                "system": enhanced_system_prompt
                # Do NOT update messages here - use current conversation during tool loops
            })
            
            logger.info(f"Updated API params with memory context. Enhanced system prompt length: {len(enhanced_system_prompt)}")
        
        # Maximum number of iterations to prevent infinite loops
        unanswered_count = 0
        response_text = ""
        max_unanswered_num = 2
        
        # Reset retry flag for new message processing
        self._retry_attempted = False

        while unanswered_count < max_unanswered_num:
            logger.info(f"Unanswered attempt {unanswered_count}")
            try:
                # Always use current conversation messages during tool calling loops
                # This ensures tool calls and results are included in context
                current_params = self.conversation.create_params_copy(
                    self.api_params, 
                    messages=self.conversation.conversation  # Always use current conversation
                )
                
                # Make API call with current parameters
                # Ensure tools are always included
                if "tools" not in current_params or not current_params.get("tools"):
                    current_params["tools"] = self.api_params.get("tools", [])
                    logger.warning("Tools were missing from current_params, restored from api_params")
                
                # Log tools being sent for debugging
                tools_in_request = current_params.get("tools", [])
                logger.info(f"Making API call with {len(tools_in_request)} tools")
                if tools_in_request:
                    tool_names = [tool.get("name", tool.get("type", "unknown")) for tool in tools_in_request[:5]]
                    logger.info(f"Sample tools in request: {tool_names}")
                
                response = self.model_client.messages.create(**current_params)
                
                # Check stop_reason to understand why Claude stopped
                stop_reason = getattr(response, 'stop_reason', None)
                logger.info(f"Response stop_reason: {stop_reason}")
                
                # Handle different stop reasons
                if stop_reason == "refusal":
                    logger.warning("Claude refused to respond")
                    response_text = "I'm unable to process this request. Please try rephrasing your question."
                    self.conversation.add_assistant_message(response_text)
                    break
                elif stop_reason == "max_tokens":
                    logger.warning("Response truncated due to max_tokens limit")
                    # Check if there are tool calls that need execution
                    has_tool_use = any(
                        getattr(block, 'type', None) in ["tool_use", "server_tool_use"]
                        for block in response.content
                    )
                    
                    if has_tool_use:
                        # If there are tool calls, continue the conversation loop to execute them
                        logger.info("Response hit max_tokens but has tool calls - continuing conversation")
                        # Use unified handler to process response (will execute tools and continue)
                        handler = self.tool_response_handler_factory.get_handler(response)
                        context = {
                            'conversation': self.conversation,
                            'tool_use': self.tool_use,
                            'data_sources': self.data_sources,
                            'streaming': streaming,
                            'memory_manager': self.memory_manager
                        }
                        handler.handle(response, context)
                        # Continue the loop to get the next response
                        continue
                    else:
                        # No tool calls, just extract text and return
                        text_content = next((block.text for block in response.content if block.type == "text"), "")
                        if text_content:
                            response_text = text_content + "\n\n[Response truncated due to token limit. Please ask me to continue if you need more information.]"
                            self.conversation.add_assistant_message(response_text)
                            break
                elif stop_reason == "model_context_window_exceeded":
                    logger.warning("Response reached model's context window limit")
                    text_content = next((block.text for block in response.content if block.type == "text"), "")
                    if text_content:
                        response_text = text_content + "\n\n[Response truncated due to context window limit]"
                        self.conversation.add_assistant_message(response_text)
                        break
                
                # Use unified handler to process response
                handler = self.tool_response_handler_factory.get_handler(response)
                
                context = {
                    'conversation': self.conversation,
                    'tool_use': self.tool_use,
                    'data_sources': self.data_sources,
                    'streaming': streaming,
                    'memory_manager': self.memory_manager
                }
                
                final_text = handler.handle(response, context)
                
                if final_text:
                    # We have a final response (from server tools or text-only)
                    response_text = final_text
                    
                    # If not thinking, skip evaluation and return response directly
                    if not thinking:
                        logger.info("Thinking disabled: skipping evaluation and critique. Returning response directly.")
                        
                        # Stream the response if streaming is enabled
                        if streaming:
                            print("🤖 Assistant response:")
                            print("-" * 50)
                            AgentHelpers.stream_text(response_text, delay=0.01)
                            print("-" * 50)
                        
                        assistant_message_obj = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response_text}]
                        }
                        if self.memory_manager:
                            self.memory_manager.update_assistant_response(assistant_message_obj)
                            logger.info(f"Updated assistant response in memory manager. Stats: {self.memory_manager.get_memory_stats()}")
                        self.conversation.add_assistant_message(response_text)
                        break
                    
                    # Extract source content for evaluation
                    source_content = self._extract_source_content_from_conversation()
                    eval_feedback = self._perform_evaluation(response_text, source_content)
                    
                    eval_score = eval_feedback.get("score", 0)
                    eval_explanation = eval_feedback.get("explanation", "No explanation provided")
                    
                    if eval_score >= 7:
                        logger.info(f"Question answered with score {eval_score}.")
                        
                        # Stream the response if streaming is enabled
                        if streaming:
                            print("🤖 Assistant response (evaluated):")
                            print("-" * 50)
                            AgentHelpers.stream_text(response_text, delay=0.01)
                            print("-" * 50)
                        
                        # Create assistant message object
                        assistant_message_obj = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response_text}]
                        }
                        
                        # Update memory manager if enabled
                        if self.memory_manager:
                            self.memory_manager.update_assistant_response(assistant_message_obj)
                            logger.info(f"Updated assistant response in memory manager. Stats: {self.memory_manager.get_memory_stats()}")
                        
                        # Add to regular conversation first
                        self.conversation.add_assistant_message(response_text)
                        break
                    
                    # Score < 7: Implement in-place retry mechanism
                    logger.info(f"Response scored {eval_score} < 7. Attempting in-place retry.")
                    
                    # Check if this is our first retry attempt for this iteration
                    retry_attempted = getattr(self, '_retry_attempted', False)
                    
                    if not retry_attempted:
                        logger.info("Injecting critique and attempting regeneration...")
                        
                        # Mark that we've attempted retry for this iteration
                        self._retry_attempted = True
                        
                        # Add assistant message with the inadequate response (temporarily)
                        self.conversation.add_assistant_message(response_text)
                        
                        # Inject critique as a user message to guide regeneration
                        critique_message = CRITIQUE_MESSAGE.format(
                            eval_score=eval_score,
                            eval_explanation=eval_explanation
                        )
                        
                        self.conversation.add_user_message(critique_message)
                        
                        # Attempt regeneration with critique context
                        current_params = self.conversation.create_params_copy(
                            self.api_params, 
                            messages=self.conversation.conversation
                        )
                        
                        retry_response = self.model_client.messages.create(**current_params)
                        retry_text = next((block.text for block in retry_response.content if block.type == "text"), "")
                        retry_has_tools = any(block.type == "tool_use" for block in retry_response.content)
                        
                        if not retry_has_tools:
                            # Evaluate the retry response
                            retry_eval = self._perform_evaluation(retry_text, source_content)
                            retry_score = retry_eval.get("score", 0)
                            
                            if retry_score >= 7:
                                logger.info(f"Retry successful with score {retry_score}.")
                                response_text = retry_text
                                
                                # Stream the response if streaming is enabled
                                if streaming:
                                    print("🤖 Assistant response (retry successful):")
                                    print("-" * 50)
                                    AgentHelpers.stream_text(response_text, delay=0.01)
                                    print("-" * 50)
                                
                                # Remove the inadequate response and critique from conversation
                                self.conversation.conversation = self.conversation.conversation[:-2]
                                
                                # Create assistant message object
                                assistant_message_obj = {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": response_text}]
                                }
                                
                                # Update memory manager if enabled
                                if self.memory_manager:
                                    self.memory_manager.update_assistant_response(assistant_message_obj)
                                    logger.info(f"Updated assistant response in memory manager. Stats: {self.memory_manager.get_memory_stats()}")
                                
                                # Add final response to conversation
                                self.conversation.add_assistant_message(response_text)
                                break
                            else:
                                logger.info(f"Retry also scored low ({retry_score}). Falling back to search knowledge.")
                                response_text = retry_text
                                # Remove the inadequate response and critique, keep the retry response
                                self.conversation.conversation = self.conversation.conversation[:-2]
                                self.conversation.add_assistant_message(response_text)
                        else:
                            logger.info("Retry generated tool calls. Processing them normally.")
                            # Remove the inadequate response and critique from conversation
                            self.conversation.conversation = self.conversation.conversation[:-2]
                            # Add the retry response with tool calls and continue the loop
                            self.conversation.add_assistant_message_with_tool_uses(retry_response)
                            # Get RAG storage tool if available
                            rag_storage = None
                            if hasattr(self.tool_use, '_get_rag_storage_tool'):
                                rag_storage = self.tool_use._get_rag_storage_tool()
                            
                            tool_results = self.conversation.parse_and_format_tool_results_with_sources(
                                retry_response, 
                                self.tool_use.function_map,
                                self.data_sources,
                                rag_storage=rag_storage
                            )
                            self.conversation.add_tool_results(tool_results)
                            continue
                    
                    # Reset retry flag and proceed with search knowledge fallback
                    self._retry_attempted = False
                    logger.info("Falling back to search knowledge approach")
                    unanswered_count += 1

                    # Use search knowledge as fallback
                    self.conversation.add_user_message(SEARCH_KNOWLEDGE_QUERY.format(
                        USER_QUERY=self.original_user_query, 
                        LAST_RESPONSE=response_text, 
                        EVALUATION_FEEDBACK=eval_explanation
                    ))

                    # Find the search_documents tool specifically
                    search_documents_tool = None
                    for tool in self.tool_use.tools:
                        if tool.get("name") == "search_documents":
                            search_documents_tool = tool
                            break
                    
                    if not search_documents_tool:
                        logger.error("search_documents tool not found in available tools")
                        response_text = "Error: search_documents tool not available"
                        break
                    
                    # Create and immediately use forced search_documents parameters
                    forced_params = self.conversation.create_params_copy(
                        self.api_params,
                        messages=self.conversation.conversation,
                        tools=[search_documents_tool],
                        tool_choice={"type": "tool", "name": "search_documents"}
                    )
                    
                    # Make API call with forced search_documents
                    response = self.model_client.messages.create(**forced_params)
                    
                    # Process the forced tool call
                    self.conversation.add_assistant_message_with_tool_uses(response)
                    
                    # Get RAG storage tool if available
                    rag_storage = None
                    if hasattr(self.tool_use, '_get_rag_storage_tool'):
                        rag_storage = self.tool_use._get_rag_storage_tool()
                    
                    tool_results = self.conversation.parse_and_format_tool_results_with_sources(
                        response, 
                        self.tool_use.function_map,
                        self.data_sources,
                        rag_storage=rag_storage
                    )
                    self.conversation.add_tool_results(tool_results)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during unanswered attempt {unanswered_count}: {str(e)}")
                response_text = f"An error occurred: {str(e)}"
                break
            

        # Remove duplicates while preserving order
        unique_sources = []
        seen = set()
        for source in self.data_sources:
            if source not in seen:
                unique_sources.append(source)
                seen.add(source)

        # Note: Messages are saved to messages collection by the frontend (source of truth)
        # Only log full response if not streaming (to avoid duplicate output)
        if not streaming:
            logger.info(f"Final answer: {response_text}")
        else:
            logger.info(f"Final answer: {len(response_text)} characters")
        logger.info(f"Data sources used: {unique_sources}")
        logger.info(f"Session now has {len(self.read_document_ids)} total documents read (deduplicated)")
        
        # Show final summary if streaming is enabled
        if streaming and unique_sources:
            print("\n📚 Data sources used:")
            for i, source in enumerate(unique_sources, 1):
                print(f"   {i}. {source}")
            print()
        
        # Log memory stats if using memory manager
        if self.memory_manager:
            logger.info(f"Memory stats: {self.memory_manager.get_memory_stats()}")
            # Periodically update user profile based on conversation
            self._update_user_profile_from_conversation()
        
        # Return dictionary with content and data sources
        result = {
            "content": response_text,
            "data_sources": unique_sources
        }
        try:
            cleanup_responses()
        except Exception:
            pass
        return result
    
    def cleanup(self):
        """Clean up resources and shutdown background threads."""
        try:
            # Cleanup DuckDB temporary files
            try:
                cleanup_responses()
                logger.info("Cleaned up DuckDB response files")
            except Exception as e:
                logger.warning(f"Error cleaning up DuckDB files: {e}")
            
            # Cleanup RAG namespace data for this session
            if hasattr(self, 'tool_use') and self.tool_use and self.session_id:
                try:
                    rag_storage = self.tool_use._get_rag_storage_tool()
                    if rag_storage:
                        rag_storage.cleanup_session_data(self.session_id)
                        logger.info(f"Cleaned up RAG namespace data for session {self.session_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up RAG namespace data: {e}")
            
            # Clear working context
            if hasattr(self, 'memory_manager') and self.memory_manager and hasattr(self.memory_manager, 'working_context'):
                self.memory_manager.working_context.clear()

            # Shutdown memory manager
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.shutdown()
            
            logger.info("ChatAgent cleanup completed")
        except Exception as e:
            logger.error(f"Error during ChatAgent cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def _extract_source_content_from_conversation(self):
        """
        Extract the actual content from tool results in the conversation history
        along with their corresponding data sources.
        Returns formatted content with source attribution.
        """
        source_content_with_attribution = []
        source_index = 0

        # Build a map from tool_use_id -> {name, input} by scanning assistant tool_use blocks
        tool_use_meta = {}
        try:
            for message in self.conversation.conversation:
                if message.get("role") == "assistant" and isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id")
                            tool_name = block.get("name")
                            tool_input = block.get("input")
                            if tool_id:
                                tool_use_meta[tool_id] = {
                                    "name": tool_name,
                                    "input": tool_input
                                }
        except Exception:
            # If anything goes wrong, proceed without tool meta
            pass

        # Look through conversation history for tool_result messages
        for message in self.conversation.conversation:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                for content_block in message["content"]:
                    if content_block.get("type") == "tool_result":
                        tool_content = content_block.get("content", "")
                        if tool_content and not tool_content.startswith("Error"):
                            # Get corresponding data source if available
                            data_source = ""
                            if source_index < len(self.data_sources):
                                data_source = f"[Source: {self.data_sources[source_index]}]\n"
                                source_index += 1

                            # Create a descriptive title including tool name and input parameters (if available)
                            title = ""
                            try:
                                tool_use_id = content_block.get("tool_use_id")
                                meta = tool_use_meta.get(tool_use_id, {})
                                tool_name = meta.get("name") or "unknown_tool"
                                tool_input = meta.get("input")
                                if tool_input is not None:
                                    import json as _json
                                    inputs_str = _json.dumps(tool_input, ensure_ascii=False)
                                else:
                                    inputs_str = "{}"
                                title = f"TOOL RESULT - Tool: {tool_name}, Inputs: {inputs_str}"
                            except Exception:
                                title = ""

                            # Combine source attribution with title and content
                            if title:
                                attributed_content = f"{data_source}{title}\n{tool_content}"
                            else:
                                attributed_content = f"{data_source}{tool_content}"
                            source_content_with_attribution.append(attributed_content)

        return "\n\n---\n\n".join(source_content_with_attribution) if source_content_with_attribution else "No source content available."

    def _get_historical_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve historical conversation messages from Firestore across all sessions for this user.
        
        Args:
            limit: Maximum number of messages to retrieve (default: 50)
            
        Returns:
            List of message dictionaries in format: [{"role": "user/assistant", "content": "text"}, ...]
        """
        if not self.firestore_client or not self.user_id or not self.org_slug:
            return []
        
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            
            # Query messages collection for this user across all sessions
            messages_collection = self.firestore_client.collection('orgs').document(self.org_slug).collection('messages')
            
            # Query all messages for this user, ordered by timestamp descending (most recent first)
            query = messages_collection.where(
                filter=FieldFilter('userId', '==', self.user_id.lower())
            )
            
            # Try to order by timestamp, but handle index errors gracefully
            try:
                query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
            except Exception as e:
                # If index doesn't exist, we'll sort in memory
                if 'index' in str(e).lower() or (hasattr(e, 'code') and e.code == 9):
                    logger.debug(f"Index not found for timestamp ordering, will sort in memory: {e}")
                else:
                    raise
            
            query = query.limit(limit)
            messages = query.get()
            
            # Convert Firestore documents to message format with timestamp
            historical_messages_with_timestamp = []
            for doc in messages:
                data = doc.to_dict()
                role = data.get('role', '')
                content = data.get('content', '')
                timestamp = data.get('timestamp')
                
                if role and content:
                    historical_messages_with_timestamp.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                        "_timestamp": timestamp  # Keep timestamp for sorting
                    })
            
            # Sort by timestamp if we have timestamps (in case index wasn't available)
            if historical_messages_with_timestamp and historical_messages_with_timestamp[0].get("_timestamp"):
                try:
                    from datetime import datetime as dt
                    historical_messages_with_timestamp.sort(
                        key=lambda x: x.get("_timestamp") or dt.min,
                        reverse=True  # Most recent first
                    )
                except Exception as e:
                    logger.debug(f"Error sorting by timestamp: {e}")
            
            # Remove timestamp and reverse to get chronological order (oldest first)
            historical_messages = [
                {k: v for k, v in msg.items() if k != "_timestamp"}
                for msg in historical_messages_with_timestamp
            ]
            historical_messages.reverse()
            
            logger.info(f"Retrieved {len(historical_messages)} historical messages for user profile analysis")
            return historical_messages
            
        except Exception as e:
            logger.warning(f"Error retrieving historical messages: {str(e)}")
            return []
    
    def _update_user_profile_from_conversation(self):
        """
        Analyze conversation history (current session + historical) to update user profile (responsibilities and work_style).
        This runs periodically after responses to keep the profile up-to-date.
        Uses historical conversation data across all sessions, not just the current session.
        Only runs after every 3 new turns to prevent duplicate updates.
        """
        if not self.memory_manager:
            return
        
        try:
            # Get current session conversation turns
            current_turn_count = len(self.memory_manager.conversation_turns)
            last_update_turn_count = getattr(self.memory_manager, '_last_profile_update_turn_count', 0)
            
            # Only update if we have at least 3 new turns since last update
            turns_since_last_update = current_turn_count - last_update_turn_count
            if turns_since_last_update < 3:
                logger.debug(f"Skipping profile update: only {turns_since_last_update} turns since last update (need 3, current: {current_turn_count}, last: {last_update_turn_count})")
                return
            
            current_turns = self.memory_manager.conversation_turns
            
            # Get historical messages from Firestore (across all sessions)
            historical_messages = self._get_historical_messages(limit=50)
            
            # Combine current session turns with historical messages
            all_messages = []
            
            # Add historical messages first (they're already in chronological order)
            all_messages.extend(historical_messages)
            
            # Add current session turns (convert to message format)
            for turn in current_turns:
                user_msg = turn.get("user_message", {})
                assistant_msg = turn.get("assistant_message", {})
                
                if user_msg:
                    all_messages.append(user_msg)
                if assistant_msg:
                    all_messages.append(assistant_msg)
            
            # Only update if we have enough conversation data (at least 3 messages total)
            if len(all_messages) < 3:
                logger.debug(f"Insufficient conversation data for profile update: {len(all_messages)} messages")
                return
            
            # Use recent messages for analysis (last 20 messages to get good context)
            recent_messages = all_messages[-20:] if len(all_messages) > 20 else all_messages
            
            # Convert to text for analysis
            conversation_text = ""
            for msg in recent_messages:
                role = msg.get("role", "")
                text = self._extract_message_text(msg)
                
                if text:
                    conversation_text += f"{role.capitalize()}: {text}\n\n"
            
            if not conversation_text.strip():
                return
            
            # Get current user profile
            current_profile = self.memory_manager.get_user_profile_dict()
            current_responsibilities = current_profile.get("responsibilities", "")
            current_work_style = current_profile.get("work_style", "")
            
            # Create prompt for analyzing user profile
            analysis_prompt = f"""Analyze the following conversation to extract insights about the user's job responsibilities and work style.

Current user profile:
- Responsibilities: {current_responsibilities or "Not yet determined"}
- Work Style: {current_work_style or "Not yet determined"}

Recent conversation:
{conversation_text}

Based on this conversation, provide updated information about:
1. Job Responsibilities: Extract or refine the user's job responsibilities based on what they work on, tasks they mention, projects they're involved in, etc. If current responsibilities are empty or incomplete, add new information. If they already exist, update them with new insights.
2. Work Style: Identify the user's work style, preferences, and patterns. Consider:
   - Communication style (concise vs detailed, formal vs casual)
   - Problem-solving approach (analytical, creative, systematic, etc.)
   - Work preferences (collaborative vs independent, structured vs flexible)
   - Time management style
   - Tools and methods they prefer
   - How they approach tasks and projects

Return a JSON object with this structure:
{{
    "responsibilities": "Updated or refined job responsibilities based on conversation",
    "work_style": "Description of work style, preferences, and patterns observed"
}}

If there's not enough information to make meaningful updates, return the current values. Be specific and concrete based on what you observe in the conversation."""
            
            # Call Claude to analyze
            messages = [{"role": "user", "content": [{"type": "text", "text": analysis_prompt}]}]
            
            response = self.model_client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheaper model for analysis
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
                timeout=30
            )
            
            analysis_text = next((block.text for block in response.content if block.type == "text"), "")
            
            if analysis_text:
                # Parse JSON response
                import json
                import re
                try:
                    # Extract JSON from response
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = analysis_text[json_start:json_end]
                        
                        # Try to parse JSON - if it fails due to control characters, clean and retry
                        try:
                            analysis_result = json.loads(json_text)
                        except json.JSONDecodeError as json_error:
                            # If parsing fails due to control characters, clean the JSON text
                            # Control characters (0x00-0x1F) in JSON strings must be escaped
                            # Remove unescaped control characters that cause parsing errors
                            def clean_json_text(text):
                                # First, try to escape control characters in string values
                                # This is more robust than just removing them
                                import json as json_module
                                
                                # Remove unescaped control characters (0x00-0x1F) except newline, carriage return, tab
                                # These are the most common problematic ones
                                cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
                                
                                # Also try to fix common JSON issues:
                                # 1. Remove trailing commas before closing braces/brackets
                                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                                
                                # 2. Fix unescaped quotes in string values (basic attempt)
                                # This is tricky, so we'll be conservative
                                
                                return cleaned
                            
                            # Clean and retry
                            json_text_cleaned = clean_json_text(json_text)
                            try:
                                analysis_result = json.loads(json_text_cleaned)
                            except json.JSONDecodeError as retry_error:
                                # Try one more time with a more aggressive approach: extract JSON using balanced braces
                                try:
                                    # Find the first { and then match balanced braces
                                    start_idx = json_text_cleaned.find('{')
                                    if start_idx >= 0:
                                        brace_count = 0
                                        end_idx = start_idx
                                        for i in range(start_idx, len(json_text_cleaned)):
                                            if json_text_cleaned[i] == '{':
                                                brace_count += 1
                                            elif json_text_cleaned[i] == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    end_idx = i + 1
                                                    break
                                        if brace_count == 0:
                                            extracted_json = json_text_cleaned[start_idx:end_idx]
                                            analysis_result = json.loads(extracted_json)
                                        else:
                                            raise retry_error
                                    else:
                                        raise retry_error
                                except (json.JSONDecodeError, AttributeError, ValueError):
                                    # If all attempts fail, log and skip this update
                                    logger.warning(f"Failed to parse profile analysis JSON even after cleaning: {retry_error}. Original error: {json_error}")
                                    return  # Skip profile update if JSON can't be parsed
                        
                        new_responsibilities = analysis_result.get("responsibilities", current_responsibilities)
                        new_work_style = analysis_result.get("work_style", current_work_style)
                        
                        # Only update if there are meaningful changes
                        if new_responsibilities != current_responsibilities or new_work_style != current_work_style:
                            self.memory_manager.update_user_profile(
                                responsibilities=new_responsibilities if new_responsibilities != current_responsibilities else None,
                                work_style=new_work_style if new_work_style != current_work_style else None
                            )
                            # Update the counter to track when we last updated the profile
                            self.memory_manager._last_profile_update_turn_count = current_turn_count
                            # Save memory state to persist the counter
                            self.memory_manager.save_memory_state()
                            logger.info(f"Updated user profile from conversation analysis (after {current_turn_count} turns, {turns_since_last_update} new turns since last update)")
                        else:
                            # Even if no changes, update the counter to prevent re-analyzing the same data
                            self.memory_manager._last_profile_update_turn_count = current_turn_count
                            self.memory_manager.save_memory_state()
                            logger.debug(f"User profile analysis found no significant updates (counter updated to {current_turn_count})")
                    else:
                        logger.warning("Could not extract JSON from profile analysis response")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse profile analysis JSON: {e}")
        except Exception as e:
            logger.error(f"Error updating user profile from conversation: {str(e)}")
            # Don't fail the main conversation flow if profile update fails
    
    def _extract_message_text(self, message: Dict[str, Any]) -> str:
        """Extract readable text from a message in Claude format."""
        if not message:
            return ""
        
        content = message.get("content", [])
        
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Summarize tool use
                        tool_name = block.get("name", "unknown_tool")
                        text_parts.append(f"[Used tool: {tool_name}]")
                    elif block.get("type") == "tool_result":
                        # Summarize tool result
                        result_preview = str(block.get("content", ""))[:100]
                        text_parts.append(f"[Tool result: {result_preview}...]")
            return " ".join(text_parts)
        
        return ""
    
    def _perform_evaluation(self, response_text, source_content=""):
        """
        Performs evaluation on the response using a separate evaluation conversation flow.
        This prevents evaluation prompts from polluting the main conversation history.
        
        Args:
            response_text (str): The response text to evaluate
            source_content (str): The source content used to generate the response
            
        Returns:
            dict: Evaluation feedback with score and explanation, or default if evaluation failed
        """
        try:
            # Use the original user query stored when processing the message
            if not hasattr(self, 'original_user_query') or not self.original_user_query or not response_text:
                logger.warning("Missing original user query or response text for evaluation")
                return {"score": 5, "explanation": "Evaluation skipped due to missing inputs"}
            
            logger.info(f"Using stored original user query for evaluation: {self.original_user_query}")
            
            # Create separate evaluation conversation (does not pollute main conversation)
            eval_messages = []
            
            # Add evaluation prompt to separate conversation
            eval_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": EVALUATION_PROMPT.format(
                    USER_QUERY=self.original_user_query, 
                    LAST_RESPONSE=response_text
                )}]
            })
            logger.info(f"Evaluation messages: {eval_messages}")
            # Create evaluation parameters with separate conversation
            eval_params = {
                "model": GENERATION_MODEL,
                "messages": eval_messages,  # Use separate evaluation messages
                "max_tokens": 512,
                "temperature": 0.1,
                "timeout": 30
            }
            
            # Make evaluation API call using separate eval conversation
            eval_response = self.model_client.messages.create(**eval_params)
            eval_text = next((block.text for block in eval_response.content if block.type == "text"), "")
            
            # Parse evaluation JSON
            try:
                import json
                
                # Extract JSON portion from the response (handle cases where there's extra text)
                json_text = eval_text.strip()
                
                # Find the JSON object boundaries
                json_start = json_text.find('{')
                if json_start != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i in range(json_start, len(json_text)):
                        if json_text[i] == '{':
                            brace_count += 1
                        elif json_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end != -1:
                        json_only = json_text[json_start:json_end]
                        eval_feedback = json.loads(json_only)
                        logger.info(f"Evaluation feedback: {eval_feedback}")
                        return eval_feedback
                
                # Fallback: try parsing the entire text
                eval_feedback = json.loads(json_text)
                logger.info(f"Evaluation feedback: {eval_feedback}")
                return eval_feedback
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse evaluation JSON: {eval_text}")
                return {"score": 5, "explanation": "Evaluation failed due to JSON parsing error"}
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during evaluation: {str(e)}")
            return {"score": 5, "explanation": f"Evaluation failed due to error: {str(e)}"}
