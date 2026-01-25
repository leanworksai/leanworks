import json
import re
import copy
import logging
import time
import tempfile
import os
from typing import List, Dict, Any, Optional
from google.cloud import firestore

logger = logging.getLogger(__name__)
class ConversationManager:
    """Class for managing conversation history with Claude"""
    
    def __init__(self, model_client, firestore_client, org_slug, user_id=None, session_id=None, memory_manager=None, tool_use=None,
                 large_response_vectordb_client=None):
        self.model_client = model_client
        self.firestore_client = firestore_client
        self.org_slug = org_slug
        self.user_id = user_id
        self.session_id = session_id
        self.memory_manager = memory_manager
        self.tool_use = tool_use  # Store tool_use reference
        self.large_response_vectordb_client = large_response_vectordb_client
        self.conversation_path = f"chat_store/{self.user_id}/{self.session_id}"
        self.conversation = []

        # Initialize background indexing manager for RAG
        from leanworks.agent.tools.rag_storage import BackgroundIndexingManager
        from leanworks.setting import LARGE_RESPONSE_CONFIG
        max_workers = LARGE_RESPONSE_CONFIG.get("rag_indexing_thread_pool_size", 2)
        self.background_indexing_manager = BackgroundIndexingManager(max_workers=max_workers)

        # Note: Conversation is now loaded from messages collection (source of truth)
        # This is done in ChatAgent.__init__ or ChatAgent.process_message()
    
    def load_conversation_from_messages(self, chat_id: str, limit: int = 10, exclude_last: bool = True, current_message: str = None):
        """
        Load conversation from Firestore messages collection by chatId.
        
        This method loads recent messages from the messages collection (source of truth for chat UI)
        and populates the conversation manager. This is used when chatId is provided (frontend mode).
        
        Args:
            chat_id: Chat ID to query messages for
            limit: Maximum number of messages to load (default: 10)
            exclude_last: If True, exclude messages matching the current message being processed
            current_message: The current user message being processed (used for content matching)
        """
        if not self.firestore_client or not self.org_slug:
            logger.warning("Cannot load conversation from messages: missing firestore_client or org_slug")
            return
        
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            
            # Build collection path
            messages_path = f"orgs/{self.org_slug}/messages"
            messages_collection = self.firestore_client.collection(messages_path)
            
            # Start building query
            query = messages_collection.where(filter=FieldFilter('chatId', '==', chat_id))
            
            # Security: For AI assistant conversations, filter by userId
            if chat_id.startswith('ai-assistant-'):
                if self.user_id and not chat_id.endswith(f"-{self.user_id}"):
                    logger.warning(f"Access denied: Cannot access another user's AI assistant conversation. chatId: {chat_id}, user_id: {self.user_id}")
                    self.conversation = []
                    return
                if self.user_id:
                    query = query.where(filter=FieldFilter('userId', '==', self.user_id.lower()))
            
            # Order by timestamp descending (most recent first) to get the latest messages
            try:
                query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
            except Exception as e:
                # If index doesn't exist, we'll sort in memory
                if 'index' in str(e).lower() or (hasattr(e, 'code') and e.code == 9):
                    logger.debug(f"Index not found for timestamp ordering, will sort in memory: {e}")
                else:
                    raise
            
            # Limit results to get the most recent N messages
            query = query.limit(limit)
            
            # Execute query with retry logic for transient errors
            logger.debug(f"Loading conversation from messages collection: chatId={chat_id}, limit={limit}")
            messages = self._execute_firestore_query_with_retry(query, max_retries=3)
            
            # Convert Firestore documents to conversation format
            conversation_messages = []
            for doc in messages:
                data = doc.to_dict()
                role = data.get('role', '')
                content = data.get('content', '')
                
                if role and content:
                    # Convert to conversation format: {"role": "...", "content": [{"type": "text", "text": "..."}]}
                    conversation_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
            
            # Query returns messages in DESCENDING order (newest first), so reverse to get chronological order (oldest first)
            if len(conversation_messages) > 0:
                conversation_messages.reverse()
            
            # Exclude the last message if it matches the current message being processed
            if exclude_last and len(conversation_messages) > 0:
                if current_message:
                    # Check only the last message: remove if it's a user message with matching content
                    last_msg = conversation_messages[-1]
                    last_msg_content = last_msg.get('content', [{}])[0].get('text', '').strip()
                    last_msg_role = last_msg.get('role', '')
                    current_message_stripped = current_message.strip()
                    
                    # Exclude if the last message is a user message with matching content
                    if last_msg_role == 'user' and last_msg_content == current_message_stripped:
                        conversation_messages = conversation_messages[:-1]
                else:
                    # Fallback: just exclude last message if current_message not provided
                    conversation_messages = conversation_messages[:-1]
            
            # Set conversation (excluding the last message which is the current one being processed)
            if len(conversation_messages) > 0:
                self.conversation = conversation_messages
                logger.debug(f"Loaded {len(self.conversation)} messages from messages collection for chatId: {chat_id}")
            else:
                self.conversation = []
                logger.debug(f"No messages found for chatId: {chat_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation from messages collection: {e}")
            import traceback
            traceback.print_exc()
            self.conversation = []
    
    def _execute_firestore_query_with_retry(self, query, max_retries=3):
        """
        Execute a Firestore query with retry logic for transient errors.
        
        Args:
            query: Firestore query object
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Query results
            
        Raises:
            Exception: If all retries fail
        """
        import random
        from google.api_core import exceptions as gcp_exceptions
        
        for attempt in range(max_retries + 1):
            try:
                return query.get()
            except Exception as e:
                error_str = str(e)
                error_lower = error_str.lower()
                
                # Check if this is a transient error that should be retried
                is_transient = (
                    "stream removed" in error_lower or
                    "ssl" in error_lower or
                    "decryption error" in error_lower or
                    "corruption detected" in error_lower or
                    "connection" in error_lower or
                    isinstance(e, gcp_exceptions.ServiceUnavailable) or
                    isinstance(e, gcp_exceptions.DeadlineExceeded) or
                    isinstance(e, gcp_exceptions.InternalServerError)
                )
                
                if is_transient and attempt < max_retries:
                    # Exponential backoff with jitter
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0.1, 0.5)
                    delay = base_delay + jitter
                    
                    logger.warning(
                        f"Transient Firestore error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:200]}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Not a transient error or max retries exceeded
                    logger.error(f"Firestore query failed after {attempt + 1} attempts: {error_str}")
                    raise
    
    def add_user_message(self, content, include_in_slim=False):
        """Add a user message to the conversation history"""
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": content}]
        }
        self.conversation.append(user_message)
    
    def add_user_message_multimodal(self, content_blocks: List[Dict], include_in_slim: bool = False):
        """
        Add a multimodal user message with content blocks.
        
        Args:
            content_blocks: List of content blocks (text, image, document)
            include_in_slim: Whether to include in slim conversation (deprecated, ignored)
        """
        message = {
            "role": "user",
            "content": content_blocks
        }
        self.conversation.append(message)

    def add_tool_results(self, tool_results):
        """Add tool results to the conversation history
        
        Tool results must be added immediately after an assistant message
        that contains the corresponding tool_use blocks.
        """
        # Only add tool results if there's at least one previous message
        if not self.conversation:
            print("Warning: Cannot add tool results to an empty conversation")
            return
            
        # The previous message should be from the assistant
        prev_message = self.conversation[-1]
        if prev_message.get("role") != "assistant":
            print("Warning: Tool results should follow an assistant message")
            # Still proceed with adding the results
            
        # Add the tool results as a user message
        self.conversation.append({
            "role": "user",
            "content": tool_results
        })

    def add_assistant_message(self, content, include_in_slim=False):
        """Add an assistant message to the conversation history"""
        self.conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": content}]
        })
            
    def clear_conversation(self):
        """Clear the conversation history and start fresh"""
        self.conversation = []
        
    def reset_for_fresh_attempt(self):
        """Reset the conversation to just the initial user query if problems occur"""
        if len(self.conversation) > 0:
            initial_message = self.conversation[0]
            self.conversation = [initial_message]
            print("Conversation reset to initial query for fresh attempt")

    def parse_and_format_tool_results(self, response, function_map):
        """
        Parse tool use blocks from Claude's response and execute the corresponding functions.
        Returns properly formatted tool_result content blocks following Anthropic's API format.
        """
        tool_results = []
        
        for block in response.content:
            if block.type == "tool_use":
                # Extract tool details - handle different API versions
                try:
                    tool_use_id = block.id  # For newer versions of Claude API
                except AttributeError:
                    tool_use_id = block.tool_use.id  # For Claude API 3.5
                
                try:
                    tool_name = block.name  # For newer versions of Claude API 
                    tool_input = block.input
                except AttributeError:
                    tool_name = block.tool_use.name  # For Claude API 3.5
                    tool_input = block.tool_use.input
                
                # Execute the tool function if it exists in our function map
                if tool_name in function_map:
                    try:
                        # Log tool call with parameters
                        logger.info(f"Tool call: {tool_name} with parameters: {json.dumps(tool_input, default=str)}")

                        # Call the function with the provided input
                        result = function_map[tool_name](**tool_input)

                        # Log tool call result preview
                        result_preview = self._get_result_preview(result)
                        logger.info(f"Tool call result for {tool_name}: {result_preview}")

                        # If tool returns an error object, surface just the error message
                        if isinstance(result, dict) and "error" in result:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(result["error"]),
                                "is_error": True
                            })
                            continue

                        # Format the result to have proper content structure
                        formatted_result = []
                        if isinstance(result, list):
                            for item in result:
                                if isinstance(item, dict):
                                    # For dictionaries, convert to a JSON string
                                    formatted_result.append(json.dumps(item))
                                elif isinstance(item, str):
                                    # For strings, just add the string
                                    formatted_result.append(item)
                                else:
                                    # For any other type, convert to string
                                    formatted_result.append(str(item))
                            # Join all items with newlines
                            formatted_result = "\n".join(formatted_result)
                        elif isinstance(result, str):
                            # If result is a string, use it directly
                            formatted_result = result
                        elif isinstance(result, dict):
                            # If result is a dictionary, convert to JSON string
                            formatted_result = json.dumps(result)
                        else:
                            # For any other type, convert to string
                            formatted_result = str(result)
                        
                        # Format as proper tool_result content block
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": formatted_result
                        })
                    except Exception as e:
                        # Handle errors in tool execution with is_error flag
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Error executing {tool_name}: {str(e)}",
                            "is_error": True
                        })
                else:
                    # Handle unknown tool as an error
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Unknown tool: {tool_name}",
                        "is_error": True
                    })
        
        return tool_results

    def parse_and_format_tool_results_with_sources(self, response, function_map, data_sources, rag_storage=None):
        """
        Parse tool use blocks from Claude's response and execute the corresponding functions.
        Returns properly formatted tool_result content blocks following Anthropic's API format.
        Also tracks data sources used by each tool.
        
        Args:
            response: Claude API response object
            function_map: Dictionary mapping tool names to functions
            data_sources: List to append data source information
            rag_storage: Optional RAGStorageTool (deprecated - unstructured responses now stored as text files)
        """
        tool_results = []
        
        for block in response.content:
            if block.type == "tool_use":
                # Extract tool details - handle different API versions
                try:
                    tool_use_id = block.id  # For newer versions of Claude API
                except AttributeError:
                    tool_use_id = block.tool_use.id  # For Claude API 3.5
                
                try:
                    tool_name = block.name  # For newer versions of Claude API 
                    tool_input = block.input
                except AttributeError:
                    tool_name = block.tool_use.name  # For Claude API 3.5
                    tool_input = block.tool_use.input
                
                # Check if this is a server tool (executed by Anthropic's servers)
                # Server tools don't need client-side execution - Anthropic handles them automatically
                from leanworks.agent.tool_registry import ToolRegistry
                is_server_tool = ToolRegistry.is_server_tool(tool_name)
                
                if is_server_tool:
                    # Server tools are executed by Anthropic's servers automatically
                    # Results are already included in the response - we don't need to process them
                    # Just track the data source and skip adding tool_result
                    logger.info(f"Server tool call: {tool_name} with parameters: {json.dumps(tool_input, default=str)} - results already in response from Anthropic")
                    # Track data source for web_search
                    if tool_name == "web_search":
                        data_sources.append("Web search results")
                    # Don't add tool_result - results are already in the response
                    # The calling code in chat.py will handle continuing the conversation
                    continue
                
                # Execute the tool function if it exists in our function map
                if tool_name in function_map:
                    try:
                        # Log client-side tool call at INFO level
                        logger.info(f"Client tool call: {tool_name} with parameters: {json.dumps(tool_input, default=str)}")

                        # Call the function with the provided input
                        result = function_map[tool_name](**tool_input)

                        # Log tool call result preview
                        result_preview = self._get_result_preview(result)
                        logger.info(f"Tool call result for {tool_name}: {result_preview}")

                        # Check if response is large and needs special handling
                        from leanworks.agent.large_response_handler import LargeResponseHandler, ResponseType
                        from leanworks.setting import LARGE_RESPONSE_CONFIG
                        
                        # Configure handler with settings
                        LargeResponseHandler.configure(LARGE_RESPONSE_CONFIG)
                        
                        # Classify response
                        response_type, is_large = LargeResponseHandler.classify_response(result)

                        # Log classification results
                        logger.info(f"Response classification for {tool_name}: type={response_type.value}, is_large={is_large}, auto_store_enabled={LARGE_RESPONSE_CONFIG.get('auto_store_enabled', True)}")

                        if is_large and LARGE_RESPONSE_CONFIG.get("auto_store_enabled", True):
                            logger.info(f"Large response detected for {tool_name}, routing to storage handler")
                            # Extract doc IDs for doc tools (preprocessing moved to storage handler)
                            doc_ids = []
                            if self._is_doc_content_tool(tool_name):
                                if isinstance(result, list):
                                    doc_ids = [doc.get('id') for doc in result if isinstance(doc, dict) and doc.get('id')]

                            # Use unified handler for all tools
                            formatted_result = self.handle_large_response(
                                result, tool_name, tool_input, tool_use_id, data_sources, doc_ids=doc_ids
                            )
                            logger.info(f"Large response storage completed for {tool_name}. Final result content: {formatted_result.get('content', '')[:200]}...")
                            tool_results.append(formatted_result)
                            continue
                        else:
                            logger.info(f"Response is small or auto-store disabled for {tool_name}, using direct response")
                        
                        # Track data sources based on tool type
                        if tool_name == "query_postgres":
                            # For PostgreSQL tools, extract table names from SQL query
                            try:
                                # Try to get org_slug from the function's bound method
                                postgres_tool = getattr(function_map[tool_name], '__self__', None)
                                org_slug = getattr(postgres_tool, 'org_slug', 'leanworks.ai') if postgres_tool else 'leanworks.ai'
                                
                                # Extract SQL query from tool_input
                                sql_query = tool_input.get('sql', '') if isinstance(tool_input, dict) else ''
                                
                                if sql_query:
                                    # Parse SQL to extract table names
                                    table_names = self._extract_table_names_from_sql(sql_query)
                                    if table_names:
                                        tables_str = ', '.join(table_names)
                                        data_sources.append(f"PostgreSQL tables: {tables_str} (org: {org_slug})")
                                    else:
                                        data_sources.append(f"PostgreSQL database (org: {org_slug})")
                                else:
                                    data_sources.append(f"PostgreSQL database (org: {org_slug})")
                            except Exception as e:
                                logger.warning(f"Failed to extract table names from SQL: {e}")
                                data_sources.append("PostgreSQL database")
                        
                        elif tool_name in ["find_available_slots", "list_upcoming_meetings"]:
                            # For Outlook/calendar tools, add calendar data source
                            data_sources.append("Outlook Calendar")
                        
                        elif tool_name in ["query_duckdb", "execute_duckdb_query"]:
                            # For DuckDB tools, add database data source
                            data_sources.append("DuckDB Database")
                        
                        
                        elif tool_name == "search_documents":
                            # For search_documents, use the data sources directly from the result
                            if hasattr(result, '_search_data_sources'):
                                # Use the actual data sources returned by the search tool
                                for source in result._search_data_sources:
                                    if source:
                                        data_sources.append(f"Knowledge base: {source}")
                            else:
                                # Fallback to parsing if the attribute is not available
                                if isinstance(result, str):
                                    lines = result.split('\n')
                                    for line in lines:
                                        if '- Source:' in line:
                                            source_part = line.split('- Source:')[1].strip()
                                            if source_part and source_part != ']':
                                                data_sources.append(f"Knowledge base: {source_part}")
                                        elif '[DOCUMENT - Date:' in line and '- Source:' in line:
                                            # Extract source from the document header format
                                            try:
                                                source_start = line.find('- Source:') + len('- Source:')
                                                source_end = line.find(']:')
                                                if source_end == -1:
                                                    source_end = len(line)
                                                source_part = line[source_start:source_end].strip()
                                                if source_part:
                                                    data_sources.append(f"Knowledge base: {source_part}")
                                            except Exception as e:
                                                logger.warning(f"Error parsing source from line: {line}, error: {e}")

                        elif tool_name in ["get_doc", "search_docs"]:
                            # Extract document titles/IDs from original result for tracking
                            if isinstance(result, list):
                                doc_titles = [doc.get('title', doc.get('id', 'Unknown')) for doc in result if isinstance(doc, dict)]
                                if doc_titles:
                                    data_sources.append(f"Documents: {', '.join(doc_titles[:3])}")

                        # If tool returns an error object, surface just the error message
                        if isinstance(result, dict) and "error" in result:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(result["error"]),
                                "is_error": True
                            })
                            continue

                        # Format the result to have proper content structure
                        formatted_result = []
                        if isinstance(result, list):
                            for item in result:
                                if isinstance(item, dict):
                                    # For dictionaries, convert to a JSON string
                                    formatted_result.append(json.dumps(item))
                                elif isinstance(item, str):
                                    # For strings, just add the string
                                    formatted_result.append(item)
                                else:
                                    # For any other type, convert to string
                                    formatted_result.append(str(item))
                            # Join all items with newlines
                            formatted_result = "\n".join(formatted_result)
                        elif isinstance(result, str):
                            # If result is a string, use it directly
                            formatted_result = result
                        elif isinstance(result, dict):
                            # If result is a dictionary, convert to JSON string
                            formatted_result = json.dumps(result)
                        else:
                            # For any other type, convert to string
                            formatted_result = str(result)
                        
                        # Format as proper tool_result content block
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": formatted_result
                        })
                    except Exception as e:
                        # Handle errors in tool execution with is_error flag
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Error executing {tool_name}: {str(e)}",
                            "is_error": True
                        })
                else:
                    # Handle unknown tool as an error
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Unknown tool: {tool_name}",
                        "is_error": True
                    })
        
        return tool_results

    def _ensure_docker_container_initialized(self) -> bool:
        """
        Ensure Docker container is initialized and healthy before file operations.

        This prevents race conditions where files are saved to temp directories
        before the Docker container's session directory is created.

        Returns:
            bool: True if container is ready, False if initialization failed
        """
        if not self.tool_use:
            logger.warning("No tool_use available for Docker container initialization")
            return False

        # Check if bash session already exists
        if hasattr(self.tool_use, '_bash_session') and self.tool_use._bash_session:
            # Verify container is still running
            try:
                session = self.tool_use._bash_session
                import subprocess
                check_cmd = ['docker', 'inspect', '--format', '{{.State.Running}}', session.container_name]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)

                if check_result.returncode == 0 and check_result.stdout.strip() == 'true':
                    logger.debug(f"Docker container {session.container_name} is healthy")
                    return True
            except Exception as e:
                logger.warning(f"Docker health check failed: {e}")

        # Initialize Docker container by calling bash tool
        try:
            logger.debug("Initializing Docker container for file operations")
            # Simple echo command to trigger container creation
            result = self.tool_use.bash("echo 'Docker initialized'")

            if "Error" in result:
                logger.error(f"Failed to initialize Docker container: {result}")
                return False

            logger.debug("Docker container initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Exception during Docker initialization: {e}")
            return False

    def handle_large_response(
        self,
        result: Any,
        tool_name: str,
        tool_input: Dict,
        tool_use_id: str,
        data_sources: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Unified entry point for handling all large tool responses.
        Routes to appropriate storage method based on data format and complexity.

        Args:
            result: The tool response data
            tool_name: Name of the tool that generated the response
            tool_input: Input parameters passed to the tool
            tool_use_id: Unique identifier for this tool use
            data_sources: List to append data source tracking info

        Returns:
            Formatted tool result with storage instructions
        """
        from leanworks.agent.large_response_handler import LargeResponseHandler, ResponseType

        # Classify response type and check if large
        response_type, is_large = LargeResponseHandler.classify_response(result)

        if not is_large:
            # Small response - should not reach here, but fallback
            logger.warning(f"handle_large_response called with small response: {response_type}")
            return self._truncate_response(result, tool_use_id)

        logger.info(f"Routing large response for {tool_name}: type={response_type.value}")

        # Route to appropriate handler based on classification
        if response_type in [ResponseType.STRUCTURED_SIMPLE, ResponseType.STRUCTURED_COMPLEX]:
            logger.info(f"Storing {tool_name} structured response as JSON file (no vectordb indexing)")
            return self._handle_structured_response(
                result, tool_name, tool_input, tool_use_id, data_sources, doc_ids=doc_ids
            )

        elif response_type == ResponseType.UNSTRUCTURED:
            logger.info(f"Storing {tool_name} unstructured text response with hybrid file + RAG indexing")
            return self._handle_hybrid_text_and_rag_storage(
                result, tool_name, tool_input, tool_use_id, data_sources, doc_ids=doc_ids
            )

        elif response_type == ResponseType.MIXED:
            logger.info(f"Storing {tool_name} mixed response - splitting into structured and unstructured parts")
            # Split and handle both parts
            structured_part, unstructured_part = LargeResponseHandler.split_mixed_response(result)
            results = []

            # Handle structured part (use unified JSON handler)
            if structured_part:
                structured_result = self._handle_structured_response(
                    structured_part, tool_name, tool_input, tool_use_id, data_sources
                )
                results.append(structured_result)

            # Handle unstructured part (hybrid approach)
            if unstructured_part:
                unstructured_result = self._handle_hybrid_text_and_rag_storage(
                    unstructured_part, tool_name, tool_input, tool_use_id, data_sources
                )
                results.append(unstructured_result)

            # Return the first result (primary) - mixed responses are rare
            return results[0] if results else self._truncate_response(result, tool_use_id)

        else:
            # Unknown type - fallback to truncation
            logger.warning(f"Unknown response type: {response_type}")
            return self._truncate_response(result, tool_use_id)

    def _handle_structured_response(
        self,
        result: Any,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: str,
        data_sources: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Handle ALL structured responses (SIMPLE and COMPLEX):
        Save as JSON file only (no vectordb indexing).
        Users access via jq, grep, or direct viewing.
        
        Args:
            result: Structured data to store
            tool_name: Name of the tool that generated this
            tool_input: Input parameters to the tool
            tool_use_id: Tool use ID for tracking
            data_sources: List to append data source info
            doc_ids: Optional document IDs (unused for structured data)
            
        Returns:
            Formatted result with jq/grep instructions
        """
        import json
        
        # Save JSON file (pretty-printed)
        json_content = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        host_path, container_path = self._save_file_to_docker_workspace(
            json_content,
            tool_name,
            suffix='.json',
            doc_ids=doc_ids
        )
        
        # Generate summary
        summary = self._generate_response_summary(result, tool_name)
        
        # Return formatted result with jq/grep instructions
        formatted_result = f"""Large structured response saved as JSON file.

Document Summary:
{summary}

JSON FILE: {container_path}

FILE ACCESS:
- Use jq via bash tool for structured queries (e.g., jq '.field' {container_path})
- Use grep for keyword search (e.g., grep "keyword" {container_path})
- Use text_editor or cat to view the file

Examples:
  jq '.[] | select(.age > 30)' {container_path}  # Filter records
  jq '.user.profile.name' {container_path}       # Navigate nested data
  grep -i "alice" {container_path}                # Text search
"""
        
        data_sources.append(f"JSON file: {container_path}")
        
        logger.info(f"Stored structured response as JSON: {container_path}")
        
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": formatted_result}

    def _handle_hybrid_text_and_rag_storage(
        self,
        result: Any,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: str,
        data_sources: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Store large unstructured response with RAG indexing.
        For Q&A workflow: Wait for RAG completion before returning.
        """
        # Convert result to content string
        if self._is_doc_content_tool(tool_name):
            content = self._preprocess_doc_response(result, tool_name)
        else:
            content = str(result) if not isinstance(result, str) else result

        import uuid

        # STEP 1: Save text file to Docker workspace (synchronous)
        suffix = '.html' if self._is_doc_content_tool(tool_name) else '.txt'
        host_path, container_path = self._save_file_to_docker_workspace(
            content,
            tool_name,
            suffix=suffix,
            doc_ids=doc_ids
        )

        document_id = str(uuid.uuid4())

        # STEP 2: Start RAG indexing and WAIT for completion
        job_id = None
        indexing_status = "disabled"

        if self._should_use_rag_for_content(content):
            rag_storage = self._get_rag_storage_tool()
            if rag_storage and self.background_indexing_manager:
                try:
                    # Submit indexing job
                    job_id = self.background_indexing_manager.submit_indexing_job(
                        content=content,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        document_id=document_id,
                        rag_storage=rag_storage
                    )
                    logger.info(f"RAG indexing started: job_id={job_id}")

                    # WAIT for completion with timeout
                    import time
                    max_wait = 45  # seconds
                    check_interval = 1  # second
                    elapsed = 0

                    while elapsed < max_wait:
                        status = self.background_indexing_manager.get_job_status(job_id)

                        if status == "completed":
                            logger.info(f"RAG indexing completed in {elapsed}s")
                            indexing_status = "completed"
                            break
                        elif status == "failed":
                            logger.error("RAG indexing failed")
                            indexing_status = "failed"
                            break

                        time.sleep(check_interval)
                        elapsed += check_interval

                    if elapsed >= max_wait and indexing_status not in ["completed", "failed"]:
                        logger.warning(f"RAG indexing timeout after {max_wait}s")
                        indexing_status = "timeout"

                except Exception as e:
                    logger.error(f"Failed to start/wait for RAG indexing: {e}")
                    indexing_status = "failed"

        # STEP 3: Generate summary
        summary = self._generate_text_summary(content)

        # STEP 4: Format response based on indexing status
        if indexing_status == "completed":
            formatted_result = f"""Large document stored and indexed successfully.

Document Summary:
{summary}

FILE PATH: {container_path}
DOCUMENT ID: {document_id}

RAG INDEXING: ✓ Complete
Use search_tool_response_in_vectorstore(query='your question', document_id='{document_id}') to find specific information.

For summarization requests, use: summarize_doc(docId='...', file_path='{container_path}')
"""
        else:
            # Fallback message if RAG failed/timeout
            formatted_result = f"""Large document stored (RAG indexing {indexing_status}).

Document Summary:
{summary}

FILE PATH: {container_path}
DOCUMENT ID: {document_id}

FALLBACK ACCESS (RAG unavailable):
- Use grep for text search: grep -i "keyword" {container_path}
- Use text_editor to read sections
- For summarization: summarize_doc(docId='...', file_path='{container_path}')
"""

        # Track data source
        data_sources.append(f"Text file: {container_path}")
        if indexing_status == "completed":
            data_sources.append(f"RAG indexed: {document_id}")

        logger.info(f"Document processing complete: file={container_path}, rag_status={indexing_status}")

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": formatted_result
        }

    def _should_use_rag_for_content(self, content: str) -> bool:
        """Check if content should be indexed in RAG based on config"""
        from leanworks.setting import LARGE_RESPONSE_CONFIG
        config = LARGE_RESPONSE_CONFIG
        return (
            config.get("use_rag_for_unstructured", True) and
            len(content) >= config.get("rag_min_semantic_value", 1000)
        )

    def _get_rag_storage_tool(self):
        """Get RAG storage tool for large responses"""
        # If we have a large response vectordb client, use it for large response RAG storage
        if self.large_response_vectordb_client and hasattr(self, 'tool_use') and self.tool_use:
            # Get embedding client from tool_use
            embedding_client = getattr(self.tool_use, 'embedding_client', None)
            if embedding_client:
                from leanworks.agent.tools.rag_storage import RAGStorageTool
                from leanworks.setting import LARGE_RESPONSE_CONFIG
                
                config = LARGE_RESPONSE_CONFIG
                large_response_config = config.get("large_response_indexes", {})
                
                # Create RAG storage tool with large response indexes
                rag_storage = RAGStorageTool(
                    vectordb_client=self.large_response_vectordb_client,
                    embedding_client=embedding_client,
                    org_slug=self.org_slug,
                    chunk_size=config.get("rag_chunk_size", 512),
                    chunk_overlap=config.get("rag_chunk_overlap", 128),
                    use_large_response_indexes=True,
                    large_response_dense_index=self.large_response_vectordb_client.dense_index,
                    large_response_sparse_index=self.large_response_vectordb_client.sparse_index
                )
                logger.info(f"Using large response RAG storage with namespace: {rag_storage.namespace}")
                return rag_storage
        
        # Fallback: Check if tool_use has access to rag_storage_tool
        if hasattr(self, 'tool_use') and self.tool_use:
            return getattr(self.tool_use, 'rag_storage_tool', None)
        return None

    def _format_hybrid_storage_summary(
        self,
        container_path: str,
        host_path: str,
        document_id: str,
        job_id: str,
        indexing_status: str,
        summary: str
    ) -> str:
        """Format summary for hybrid text + RAG storage"""

        # Build indexing status message
        if indexing_status == "in_progress":
            indexing_msg = f"Status: Background indexing in progress (job_id: {job_id})"
            rag_note = "Semantic search available once background indexing completes."
        elif indexing_status == "failed":
            indexing_msg = "Status: Background indexing failed"
            rag_note = "Only text search available."
        else:
            indexing_msg = "Status: Background indexing disabled"
            rag_note = "Only text search available."

        return f"""Large unstructured text saved to: {container_path}
- Size: {summary.split('characters,')[0].strip()} characters
- Lines: {summary.split('words')[0].split(',')[1].strip() if ',' in summary else 'N/A'}

IMMEDIATE ACCESS:
- Use grep and text_editor (see <core_tools_reference> in system prompt)
- File path: {container_path}

SEMANTIC SEARCH (background RAG indexing):
- Document ID: {document_id}
- {indexing_msg}
- Once ready, use: search_tool_response_in_vectorstore(query='...', document_id='{document_id}')

NOTE: {rag_note}
"""

    def _handle_large_unstructured_response(
        self,
        content: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: str,
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Store large unstructured response as text file and return summary"""
        
        # Use text file storage instead of RAG
        return self._handle_large_unstructured_response_as_file(
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            data_sources=data_sources
        )
    
    def _handle_large_unstructured_response_as_file(
        self,
        content: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: str,
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """
        Save large unstructured response as text file (similar to doc responses).
        
        Args:
            content: The unstructured text content to save
            tool_name: Name of the tool that generated this
            tool_input: Input parameters to the tool
            tool_use_id: Tool use ID for tracking
            data_sources: List to append data source info
            
        Returns:
            Formatted result with file path for text editor tool access
        """
        try:
            # Create temporary file with appropriate naming
            # Use tool name (sanitized) and short identifier
            tool_name_safe = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name)[:30]  # Sanitize and limit length
            fd, file_path = tempfile.mkstemp(
                suffix='.txt',
                prefix=f'tool_response_{tool_name_safe}_',
                text=True
            )
            
            try:
                # Write content to file
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Generate a short summary for the response
                summary = self._generate_text_summary(content)
                
                # Format response similar to doc response format
                summary_lines = [
                    "Large unstructured response stored in temporary text file.",
                    "",
                    f"File saved: {file_path}",
                    "",
                    f"Summary: {summary}",
                    "",
                    f'Use text editor tool with path="{file_path}" to view.',
                    "For large files, use grep (via bash tool) to locate specific sections before viewing."
                ]
                
                formatted_result = "\n".join(summary_lines)
                
                # Track data source
                data_sources.append(f"Text file: {file_path}")
                
                logger.info(f"Stored large unstructured response in text file: {file_path}")
                
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": formatted_result
                }
            except Exception as e:
                # Close file descriptor if write failed
                os.close(fd)
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise
        except Exception as e:
            logger.error(f"Failed to store large unstructured response in text file: {e}")
            # Fallback to truncation
            return self._truncate_response(content, tool_use_id)
    
    
    def _get_table_name_for_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Generate appropriate table name based on tool and query"""
        import re
        
        # PostgreSQL queries
        if tool_name == "query_postgres":
            sql = tool_input.get("sql", "").upper() if isinstance(tool_input, dict) else ""
            if sql and "FROM" in sql:
                # Try to extract table name from SQL
                match = re.search(r'FROM\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
            return "query_results"
        
        # Search results
        if tool_name == "search_documents":
            return "search_results"
        
        # GitHub/Jira issues
        if tool_name in ["github_search_issues", "search_issues"]:
            return "issues"
        
        # Default: use tool name
        return tool_name.replace("_", "_") + "_results"

    def _get_session_temp_dir(self) -> str:
        """Get the session-specific temp directory that's mounted in Docker container"""
        if hasattr(self.tool_use, '_bash_session') and self.tool_use._bash_session:
            return self.tool_use._bash_session.session_temp_dir
        # Fallback: create if container not initialized yet
        import tempfile
        import os
        session_temp_dir = os.path.join(tempfile.gettempdir(), f"session_{self.session_id}")
        os.makedirs(session_temp_dir, exist_ok=True)
        return session_temp_dir

    def _save_file_to_docker_workspace(self, content: str, tool_name: str, suffix: str = '.txt', doc_ids: Optional[List[str]] = None) -> tuple[str, str]:
        """
        Save file to Docker workspace directory and register in working context.

        Ensures Docker container is initialized before saving to guarantee
        consistent session directory paths.

        Returns:
            tuple: (host_path, container_path)
        """
        import uuid
        import re
        import os

        # Ensure Docker container is initialized before saving files
        if not self._ensure_docker_container_initialized():
            logger.error("Failed to initialize Docker container for file save operation")
            raise Exception("Docker container initialization failed - cannot save file to workspace")

        session_temp_dir = self._get_session_temp_dir()
        tool_name_safe = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name)[:30]

        # Create unique filename
        filename = f'tool_response_{tool_name_safe}_{uuid.uuid4().hex[:8]}{suffix}'
        host_path = os.path.join(session_temp_dir, filename)

        # Write file on host
        with open(host_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Register in working context if available
        if self.memory_manager and hasattr(self.memory_manager, 'working_context'):
            resource_id = f"tool_response_file_{uuid.uuid4().hex[:8]}"
            self.memory_manager.working_context.register_resource(
                resource_id=resource_id,
                type='tool_response_file',
                path=f'/workspace/{filename}',
                metadata={
                    'tool': tool_name,
                    'host_path': host_path,
                    'file_type': suffix,
                    'created_via': 'unified_response_handler',
                    'doc_ids': doc_ids if doc_ids else []
                }
            )

        # Return both host path and container path
        container_path = f'/workspace/{filename}'
        return host_path, container_path

    def _generate_response_summary(self, result: Any, tool_name: str) -> Dict[str, Any]:
        """Generate summary statistics for large responses"""
        summary = {}
        
        if isinstance(result, list):
            summary = {
                "type": "list",
                "count": len(result),
                "total_items": len(result)
            }
            
            # If list of dicts, add column info
            if result and isinstance(result[0], dict):
                summary["columns"] = list(result[0].keys())
                summary["sample_keys"] = list(result[0].keys())[:10]
                # Sample items
                from leanworks.setting import LARGE_RESPONSE_CONFIG
                sample_size = LARGE_RESPONSE_CONFIG.get("summary_sample_size", 3)
                summary["sample_items"] = result[:sample_size] if len(result) > sample_size else result
        
        elif isinstance(result, dict):
            summary = {
                "type": "dict",
                "keys": list(result.keys()),
                "key_count": len(result.keys())
            }
            
            # If dict contains lists, summarize those too
            for key, value in result.items():
                if isinstance(value, list):
                    summary[f"{key}_count"] = len(value)
        
        elif isinstance(result, str):
            summary = {
                "type": "string",
                "length": len(result)
            }
        
        return summary
    
    def _generate_text_summary(self, text: str) -> str:
        """Generate summary of long text"""
        lines = text.split('\n')
        first_paragraph = lines[0][:200] if lines else text[:200]
        
        word_count = len(text.split())
        char_count = len(text)
        paragraph_count = len([l for l in lines if l.strip()])
        
        return f"""- Length: {char_count:,} characters, {word_count:,} words
- Paragraphs: {paragraph_count}
- Preview: {first_paragraph}..."""
    
    def _format_large_structured_summary(
        self,
        summary: Dict[str, Any],
        response_id: str,
        table_name: str,
        tool_name: str
    ) -> str:
        """Format summary with DuckDB storage information"""
        
        if summary.get("type") == "list":
            count = summary.get("count", 0)
            columns = summary.get("columns", [])
            sample_items = summary.get("sample_items", [])
            
            columns_str = ', '.join(columns[:10]) if columns else "N/A"
            if len(columns) > 10:
                columns_str += "..."
            
            sample_str = ""
            if sample_items:
                import json
                sample_str = "\n\nSample items:\n" + "\n".join(
                    json.dumps(item, default=str) for item in sample_items[:3]
                )
            
            return f"""Large response stored in DuckDB database.

Summary:
- Total items: {count}
- Columns: {columns_str}{sample_str}

The full data has been saved to DuckDB response database '{response_id}' in table '{table_name}'.

To query this data:
1. Use get_response_schema tool with response_id='{response_id}' to see the schema
2. Use query_response_duckdb tool with response_id='{response_id}' and your SQL query

Example query:
query_response_duckdb(response_id='{response_id}', sql='SELECT * FROM {table_name} LIMIT 10')
"""
        
        elif summary.get("type") == "dict":
            keys = summary.get("keys", [])
            keys_str = ', '.join(keys[:10])
            if len(keys) > 10:
                keys_str += f"... ({len(keys)} total keys)"
            
            return f"""Large response stored in DuckDB database.

Summary:
- Type: Dictionary
- Keys: {keys_str}

The full data has been saved to DuckDB response database '{response_id}' in table '{table_name}'.

To query this data:
1. Use get_response_schema tool with response_id='{response_id}' to see the schema
2. Use query_response_duckdb tool with response_id='{response_id}' and your SQL query
"""
        
        else:
            return f"""Large response stored in DuckDB database.

Summary:
- Type: {summary.get('type', 'unknown')}
- Length: {summary.get('length', 'N/A')}

The full data has been saved to DuckDB response database '{response_id}' in table '{table_name}'.

To query this data:
1. Use get_response_schema tool with response_id='{response_id}' to see the schema
2. Use query_response_duckdb tool with response_id='{response_id}' and your SQL query
"""
    
    def _format_large_unstructured_summary(
        self,
        summary: str,
        document_id: str,
        tool_name: str,
        content: str
    ) -> str:
        """Format summary with RAG storage information"""
        from leanworks.setting import LARGE_RESPONSE_CONFIG
        preview_length = LARGE_RESPONSE_CONFIG.get("summary_preview_length", 500)
        
        return f"""Large text response stored in RAG vector database.

Summary:
{summary}

The full content has been stored and can be retrieved using semantic search.
Document ID: {document_id}

To retrieve relevant information from this stored response:
- Use search_documents tool with a query related to what you need
- The stored content will be automatically included in search results
- You can also ask follow-up questions and the system will retrieve relevant parts

Preview (first {preview_length} chars):
{content[:preview_length]}...
"""
    
    def _truncate_response(self, result: Any, tool_use_id: str) -> Dict[str, Any]:
        """Fallback: truncate response if storage fails"""
        if isinstance(result, str):
            truncated = result[:2000] + "..." if len(result) > 2000 else result
        else:
            import json
            json_str = json.dumps(result, default=str)
            truncated = json_str[:2000] + "..." if len(json_str) > 2000 else json_str
        
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"[Response truncated due to size. First 2000 chars: {truncated}]"
        }

    def _get_result_preview(self, result):
        """
        Generate a preview of the tool call result for logging purposes.
        
        Args:
            result: The result from a tool call
            
        Returns:
            str: A preview string of the result
        """
        try:
            if isinstance(result, list):
                if len(result) == 0:
                    return "Empty list"
                elif len(result) == 1:
                    return f"List with 1 item: {self._truncate_preview(str(result[0]))}"
                else:
                    return f"List with {len(result)} items: {self._truncate_preview(str(result[0]))} ..."
            elif isinstance(result, dict):
                if len(result) == 0:
                    return "Empty dict"
                else:
                    # Get first key-value pair for preview
                    first_key = next(iter(result))
                    first_value = self._truncate_preview(str(result[first_key]))
                    return f"Dict with {len(result)} keys: {first_key}={first_value} ..."
            elif isinstance(result, str):
                return self._truncate_preview(result)
            else:
                return self._truncate_preview(str(result))
        except Exception as e:
            return f"Error generating preview: {str(e)}"
    
    def _truncate_preview(self, text, max_length=200):
        """
        Truncate text to a maximum length for preview purposes.
        
        Args:
            text: The text to truncate
            max_length: Maximum length for the preview
            
        Returns:
            str: Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length] + "..."
    
    def _extract_table_names_from_sql(self, sql: str) -> List[str]:
        """
        Extract table names from SQL SELECT/WITH queries.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of table names found in the query
        """
        if not sql or not isinstance(sql, str):
            return []
        
        table_names = set()
        
        # Remove comments and normalize whitespace
        sql_clean = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = ' '.join(sql_clean.split())
        
        # Pattern to match FROM and JOIN clauses
        # Match: FROM table_name, FROM schema.table_name, FROM "table_name"
        # Match: JOIN table_name, JOIN schema.table_name, JOIN "table_name"
        patterns = [
            r'\bFROM\s+["\']?(\w+)["\']?',  # FROM table
            r'\bJOIN\s+["\']?(\w+)["\']?',  # JOIN table
            r'\bINTO\s+["\']?(\w+)["\']?',  # INTO table (for CTEs)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sql_clean, re.IGNORECASE)
            for match in matches:
                table_name = match.group(1).lower()
                # Filter out common SQL keywords that might be matched
                if table_name not in ['select', 'where', 'group', 'order', 'having', 'limit', 'offset', 'as']:
                    table_names.add(table_name)
        
        # Also check for CTE names in WITH clauses
        with_pattern = r'\bWITH\s+(\w+)\s+AS'
        with_matches = re.finditer(with_pattern, sql_clean, re.IGNORECASE)
        for match in with_matches:
            cte_name = match.group(1).lower()
            # CTE names are temporary, but we can still track them
            # For now, we'll skip CTE names as they're not actual tables
        
        return sorted(list(table_names))
    
    def extract_json_from_text(self, text):
        """
        Extract and parse JSON from text content.
        
        Args:
            text (str): Text that may contain JSON.
            
        Returns:
            dict: Parsed JSON as a dictionary, or empty dict if parsing fails.
        """    
        # Return empty dict for empty or None input
        if not text:
            return {}
        
        if "{" not in text and "}" not in text:
            return {"content": text, "answered": "false"}
        
        
        try:
            # First try to parse the entire text as JSON
            parsed_json = json.loads(text)         
            return parsed_json
        except json.JSONDecodeError:

            # Look for text that appears to be JSON (between curly braces)
            # Use non-greedy matching to find the outermost JSON object
            try:
                # Find the first JSON object in the text, regardless of surrounding content
                json_match = re.search(r'{.*?}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
            except Exception as e:
                # If regex extraction fails, use Claude to extract the JSON
                if self.model_client:
                    prompt = f'''
                    Extract the content from below JSON. The response should be exactly the same as the content in the JSON:
                    ###JSON
                    {text}
                    ###
                    Output ONLY the extracted content text—no explanations, no reasoning, no headings.
                    '''
                    try:
                        # Use Claude API to extract JSON using JSON mode
                        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                        response = self.model_client.messages.create(
                            model="claude-3-haiku-20240307",
                            messages=messages,
                            max_tokens=1000,
                            temperature=0.0
                        )
                        if "true" in text.lower():
                            return {"content": response.content[0].text, "answered": "true"}
                        else:
                            return {"content": response.content[0].text, "answered": "false"}
                    except Exception as e:
                        print(f"Claude JSON extraction failed: {e}")
                        
            return {"content": text, "answered": "false"}

    def add_assistant_message_with_tool_uses(self, response):
        """Add an assistant message including text content, tool use blocks, and tool result blocks
        
        This method is specifically for adding a Claude response that contains tool use blocks,
        converting them to a JSON-serializable format. For server tools, Anthropic may include
        tool_result blocks in the response automatically.
        """
        # Create a serializable representation of the content
        serializable_content = []
        
        for block in response.content:
            if block.type == "text":
                serializable_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                # Extract tool use details
                try:
                    tool_use_id = block.id  # For newer versions of Claude API
                    tool_name = block.name  # For newer versions of Claude API 
                    tool_input = block.input
                except AttributeError:
                    tool_use_id = block.tool_use.id  # For Claude API 3.5
                    tool_name = block.tool_use.name  # For Claude API 3.5
                    tool_input = block.tool_use.input
                
                # Create a serializable representation of the tool_use block
                serializable_content.append({
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": tool_input
                })
            elif block.type == "tool_result":
                # For server tools, Anthropic may include tool_result blocks in the response
                # We should include these in the conversation so Claude can see the results
                try:
                    tool_use_id = block.id  # For newer versions of Claude API
                    content = block.content  # Can be string or list
                    is_error = getattr(block, 'is_error', False)
                except AttributeError:
                    tool_use_id = block.tool_result.id  # For Claude API 3.5
                    content = block.tool_result.content
                    is_error = getattr(block.tool_result, 'is_error', False)
                
                # Create a serializable representation of the tool_result block
                tool_result_dict = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content
                }
                if is_error:
                    tool_result_dict["is_error"] = True
                serializable_content.append(tool_result_dict)
        
        # Add the assistant message with the serializable content
        self.conversation.append({
            "role": "assistant",
            "content": serializable_content
        })

    def _is_doc_content_tool(self, tool_name: str) -> bool:
        """Check if tool returns document content that should be converted to HTML"""
        DOC_CONTENT_TOOLS = {
            'get_doc',
            'search_docs',
            # list_docs returns previews only, so exclude it
        }
        return tool_name in DOC_CONTENT_TOOLS

    def _preprocess_doc_response(self, result: Any, tool_name: str) -> str:
        """
        Convert doc management tool responses to clean HTML format.

        Extracts document content and metadata from JSON structure and
        formats as readable HTML for better storage and RAG indexing.

        Args:
            result: Doc tool response (List[Dict] or error Dict)
            tool_name: Name of the doc tool

        Returns:
            HTML formatted string with document(s) content
        """
        # Handle error responses - pass through unchanged
        if isinstance(result, dict) and 'error' in result:
            return str(result)  # Convert to string for storage

        # Must be a list of documents
        if not isinstance(result, list):
            logger.warning(f"Unexpected doc response type: {type(result)}, returning as-is")
            return str(result)

        if not result:
            return "No documents found."

        html_parts = []

        for doc in result:
            if not isinstance(doc, dict):
                continue

            doc_id = doc.get('id', 'Unknown')
            title = doc.get('title', 'Untitled Document')
            content = doc.get('content', '')
            owner_email = doc.get('owner_email', 'Unknown')
            created_at = doc.get('created_at', 'Unknown')
            updated_at = doc.get('updated_at', 'Unknown')

            # Format metadata as HTML comments
            metadata = f"""<!-- Document: {title} -->
<!-- ID: {doc_id} -->
<!-- Owner: {owner_email} -->
<!-- Created: {created_at} -->
<!-- Updated: {updated_at} -->

"""

            # Add title as H1 if content doesn't start with HTML
            if content and not content.strip().startswith('<'):
                content = f"<h1>{title}</h1>\n\n{content}"

            # Format document
            doc_html = f"{metadata}{content}\n\n<!-- End of Document: {title} -->\n\n"

            html_parts.append(doc_html)

        # Join multiple documents with separator
        separator = "=" * 80 + "\n\n"
        return separator.join(html_parts)

    # Helper function to create a modified copy of parameters
    def create_params_copy(self, params, **modifications):
        params_copy = copy.deepcopy(params)

        # Make sure required parameters are present in the copy
        required_params = ["model", "max_tokens", "messages"]
        for param in required_params:
            if param not in params_copy and param not in modifications:
                raise ValueError(f"Missing required parameter: {param}")

        # Apply modifications
        for key, value in modifications.items():
            if value is None and key in params_copy:
                # Don't remove required parameters even if explicitly set to None
                if key not in required_params:
                    del params_copy[key]
            elif value is not None:
                params_copy[key] = value

        return params_copy