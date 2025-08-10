from leanworks.agent.tools.toolkit import ToolUse
from datetime import datetime, timezone
from leanworks.agent.conversation import ConversationManager
from leanworks.agent.memory import MemoryManager
from leanworks.setting import AGENT_SYSTEM_PROMPT, SEARCH_KNOWLEDGE_QUERY, EVALUATION_PROMPT, CRITIQUE_MESSAGE, GENERATION_MODEL
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
                 storage_client,
                 secret_client,
                 model_client,
                 bq_client_wrapper,
                 user_id=None,
                 session_id=None,
                 clear_conversation=True,
                 tools=None
                 ):
        """
        Initialize the ChatAgent with necessary clients and settings.
        
        Args:
            storage_client: The storage client for GCS operations
            secret_client: The secret client for accessing secrets
            model_client: The Claude model client for main chat
            bq_client_wrapper: The BigQuery client object that contains dataset_id
            user_id (str): The user ID for conversation tracking
            session_id (str): The session ID for conversation tracking
            clear_conversation (bool): Whether to clear conversation history on init
            tools (list): List of additional tools to enable. These will be added to the default tools ['leanworks', 'search']. ToolUse handles the processing and filtering.
        """
        # Initialize clients
        self.storage_client = storage_client
        self.secret_client = secret_client
        self.model_client = model_client
        self.bq_client_wrapper = bq_client_wrapper
        
        # Set parameters
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize data source tracking
        self.data_sources = []
        
        # Initialize document ID tracking for aggressive deduplication
        self.read_document_ids = set()
        
        # Initialize tool use with BigQuery client and tools (ToolUse handles tool processing and credential retrieval)
        self.tool_use = ToolUse(bq_client_wrapper, storage_client, secret_client, self.read_document_ids, tools=tools)
        

        
        # Initialize memory management (always enabled)
        try:
            # Use model-aware factory method for optimal defaults
            self.memory_manager = MemoryManager.create_for_model(
                model_name=GENERATION_MODEL,
                model_client=model_client,
                storage_client=storage_client,
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
            self.storage_client, 
            self.user_id, 
            self.session_id
        )
        if clear_conversation:
            self.conversation.clear_conversation()
            # Also clear read document IDs when clearing conversation
            self.read_document_ids.clear()
        else:
            # When clear_conversation is False, clear memory (inverted logic)
            if self.memory_manager:
                self.memory_manager.clear_memory()
        
        # Get user info from BigQuery
        user_info = self._get_user_info()
        user_timezone = user_info.get("timezone", "UTC")
        user_timezone = pytz.timezone(user_timezone)
        # Set up API parameters for main model
        self.system_prompt = AGENT_SYSTEM_PROMPT.format(
            USER_INFO=user_info, 
            CURRENT_DATE_UTC=datetime.now(timezone.utc).isoformat(),
            CURRENT_DATE_LOCAL=datetime.now(user_timezone).isoformat()
        )
        
        # Set the system prompt and user profile for memory manager
        if self.memory_manager:
            self.memory_manager.set_system_prompt(self.system_prompt)
            self.memory_manager.set_user_profile(str(user_info))
        
        self.api_params = {
            "model": GENERATION_MODEL,
            "system": self.system_prompt,
            "messages": self.conversation.conversation,
            "tools": self.tool_use.tools,
            "max_tokens": 2048,
            "temperature": 0.1,
            "timeout": 30
        }

    def _get_user_info(self):
        """
        Query user information from BigQuery user_config table.
        
        Returns:
            dict: User information dictionary with user_id, alias_email, first_name, last_name
        """
        try:
            if not self.user_id:
                return {"user_id": "Unknown", "alias_email": "", "first_name": "", "last_name": ""}
            
            query = f"""
            SELECT user_id, alias_email, first_name, last_name, timezone 
            FROM `leanworks.{self.bq_client_wrapper.client_name}.user_config` 
            WHERE user_id = '{self.user_id}'
            """
            
            query_job = self.bq_client_wrapper.bq_client.query(query)
            results = query_job.result()
            
            # Convert results to dict
            for row in results:
                user_info = {
                    "user_id": row.user_id or "",
                    "alias_email": row.alias_email or "",
                    "first_name": row.first_name or "",
                    "last_name": row.last_name or "",
                    "timezone": row.timezone or "UTC"
                }
                logger.info(f"Retrieved user info: {user_info}")
                return user_info
            
            # If no results found, return default dict
            logger.warning(f"No user info found for user_id: {self.user_id}")
            return {"user_id": self.user_id, "alias_email": "", "first_name": "", "last_name": ""}
            
        except Exception as e:
            logger.error(f"Error retrieving user info from BigQuery: {str(e)}")
            # Return default dict on error
            return {"user_id": self.user_id or "Unknown", "alias_email": "", "first_name": "", "last_name": ""}


    def process_message(self, user_message, cited_context=None, deep_research=False):
        """
        Process a user message and handle the conversation flow.
        
        Args:
            user_message (str): The user's message content
            cited_context (str): The cited context for the user message
        Returns:
            dict: Dictionary with 'content' (response text) and 'data_sources' (list of sources)
        """
        # Reset data sources for new message
        self.data_sources = []
        
        # Store the original user query for evaluation (before adding cited context)
        self.original_user_query = user_message
        logger.info(f"Stored original user query for evaluation: {self.original_user_query}")
        
        # Log current state of document deduplication
        logger.info(f"Processing message with {len(self.read_document_ids)} documents already read for deduplication")
        
        # Prepare user message
        if cited_context:
            user_message = f"<cited_context>{cited_context}</cited_context>\n{user_message}"
        
        # Create user message object
        user_message_obj = {
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        }
        
        # Add to memory manager if enabled
        if self.memory_manager:
            self.memory_manager.add_turn(user_message_obj)
            logger.info(f"Added user message to memory manager. Stats: {self.memory_manager.get_memory_stats()}")
        
        # Add the user message to conversation
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
        max_unanswered_num = 2 if not deep_research else 5
        
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
                response = self.model_client.messages.create(**current_params)
                text_content = next((block.text for block in response.content if block.type == "text"), "")
                has_tool_calls = any(block.type == "tool_use" for block in response.content)   
                
                if has_tool_calls:
                    # Add the assistant message with tool_use blocks to the conversation
                    self.conversation.add_assistant_message_with_tool_uses(response)
                    
                    # Process tool calls and add results to conversation with data source tracking
                    tool_results = self.conversation.parse_and_format_tool_results_with_sources(
                        response, 
                        self.tool_use.function_map,
                        self.data_sources
                    )
                    self.conversation.add_tool_results(tool_results)
                else:
                    # Assign the actual response text
                    response_text = text_content
                    
                    # Extract source content for evaluation
                    source_content = self._extract_source_content_from_conversation()
                    eval_feedback = self._perform_evaluation(response_text, source_content)
                    
                    eval_score = eval_feedback.get("score", 0)
                    eval_explanation = eval_feedback.get("explanation", "No explanation provided")
                    
                    if eval_score >= 7:
                        logger.info(f"Question answered with score {eval_score}.")
                        
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
                            tool_results = self.conversation.parse_and_format_tool_results_with_sources(
                                retry_response, 
                                self.tool_use.function_map,
                                self.data_sources
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

                    # Find the search_knowledge tool specifically
                    search_knowledge_tool = None
                    for tool in self.tool_use.tools:
                        if tool.get("name") == "search_knowledge":
                            search_knowledge_tool = tool
                            break
                    
                    if not search_knowledge_tool:
                        logger.error("search_knowledge tool not found in available tools")
                        response_text = "Error: search_knowledge tool not available"
                        break
                    
                    # Create and immediately use forced search_knowledge parameters
                    forced_params = self.conversation.create_params_copy(
                        self.api_params,
                        messages=self.conversation.conversation,
                        tools=[search_knowledge_tool],
                        tool_choice={"type": "tool", "name": "search_knowledge"}
                    )
                    
                    # Make API call with forced search_knowledge
                    response = self.model_client.messages.create(**forced_params)
                    
                    # Process the forced tool call
                    self.conversation.add_assistant_message_with_tool_uses(response)
                    tool_results = self.conversation.parse_and_format_tool_results_with_sources(
                        response, 
                        self.tool_use.function_map,
                        self.data_sources
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

        # Always add last response to slim_conversation
        self.conversation.add_assistant_message(response_text, include_in_slim=True)
        self.conversation.save_conversation()
        logger.info(f"Final answer: {response_text}")
        logger.info(f"Data sources used: {unique_sources}")
        logger.info(f"Session now has {len(self.read_document_ids)} total documents read (deduplicated)")
        
        # Log memory stats if using memory manager
        if self.memory_manager:
            logger.info(f"Memory stats: {self.memory_manager.get_memory_stats()}")
        
        # Return dictionary with content and data sources
        return {
            "content": response_text,
            "data_sources": unique_sources
        }
    
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
                    LAST_RESPONSE=response_text,
                    SOURCE_CONTEXT=source_content
                )}]
            })
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