from leanworks.agent.tools.toolkit import ToolUse
from datetime import datetime
from leanworks.agent.conversation import ConversationManager
from leanworks.setting import AGENT_SYSTEM_PROMPT, SEARCH_KNOWLEDGE_QUERY, EVALUATION_PROMPT, GENERATION_MODEL
import traceback
import logging

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
                 gitlab_client,
                 user_id=None,
                 session_id=None,
                 clear_conversation=True
                 ):
        """
        Initialize the ChatAgent with necessary clients and settings.
        
        Args:
            storage_client: The storage client for GCS operations
            secret_client: The secret client for accessing secrets
            model_client: The Claude model client for main chat
            bq_client_wrapper: The BigQuery client object that contains dataset_id
            gitlab_client: The GitLab client object that contains gitlab_url and access_token
            user_id (str): The user ID for conversation tracking
            session_id (str): The session ID for conversation tracking
            max_unanswered_num (int): Maximum number of attempts to answer a question
            clear_conversation (bool): Whether to clear conversation history on init
        """
        # Initialize clients
        self.storage_client = storage_client
        self.secret_client = secret_client
        self.model_client = model_client
        self.bq_client_wrapper = bq_client_wrapper
        self.gitlab_client = gitlab_client
        # Set parameters
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize data source tracking
        self.data_sources = []
        
        # Initialize document ID tracking for aggressive deduplication
        self.read_document_ids = set()
        
        # Initialize tool use with BigQuery client
        self.tool_use = ToolUse(bq_client_wrapper, storage_client, secret_client, self.read_document_ids, self.gitlab_client)
        
        # Verify that the reference is maintained
        logger.info(f"ChatAgent read_document_ids id: {id(self.read_document_ids)}")
        logger.info(f"ToolUse read_document_ids id: {id(self.tool_use.read_document_ids)}")
        logger.info(f"References are same: {self.read_document_ids is self.tool_use.read_document_ids}")
        
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
        
        # Get user info from BigQuery
        user_info = self._get_user_info()
        
        # Set up API parameters for main model
        self.system_prompt = AGENT_SYSTEM_PROMPT.format(
            USER_INFO=user_info, 
            CURRENT_DATE=datetime.now().strftime("%Y-%m-%d")
        )
        
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
            SELECT user_id, alias_email, first_name, last_name 
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
                    "last_name": row.last_name or ""
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

    def reset_read_documents(self):
        """
        Reset the set of read document IDs for a fresh start.
        Useful when starting a new conversation or topic.
        """
        logger.info(f"Resetting read document IDs (previously had {len(self.read_document_ids)} documents)")
        self.read_document_ids.clear()

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
        
        # Add the user message
        if cited_context:
            user_message = f"<cited_context>{cited_context}</cited_context>\n{user_message}"
        self.conversation.add_user_message(user_message, include_in_slim=True)
        
        # Maximum number of iterations to prevent infinite loops
        unanswered_count = 0
        response_text = ""
        max_unanswered_num = 2 if not deep_research else 5

        while unanswered_count < max_unanswered_num:
            logger.info(f"Unanswered attempt {unanswered_count}")
            try:
                # Create a copy with updated messages
                current_params = self.conversation.create_params_copy(
                    self.api_params, 
                    messages=self.conversation.conversation
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
                    eval_feedback = self._perform_evaluation(response_text)

                    # Add null check for eval_feedback
                    if eval_feedback and eval_feedback.get("score", 0) >= 7:
                        logger.info("Question answered.")
                        # Add to regular conversation first
                        self.conversation.add_assistant_message(response_text)
                        break
                    # No tool calls were made and the answer is still not complete
                    logger.info("No tool calls were made and the answer is still not complete")
                    unanswered_count += 1

                    # Now response_text has the actual content for the search query
                    self.conversation.add_user_message(SEARCH_KNOWLEDGE_QUERY.format(
                        USER_QUERY=self.original_user_query, 
                        LAST_RESPONSE=response_text, 
                        EVALUATION_FEEDBACK=eval_feedback or "No feedback available"
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
                            
                            # Combine source attribution with content
                            attributed_content = f"{data_source}{tool_content}"
                            source_content_with_attribution.append(attributed_content)
        
        return "\n\n---\n\n".join(source_content_with_attribution) if source_content_with_attribution else "No source content available."

    def _perform_evaluation(self, response_text):
        """
        Performs evaluation on the response using a separate evaluation LLM.
        Returns evaluation feedback including potential search queries.
        
        Args:
            response_text (str): The response text to evaluate
            
        Returns:
            dict: Evaluation feedback with search queries, or None if evaluation failed
        """
        try:
            # Use the original user query stored when processing the message
            if not hasattr(self, 'original_user_query') or not self.original_user_query or not response_text:
                logger.warning("Missing original user query or response text for evaluation")
                return None
            
            logger.info(f"Using stored original user query for evaluation: {self.original_user_query}")
            
            # Extract actual source content from tool results in conversation
            sources_content = self._extract_source_content_from_conversation()
            
            eval_conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": EVALUATION_PROMPT.format(USER_QUERY=self.original_user_query, LAST_RESPONSE=response_text, SOURCES=sources_content)}]
                }
            ]
            
            # Create evaluation parameters
            eval_params = {
                "model": GENERATION_MODEL,
                "messages": eval_conversation,
                "max_tokens": 512,
                "temperature": 0.1,
                "timeout": 30
            }
            
            # Make evaluation API call using separate eval model client
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
                return None
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during evaluation: {str(e)}")
            return None