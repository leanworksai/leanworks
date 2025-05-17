from leanworks.agent.tools.toolkit import ToolUse
from datetime import datetime
from leanworks.agent.conversation import ConversationManager
from leanworks.agent.setting import AGENT_SYSTEM_PROMPT, VERIFICATION_QUERY, SEARCH_KNOWLEDGE_QUERY
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
                 user_id=None,
                 session_id=None,
                 max_unanswered_num=2,
                 clear_conversation=True
                 ):
        """
        Initialize the ChatAgent with necessary clients and settings.
        
        Args:
            credentials_path (str): Path to the GCP credentials JSON file
            bucket (str): The storage bucket name
            user_id (str): The user ID for conversation tracking
            session_id (str): The session ID for conversation tracking
            model (str): The Claude model to use
            max_tokens (int): Maximum number of tokens for Claude responses
            temperature (float): Temperature setting for response generation
            timeout (int): API timeout in seconds
            max_unanswered_num (int): Maximum number of attempts to answer a question
        """
        # Initialize clients
        self.storage_client = storage_client
        self.secret_client = secret_client
        self.model_client = model_client
        
        # Set parameters
        self.user_id = user_id
        self.session_id = session_id
        self.max_unanswered_num = max_unanswered_num
        
        # Initialize tool use
        self.tool_use = ToolUse()
        
        # Initialize conversation manager
        self.conversation = ConversationManager(
            self.model_client, 
            self.storage_client, 
            self.user_id, 
            self.session_id
        )
        if clear_conversation:
            self.conversation.clear_conversation()
        
        # Set up API parameters
        self.system_prompt = AGENT_SYSTEM_PROMPT.format(
            USER_ID=self.user_id, 
            CURRENT_DATE=datetime.now().strftime("%Y-%m-%d")
        )
        
        self.api_params = {
            "model": "claude-3-5-haiku-20241022",
            "system": self.system_prompt,
            "messages": self.conversation.conversation,
            "tools": self.tool_use.tools,
            "max_tokens": 512,
            "temperature": 0.1,
            "timeout": 30
        }
    
    def process_message(self, user_message):
        """
        Process a user message and handle the conversation flow.
        
        Args:
            user_message (str): The user's message content
            
        Returns:
            str: The final response to the user
        """
        # Add the user message
        self.conversation.add_user_message(user_message)
        
        # Maximum number of iterations to prevent infinite loops
        unanswered_count = 0
        answered = "false"
        response_text = ""

        while unanswered_count < self.max_unanswered_num:
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
                response_json = self.conversation.extract_json_from_text(text_content)
                response_text = response_json.get("content")
                answered = response_json.get("answered", "false")
                
                # If the answer is complete, break the loop
                if answered == "true":
                    logger.info("Question answered. Starting verification...")
                    self.conversation.add_assistant_message(response_text)
                    
                    # Perform verification
                    verification_result = self._perform_verification()
                    if verification_result:
                        response_text = verification_result
                    
                    break
                
                # Check if there are any tool calls
                has_tool_calls = any(block.type == "tool_use" for block in response.content)
                
                if has_tool_calls:
                    # Add the assistant message with tool_use blocks to the conversation
                    self.conversation.add_assistant_message_with_tool_uses(response)
                    
                    # Process tool calls and add results to conversation
                    tool_results = self.conversation.parse_and_format_tool_results(
                        response, 
                        self.tool_use.function_map
                    )
                    self.conversation.add_tool_results(tool_results)
                else:
                    # No tool calls were made and the answer is still not complete
                    logger.info("No tool calls were made and the answer is still not complete")
                    unanswered_count += 1
                    self.conversation.add_user_message(SEARCH_KNOWLEDGE_QUERY)
                    
                    # Create a copy for search_knowledge with updated parameters
                    current_params = self.conversation.create_params_copy(
                        self.api_params,
                        messages=self.conversation.conversation,
                        tools=[self.tool_use.tools[-1]],
                        tool_choice={"type": "tool", "name": "search_knowledge"}
                    )
                    
                    # Continue to the next iteration with these updated parameters

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during unanswered attempt {unanswered_count}: {str(e)}")
                response_text = f"An error occurred: {str(e)}"
                break
        
        # If we exited the loop without adding the final response, add it now
        if answered == "false" and unanswered_count >= self.max_unanswered_num:
            self.conversation.add_assistant_message(response_text)
        
        self.conversation.save_conversation()
        logger.info(f"Final answer: {response_text}")
        return response_text
    
    def _perform_verification(self):
        """
        Performs verification on the last answer using the search_knowledge tool.
        
        Returns:
            str: The verified response text, or None if verification failed
        """
        try:
            # Add verification query
            self.conversation.add_user_message(VERIFICATION_QUERY)
            
            # Create a new params copy for verification
            verification_params = self.conversation.create_params_copy(
                self.api_params,
                messages=self.conversation.conversation,
                tools=[self.tool_use.tools[-1]],
                tool_choice={"type": "tool", "name": "search_knowledge"}
            )
            
            # Make the verification API call
            verification_response = self.model_client.messages.create(**verification_params)
            
            # Process tool calls and add results to conversation
            has_tool_calls = any(block.type == "tool_use" for block in verification_response.content)
            if has_tool_calls:
                self.conversation.add_assistant_message_with_tool_uses(verification_response)
                tool_results = self.conversation.parse_and_format_tool_results(
                    verification_response, 
                    self.tool_use.function_map
                )
                self.conversation.add_tool_results(tool_results)
                
                # Create a copy without tools and tool_choice for final response
                final_params = self.conversation.create_params_copy(
                    self.api_params,
                    messages=self.conversation.conversation,
                    tools=None,
                    tool_choice=None
                )
                final_response = self.model_client.messages.create(**final_params)

                final_text = next((block.text for block in final_response.content if block.type == "text"), "")
                final_json = self.conversation.extract_json_from_text(final_text)
                final_response_text = final_json.get("content")
                self.conversation.add_assistant_message(final_response_text)
                return final_response_text
                
            return None
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during verification: {str(e)}")
            return None
