import json
import re
import copy
class ConversationManager:
    """Class for managing conversation history with Claude"""
    
    def __init__(self, model_client, storage_client, user_id=None, session_id=None):
        self.model_client = model_client
        self.storage_client = storage_client
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_path = f"chat_store/{self.user_id}/{self.session_id}.json"
        self.conversation = []
        
        # Load previous conversation if storage client and user_id are provided
        if storage_client and user_id and session_id:
            self.load_conversation()
        
    def load_conversation(self):
        """Load the most recent conversation from CloudStorage"""
        if not self.storage_client or not self.user_id:
            return
            
        conversation_data = self.storage_client.download_blob_to_memory(self.conversation_path)
        
        if conversation_data:
            try:
                loaded_conversation = json.loads(conversation_data)
                
                # Convert any old-format messages to the new format
                for message in loaded_conversation:
                    if "content" in message and isinstance(message["content"], str):
                        message["content"] = [{"type": "text", "text": message["content"]}]
                
                self.conversation = loaded_conversation
                print(f"Loaded conversation history for user {self.user_id}")
            except json.JSONDecodeError:
                print(f"Error decoding conversation data for user {self.user_id}")
                self.conversation = []
        else:
            print(f"No previous conversation found for user {self.user_id}")
            self.conversation = []
    
    def save_conversation(self):
        """Save the current conversation to CloudStorage"""
        if not self.storage_client or not self.user_id:
            return
            
        # Save as latest conversation
        conversation_json = json.dumps(self.conversation)
        self.storage_client.upload_blob_from_memory(conversation_json, self.conversation_path)
        print(f"Saved conversation for user {self.user_id}")
        
    def add_user_message(self, content):
        """Add a user message to the conversation history"""
        self.conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": content}]
        })

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

    def add_assistant_message(self, content):
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
                        # Call the function with the provided input
                        result = function_map[tool_name](**tool_input)
                        
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
            
        try:
            # First try to parse the entire text as JSON
            parsed_json = json.loads(text)         
            return parsed_json
        except json.JSONDecodeError:

            # Look for text that appears to be JSON (between curly braces)
            # Use non-greedy matching to find the outermost JSON object
            try:
                json_match = re.search(r'({.*?})', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                    
                # If no match found with simple regex, try more aggressive pattern
                json_match = re.search(r'({[\s\S]*})', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
                # Try to find JSON in text with preceding content (like in LLM responses)
                json_match = re.search(r'.*?({.*})', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
            except Exception as e:
                # If regex extraction fails, use Claude to extract the JSON
                if self.model_client:
                    prompt = f"Extract the content from below JSON and output it as a natural text:\n\n{text}"
                    try:
                        # Use Claude API to extract JSON using JSON mode
                        messages = [{"role": "user", "content": prompt}]
                        response = self.model_client.messages.create(
                            model="claude-3-haiku-20240307",
                            messages=messages,
                            max_tokens=1000,
                        )
                        if "true" in text.lower():
                            return {"content": response.content[0].text, "answered": "true"}
                        else:
                            return {"content": response.content[0].text, "answered": "false"}
                    except Exception as e:
                        print(f"Claude JSON extraction failed: {e}")
                        
            return {"content": text, "answered": "false"}

    def add_assistant_message_with_tool_uses(self, response):
        """Add an assistant message including both text content and tool use blocks
        
        This method is specifically for adding a Claude response that contains tool use blocks,
        converting them to a JSON-serializable format.
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
        
        # Add the assistant message with the serializable content
        self.conversation.append({
            "role": "assistant",
            "content": serializable_content
        })

    # Helper function to create a modified copy of parameters
    def create_params_copy(self, params, **modifications):
        params_copy = copy.deepcopy(params)
        for key, value in modifications.items():
            if value is None and key in params_copy:
                del params_copy[key]
            elif value is not None:
                params_copy[key] = value
        return params_copy