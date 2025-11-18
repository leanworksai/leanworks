import json
import re
import copy
import logging
from typing import List
from google.cloud import firestore

logger = logging.getLogger(__name__)
class ConversationManager:
    """Class for managing conversation history with Claude"""
    
    def __init__(self, model_client, firestore_client, domain, user_id=None, session_id=None):
        self.model_client = model_client
        self.firestore_client = firestore_client
        self.domain = domain
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_path = f"chat_store/{self.user_id}/{self.session_id}"
        self.conversation = []
        self.slim_conversation = []  # Only tracks initial user query and verified responses
        
        # Load previous conversation if firestore client and user_id are provided
        if firestore_client and user_id and session_id:
            self.load_conversation()
        
    def load_conversation(self):
        """Load the most recent slim conversation from Firestore."""
        if not self.firestore_client or not self.user_id or not self.domain:
            return
        
        try:
            file_ref = self.firestore_client.collection('domains').document(self.domain).collection('files').document(self.conversation_path)
            doc = file_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                content = data.get('content', [])
                
                # Ensure slim_conversation is always a list and take last 6 messages
                if isinstance(content, list):
                    self.slim_conversation = content[-6:] if len(content) > 6 else content
                else:
                    print(f"Conversation data for user {self.user_id} is not a list. Creating new list.")
                    self.slim_conversation = []
                print(f"Loaded slim conversation history for user {self.user_id}")
            else:
                print(f"No previous conversation found for user {self.user_id}")
                self.slim_conversation = []
        except Exception as e:
            logger.error(f"Error loading conversation from Firestore: {e}")
            self.slim_conversation = []
        
        # Initialize the current conversation as empty
        self.conversation = self.slim_conversation.copy()
    
    def save_conversation(self):
        """Save conversation messages to Firestore in the same format as web frontend.
        
        Messages are saved to domains/{domain}/messages collection with format:
        - chatId: session_id (e.g., 'ai-assistant-{user_id}')
        - role: 'user' or 'assistant'
        - content: message text content
        - timestamp: Firestore Timestamp
        - userId: user_id (email)
        - memberName: extracted from user info or 'AI Assistant' for assistant messages
        - memberAvatar: initials or 'AI' for assistant
        """
        if not self.firestore_client or not self.user_id or not self.domain or not self.session_id:
            return
        
        try:
            from datetime import datetime
            
            # Collection path matches web frontend: domains/{domain}/messages
            messages_collection = self.firestore_client.collection('domains').document(self.domain).collection('messages')
            
            # Extract user info for memberName and memberAvatar
            user_doc_ref = self.firestore_client.collection('domains').document(self.domain).collection('users').document(self.user_id)
            user_doc = user_doc_ref.get()
            
            user_first_name = ""
            user_last_name = ""
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_first_name = user_data.get('firstName', '')
                user_last_name = user_data.get('lastName', '')
            
            user_member_name = f"{user_first_name} {user_last_name}".strip() or self.user_id
            user_member_avatar = f"{user_first_name[0] if user_first_name else ''}{user_last_name[0] if user_last_name else ''}".upper() or self.user_id[0].upper()
            
            # Process slim_conversation to save individual messages
            # Only save new messages (check what's already saved)
            chat_id = self.session_id  # Use session_id as chatId
            
            # Get existing messages for this chatId to avoid duplicates
            existing_messages_query = messages_collection.where('chatId', '==', chat_id).where('userId', '==', self.user_id.lower())
            existing_docs = existing_messages_query.get()
            existing_contents = {doc.get('content') for doc in existing_docs}
            
            # Save each message in the slim_conversation
            for message in self.slim_conversation:
                role = message.get('role', '')
                content_blocks = message.get('content', [])
                
                # Extract text content from content blocks
                text_content = ""
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_content = block.get('text', '')
                            break
                        elif isinstance(block, str):
                            text_content = block
                            break
                elif isinstance(content_blocks, str):
                    text_content = content_blocks
                
                # Skip if content is empty or already exists
                if not text_content or text_content in existing_contents:
                    continue
                
                # Determine memberName and memberAvatar based on role
                if role == 'user':
                    member_name = user_member_name
                    member_avatar = user_member_avatar
                else:  # assistant
                    member_name = 'AI Assistant'
                    member_avatar = 'AI'
                
                # Create message document matching web frontend format
                # Use Firestore Timestamp (same as web frontend uses new Date())
                message_data = {
                    'chatId': chat_id,
                    'role': role,
                    'content': text_content,
                    'timestamp': firestore.SERVER_TIMESTAMP,  # Firestore Timestamp (matches web frontend's new Date())
                    'userId': self.user_id.lower(),
                    'memberName': member_name,
                    'memberAvatar': member_avatar,
                    'projectId': None,
                    'teamId': None
                }
                
                # Save to Firestore
                messages_collection.add(message_data)
                logger.info(f"Saved {role} message to Firestore for chatId: {chat_id}")
            
            logger.info(f"Saved conversation messages to Firestore for user {self.user_id}, session {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving conversation to Firestore: {e}")
            import traceback
            traceback.print_exc()
        
    def add_user_message(self, content, include_in_slim=False):
        """Add a user message to the conversation history"""
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": content}]
        }
        self.conversation.append(user_message)
        
        if include_in_slim:
            self.slim_conversation.append(user_message)

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
        if include_in_slim:
            self.slim_conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": content}]
            })
            
    def clear_conversation(self):
        """Clear the conversation history and start fresh"""
        self.conversation = []
        self.slim_conversation = []
        
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

    def parse_and_format_tool_results_with_sources(self, response, function_map, data_sources):
        """
        Parse tool use blocks from Claude's response and execute the corresponding functions.
        Returns properly formatted tool_result content blocks following Anthropic's API format.
        Also tracks data sources used by each tool.
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
                        
                        # Log tool call result preview
                        result_preview = self._get_result_preview(result)
                        logger.info(f"Tool call result for {tool_name}: {result_preview}")
                        
                        # Track data sources based on tool type
                        if tool_name == "query_postgres":
                            # For PostgreSQL tools, extract table names from SQL query
                            try:
                                # Try to get domain from the function's bound method
                                postgres_tool = getattr(function_map[tool_name], '__self__', None)
                                domain = getattr(postgres_tool, 'domain', 'leanworks.ai') if postgres_tool else 'leanworks.ai'
                                
                                # Extract SQL query from tool_input
                                sql_query = tool_input.get('sql', '') if isinstance(tool_input, dict) else ''
                                
                                if sql_query:
                                    # Parse SQL to extract table names
                                    table_names = self._extract_table_names_from_sql(sql_query)
                                    if table_names:
                                        tables_str = ', '.join(table_names)
                                        data_sources.append(f"PostgreSQL tables: {tables_str} (domain: {domain})")
                                    else:
                                        data_sources.append(f"PostgreSQL database (domain: {domain})")
                                else:
                                    data_sources.append(f"PostgreSQL database (domain: {domain})")
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