from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import json
import tiktoken
from leanworks.rag.setting import OTHER_MODEL

@dataclass
class Memory:
    """A class representing a single memory entry of a conversation."""
    query: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)

class MemoryManager:
    """Class for managing conversation memory with persistence to cloud storage."""
    
    MAX_TOKENS = 4000  # Maximum number of tokens to keep in memory
    
    def __init__(self, model_client, storage_client, user_id: str, session_id: str):
        """
        Initialize MemoryManager with storage client and session information.
        
        Args:
            storage_client: Initialized CloudStorage client for memory persistence
            user_id: ID of the user whose memory is being managed
            session_id: ID of the current conversation session
        """
        self.model_client = model_client
        self.storage = storage_client
        self.user_id = user_id
        self.session_id = session_id
        self.memory: List[Memory] = []
        self.memory_summary: str = ""
        self.encoding = tiktoken.get_encoding("cl100k_base")  # For token counting
        
        # Load existing memory from storage if available
        self._load_memory()

    def _load_memory(self):
        """Load memory from cloud storage if available."""
        try:
            # Construct the memory file path
            memory_path = f'chat_store/{self.user_id}/{self.session_id}.json'
            
            memory_data = self.storage.download_blob_to_memory(memory_path)
            if memory_data is None:
                print(f"No existing memory found for user {self.user_id} and session {self.session_id}")
                self.memory = []
                self.memory_summary = ""
                return
                
            memory_json = json.loads(memory_data)
            
            # Load memory summary if it exists
            if 'summary' in memory_json:
                self.memory_summary = memory_json['summary']
            
            # Convert JSON to Memory objects
            self.memory = []
            if 'conversations' in memory_json:
                # Process messages in pairs (user + assistant)
                messages = memory_json['conversations']
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):  # Ensure we have both user and assistant messages
                        user_message = messages[i]
                        assistant_message = messages[i + 1]
                        
                        if (user_message['role'] == 'user' and 
                            assistant_message['role'] == 'assistant'):
                            memory_entry = Memory(
                                query=user_message['content'],
                                response=assistant_message['content'],
                                timestamp=datetime.fromisoformat(user_message['timestamp'])
                            )
                            self.memory.append(memory_entry)
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = []
            self.memory_summary = ""

    def _save_memory(self):
        """Save current memory to cloud storage in JSON format."""
        try:
            # Construct the memory file path
            memory_path = f'chat_store/{self.user_id}/{self.session_id}.json'
            
            # Convert Memory objects to JSON-serializable format
            memory_json = {
                'summary': self.memory_summary,
                'conversations': []
            }
            
            for memory_item in self.memory:
                # Create separate entries for user and bot in OpenAI format
                user_message = {
                    'role': 'user',
                    'content': memory_item.query,
                    'timestamp': memory_item.timestamp.isoformat()
                }
                assistant_message = {
                    'role': 'assistant',
                    'content': memory_item.response,
                    'timestamp': memory_item.timestamp.isoformat()
                }
                memory_json['conversations'].append(user_message)
                memory_json['conversations'].append(assistant_message)
            
            # Convert to JSON string
            memory_data = json.dumps(memory_json, indent=2)
            
            # Ensure parent directory exists by creating an empty file if needed
            chat_store_dir = f'chat_store/{self.user_id}/'
            try:
                # Check if directory exists by trying to read a marker file
                marker_path = f'{chat_store_dir}/.marker'
                if self.storage.download_blob_to_memory(marker_path) is None:
                    # Create the marker file to ensure the directory exists
                    self.storage.upload_blob_from_memory("", marker_path)
            except Exception as dir_error:
                print(f"Error ensuring directory exists: {dir_error}, will attempt to save file directly")
            
            # Upload to cloud storage
            self.storage.upload_blob_from_memory(memory_data, memory_path)
        except Exception as e:
            print(f"Error saving memory to cloud storage: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def _get_total_memory_tokens(self) -> int:
        """Calculate the total number of tokens in current memory."""
        total = 0
        for memory in self.memory:
            total += self._count_tokens(memory.query + memory.response)
        return total
    
    def _summarize_memory(self):
        """Summarize older memories when token count exceeds the maximum."""
        if not self.memory:
            return
            
        # If we have a summary and conversations, check total token count
        total_tokens = self._get_total_memory_tokens()
        summary_tokens = self._count_tokens(self.memory_summary) if self.memory_summary else 0
        
        if total_tokens + summary_tokens > self.MAX_TOKENS:
            # Keep the most recent conversations that fit within half the token limit
            token_budget = self.MAX_TOKENS // 2
            recent_memories = []
            current_tokens = 0
            
            # Start from the most recent and work backwards
            for memory in reversed(self.memory):
                memory_tokens = self._count_tokens(memory.query + memory.response)
                if current_tokens + memory_tokens <= token_budget:
                    recent_memories.insert(0, memory)  # Insert at beginning to maintain order
                    current_tokens += memory_tokens
                else:
                    break
            
            # Summarize the older conversations
            older_memories = [m for m in self.memory if m not in recent_memories]
            if older_memories:
                # Format older memories for summarization
                older_memory_text = ""
                for memory in older_memories:
                    older_memory_text += f"User: {memory.query}\nAssistant: {memory.response}\n\n"
                
                # Generate summary using the model client
                try:
                    
                    response = self.model_client.chat.completions.create(
                        model=OTHER_MODEL,
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": "Summarize the following conversation history concisely, preserving key information and context:"},
                            {"role": "user", "content": older_memory_text}
                        ]
                    )
                    new_summary = response.choices[0].message
                    
                    # Combine with existing summary if needed
                    if self.memory_summary:
                        combined_summary = f"{self.memory_summary}\n\nAdditional conversation summary: {new_summary}"
                        # Check if combined summary is too large and summarize again if needed
                        if self._count_tokens(combined_summary) > token_budget:
                            response = self.model_client.chat.completions.create(
                                model=OTHER_MODEL,
                                max_tokens=1024,
                                messages=[
                                    {"role": "system", "content": "Condense these two summaries into a single concise summary:"},
                                    {"role": "user", "content": combined_summary}
                                ]
                            )
                            self.memory_summary = response.choices[0].message
                        else:
                            self.memory_summary = combined_summary
                    else:
                        self.memory_summary = new_summary
                except Exception as e:
                    print(f"Error generating memory summary: {e}")
                    # Fallback: just keep the most recent summary
                    if not self.memory_summary:
                        self.memory_summary = "Previous conversation history exists but could not be summarized."
            
            # Update memory to only include recent conversations
            self.memory = recent_memories

    def add_memory(self, query: str, response: str):
        """
        Add a new conversation to memory and persist to storage.
        
        Args:
            query: The user's question
            response: The system's response
        """
        memory_entry = Memory(query=query, response=response)
        self.memory.append(memory_entry)
        
        # Check if we need to summarize older memories
        self._summarize_memory()
        
        # Persist memory to storage
        self._save_memory()

    def get_recent_memories(self, limit: int = None) -> List[Memory]:
        """
        Get the most recent memories.
        
        Args:
            limit: Maximum number of memories to return. If None, returns all memories.
        
        Returns:
            List of Memory objects, ordered from oldest to newest
        """
        if limit is None or limit > len(self.memory):
            return self.memory
        return self.memory[-limit:]

    def format_memory_context(self, memories: List[Memory] = None) -> List[str]:
        """
        Format memories into a list of conversation strings.
        
        Args:
            memories: List of Memory objects to format. If None, uses all memories.
        
        Returns:
            List of formatted conversation strings
        """
        context = []
        
        # Add summary if it exists
        if self.memory_summary:
            context.append(f"[CONVERSATION SUMMARY]\n{self.memory_summary}\n")
        
        # Add recent conversations
        memories = memories or self.memory
        for memory in memories:
            memory_text = (
                f"[PAST CONVERSATION]\n"
                f"Question: {memory.query}\n"
                f"Answer: {memory.response}\n"
            )
            context.append(memory_text)
            
        return context

    def clear_memory(self):
        """Clear all memories from both local storage and cloud storage."""
        self.memory = []
        self.memory_summary = ""
        try:
            memory_path = f'chat_store/{self.user_id}/{self.session_id}.json'
            self.storage.upload_blob_from_memory('{"summary":"","conversations":[]}', memory_path)
        except Exception as e:
            print(f"Error clearing memory in cloud storage: {e}")