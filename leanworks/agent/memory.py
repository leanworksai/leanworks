import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages conversation memory using an incremental summary buffer approach.
    
    This implementation follows the sliding window + rolling summary pattern:
    1. Track tokens in the conversation
    2. When approaching token limit, trigger summarization
    3. Summarize new turns since last summary, combining with previous summary  
    4. Replace old summary with new one and drop summarized raw turns
    5. Maintain system prompt + user profile + running summary + recent N turns
    """
    
    def __init__(self, 
                 model_client,
                 storage_client,
                 user_id: str = None,
                 session_id: str = None,
                 max_context_tokens: int = 180000,  # Conservative for 200K models
                 trigger_threshold: int = 150000,   # Trigger at ~75% capacity
                 summary_max_tokens: int = 2000,    # More detailed summaries
                 recent_turns_to_keep: int = 20,    # Keep more recent context
                 summarization_model: str = "claude-3-haiku-20240307",
                 main_model: str = None):
        """
        Initialize the MemoryManager.
        
        Args:
            model_client: Claude model client for summarization
            storage_client: Storage client for persisting memory state
            user_id: User identifier for storage
            session_id: Session identifier for storage
            max_context_tokens: Maximum tokens allowed in context
            trigger_threshold: Token count that triggers summarization
            summary_max_tokens: Maximum tokens for summary
            recent_turns_to_keep: Number of recent message pairs to always keep
            summarization_model: Model to use for summarization (cheaper model)
            main_model: Model used for main conversation (for accurate token counting)
        """
        self.model_client = model_client
        self.storage_client = storage_client
        self.user_id = user_id
        self.session_id = session_id
        
        # Token management settings
        self.max_context_tokens = max_context_tokens
        self.trigger_threshold = trigger_threshold  
        self.summary_max_tokens = summary_max_tokens
        self.recent_turns_to_keep = recent_turns_to_keep
        self.summarization_model = summarization_model
        self.main_model = main_model or summarization_model  # Use main model or fall back to summarization model
        
        # Memory state
        self.running_summary = ""
        self.conversation_turns = []  # List of message turns (user + assistant pairs)
        self.system_prompt = ""
        self.user_profile = ""
        
        # Storage path for memory state
        self.memory_path = f"memory_store/{self.user_id}/{self.session_id}_memory.json"
        
        # Background processing setup
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-bg")
        self._token_count_cache = 0  # Cached token count
        self._token_count_future = None  # Future for ongoing token calculation
        self._token_count_lock = threading.Lock()  # Thread safety
        self._last_token_calculation_turns = 0  # Track when we last calculated
        self._shutdown = False  # Track shutdown state
        
        # Load existing memory state
        self.load_memory_state()
    
    @classmethod
    def create_for_model(cls, 
                        model_name: str,
                        model_client,
                        storage_client,
                        user_id: str = None,
                        session_id: str = None,
                        **kwargs):
        """
        Create a MemoryManager with optimal settings for a specific Claude model.
        
        Args:
            model_name: Claude model name (e.g., 'claude-sonnet-4-20250514')
            model_client: Claude model client
            storage_client: Storage client
            user_id: User identifier
            session_id: Session identifier
            **kwargs: Override any default settings
        
        Returns:
            MemoryManager: Configured instance
        """
        # Model-specific configurations based on Anthropic documentation
        # https://docs.anthropic.com/en/docs/about-claude/models/overview#model-comparison-table
        model_configs = {
            # Claude 4 models - 200K context
            "claude-opus-4": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 3000,
                "recent_turns_to_keep": 25,
                "summarization_model": "claude-3-haiku-20240307"
            },
            "claude-sonnet-4": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 2500,
                "recent_turns_to_keep": 25,
                "summarization_model": "claude-3-haiku-20240307"
            },
            "claude-haiku-4": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 1500,
                "recent_turns_to_keep": 20,
                "summarization_model": "claude-3-haiku-20240307"
            },
            # Claude 3.7 - 200K context  
            "claude-3-7-sonnet": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 2000,
                "recent_turns_to_keep": 20,
                "summarization_model": "claude-3-haiku-20240307"
            },
            # Claude 3.5 models - 200K context
            "claude-3-5-sonnet": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 2000,
                "recent_turns_to_keep": 20,
                "summarization_model": "claude-3-haiku-20240307"
            },
            "claude-3-5-haiku": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 1500,
                "recent_turns_to_keep": 15,
                "summarization_model": "claude-3-haiku-20240307"
            },
            # Claude 3 Haiku - 200K context but smaller output
            "claude-3-haiku": {
                "max_context_tokens": 180000,
                "trigger_threshold": 150000,
                "summary_max_tokens": 1000,
                "recent_turns_to_keep": 15,
                "summarization_model": "claude-3-haiku-20240307"
            }
        }
        
        # Find matching configuration
        config = None
        for model_key, model_config in model_configs.items():
            if model_key in model_name.lower():
                config = model_config.copy()
                break
        
        # Fallback to conservative defaults for unknown models
        if config is None:
            logger.warning(f"Unknown model {model_name}, using conservative defaults")
            config = {
                "max_context_tokens": 60000,  # Conservative for older models
                "trigger_threshold": 45000,
                "summary_max_tokens": 1500,
                "recent_turns_to_keep": 15,
                "summarization_model": "claude-3-haiku-20240307"
            }
        
        # Apply any user overrides
        config.update(kwargs)
        
        logger.info(f"Creating MemoryManager for {model_name} with config: {config}")
        
        return cls(
            model_client=model_client,
            storage_client=storage_client,
            user_id=user_id,
            session_id=session_id,
            main_model=model_name,  # Pass the actual model name for accurate token counting
            **config
        )
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Uses a simple approximation: ~4 characters per token for English text.
        This is a fallback when the official API is not available.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def count_tokens_accurate(self, messages: List[Dict[str, Any]], system_prompt: str = None) -> int:
        """
        Get accurate token count using Claude's official token counting API.
        
        Args:
            messages: List of messages in Claude format
            system_prompt: Optional system prompt
            
        Returns:
            int: Accurate token count from Claude API, or estimate if API fails
        """
        try:
            # Prepare request for token counting API
            count_params = {
                "model": self.main_model,  # Use the same model as the main conversation for accuracy
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                count_params["system"] = system_prompt
            
            # Call Claude's token counting API
            # https://docs.anthropic.com/en/docs/build-with-claude/token-counting
            response = self.model_client.messages.count_tokens(**count_params)
            
            logger.debug(f"Accurate token count: {response.input_tokens} tokens using model {self.main_model}")
            return response.input_tokens
            
        except Exception as e:
            logger.warning(f"Failed to get accurate token count, falling back to estimation: {e}")
            
            # Fallback to estimation
            total_tokens = 0
            
            # Count system prompt tokens
            if system_prompt:
                total_tokens += self.estimate_tokens(system_prompt)
            
            # Count message tokens
            for msg in messages:
                total_tokens += self.estimate_message_tokens(msg)
            
            return total_tokens
    
    def estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate tokens for a single message in Claude format."""
        tokens = 0
        
        # Add role tokens
        tokens += 3  # ~3 tokens for role formatting
        
        # Add content tokens
        content = message.get("content", [])
        if isinstance(content, str):
            tokens += self.estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        tokens += self.estimate_tokens(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Tool use blocks are typically small
                        tokens += 20  # Rough estimate
                    elif block.get("type") == "tool_result":
                        tokens += self.estimate_tokens(str(block.get("content", "")))
        
        return tokens
    
    def estimate_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens for a list of messages."""
        return sum(self.estimate_message_tokens(msg) for msg in messages)
    
    def load_memory_state(self):
        """Load memory state from storage."""
        if not self.storage_client or not self.user_id:
            logger.info("No storage client or user_id provided, starting with empty memory")
            return
            
        try:
            memory_data = self.storage_client.download_blob_to_memory(self.memory_path)
            if memory_data:
                state = json.loads(memory_data)
                self.running_summary = state.get("running_summary", "")
                self.conversation_turns = state.get("conversation_turns", [])
                self.system_prompt = state.get("system_prompt", "")
                self.user_profile = state.get("user_profile", "")
                logger.info(f"Loaded memory state with {len(self.conversation_turns)} turns and summary length {len(self.running_summary)}")
            else:
                logger.info("No existing memory state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading memory state: {str(e)}")
            # Start with empty state on error
            self.running_summary = ""
            self.conversation_turns = []
    
    def save_memory_state(self):
        """Save memory state to storage."""
        if not self.storage_client or not self.user_id:
            return
            
        try:
            state = {
                "running_summary": self.running_summary,
                "conversation_turns": self.conversation_turns,
                "system_prompt": self.system_prompt,
                "user_profile": self.user_profile,
                "last_updated": datetime.now().isoformat()
            }
            
            memory_json = json.dumps(state)
            self.storage_client.upload_blob_from_memory(memory_json, self.memory_path)
            logger.info("Saved memory state")
        except Exception as e:
            logger.error(f"Error saving memory state: {str(e)}")
    
    def add_turn(self, user_message: Dict[str, Any], assistant_message: Dict[str, Any] = None):
        """
        Add a conversation turn (user message + optional assistant response).
        Triggers background token calculation and non-blocking summarization if needed.
        """
        turn = {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_turns.append(turn)
        
        # Invalidate token cache since we added content
        with self._token_count_lock:
            self._token_count_cache = 0
            self._last_token_calculation_turns = 0
        
        # Start background token calculation for next time
        self._start_background_token_calculation()
        
        # Check if we need to summarize (non-blocking)
        if self._should_trigger_summarization():
            logger.info("Token threshold exceeded, triggering background summarization")
            self._perform_summarization_background()
    
    def update_assistant_response(self, assistant_message: Dict[str, Any]):
        """Update the assistant response for the most recent turn."""
        if self.conversation_turns:
            self.conversation_turns[-1]["assistant_message"] = assistant_message
            
            # Invalidate token cache since we updated content
            with self._token_count_lock:
                self._token_count_cache = 0
                self._last_token_calculation_turns = 0
            
            # Start background token calculation for next time
            self._start_background_token_calculation()
            
            # Check if we need to summarize after adding the response (non-blocking)
            if self._should_trigger_summarization():
                logger.info("Token threshold exceeded after assistant response, triggering background summarization")
                self._perform_summarization_background()
    
    def _should_trigger_summarization(self) -> bool:
        """Check if summarization should be triggered based on token count."""
        if len(self.conversation_turns) <= self.recent_turns_to_keep:
            return False
            
        # Calculate current token usage
        current_tokens = self._calculate_current_tokens()
        return current_tokens >= self.trigger_threshold
    
    def _calculate_current_tokens(self) -> int:
        """
        Calculate current total token usage using background processing when possible.
        Falls back to synchronous calculation if needed.
        """
        with self._token_count_lock:
            current_turns = len(self.conversation_turns)
            
            # If we have a recent calculation and turns haven't changed much, use cache
            if (self._token_count_cache > 0 and 
                abs(current_turns - self._last_token_calculation_turns) <= 1):
                logger.debug(f"Using cached token count: {self._token_count_cache}")
                return self._token_count_cache
            
            # Check if background calculation is complete
            if self._token_count_future and self._token_count_future.done():
                try:
                    self._token_count_cache = self._token_count_future.result()
                    self._last_token_calculation_turns = current_turns
                    logger.debug(f"Retrieved background token count: {self._token_count_cache}")
                    return self._token_count_cache
                except Exception as e:
                    logger.warning(f"Background token calculation failed: {e}")
                    self._token_count_future = None
            
            # If no background calculation is running, start one for next time
            if not self._token_count_future or self._token_count_future.done():
                self._start_background_token_calculation()
            
            # For immediate needs, use fast estimation
            estimated_tokens = self._estimate_current_tokens_fast()
            logger.debug(f"Using estimated token count: {estimated_tokens}")
            return estimated_tokens
    
    def _calculate_current_tokens_sync(self) -> int:
        """Synchronous token calculation (original implementation)."""
        # Build the complete context as it would be sent to Claude
        context_parts = []
        
        # System prompt with user profile and summary
        if self.system_prompt:
            context_parts.append(f"System: {self.system_prompt}")
        
        if self.user_profile:
            context_parts.append(f"User Profile: {self.user_profile}")
        
        if self.running_summary:
            context_parts.append(f"Previous Conversation Summary: {self.running_summary}")
        
        combined_system_prompt = "\n\n".join(context_parts) if context_parts else ""
        
        # Convert conversation turns to messages
        messages = []
        for turn in self.conversation_turns:
            messages.append(turn["user_message"])
            if turn["assistant_message"]:
                messages.append(turn["assistant_message"])
        
        # Use accurate token counting if we have messages, otherwise fall back to estimation
        if messages:
            return self.count_tokens_accurate(messages, combined_system_prompt)
        else:
            # No messages yet, just count system context
            return self.estimate_tokens(combined_system_prompt)
    
    def _estimate_current_tokens_fast(self) -> int:
        """Fast token estimation for immediate use."""
        total_tokens = 0
        
        # System context estimation
        if self.system_prompt:
            total_tokens += self.estimate_tokens(self.system_prompt)
        if self.user_profile:
            total_tokens += self.estimate_tokens(self.user_profile)
        if self.running_summary:
            total_tokens += self.estimate_tokens(self.running_summary)
        
        # Conversation turns estimation
        for turn in self.conversation_turns:
            total_tokens += self.estimate_message_tokens(turn["user_message"])
            if turn["assistant_message"]:
                total_tokens += self.estimate_message_tokens(turn["assistant_message"])
        
        return total_tokens
    
    def _start_background_token_calculation(self):
        """Start background token calculation for next time."""
        if self._shutdown:
            return
            
        try:
            self._token_count_future = self._executor.submit(self._calculate_current_tokens_sync)
            logger.debug("Started background token calculation")
        except Exception as e:
            logger.warning(f"Failed to start background token calculation: {e}")
            self._token_count_future = None
    
    def _perform_summarization_background(self):
        """
        Trigger background summarization without blocking the main thread.
        """
        if self._shutdown:
            return
            
        try:
            self._executor.submit(self._perform_summarization)
            logger.debug("Started background summarization")
        except Exception as e:
            logger.warning(f"Failed to start background summarization: {e}")
            # Fallback to synchronous summarization
            self._perform_summarization()
    
    def _perform_summarization(self):
        """
        Perform summarization of older conversation turns.
        Keeps the most recent N turns and summarizes the rest.
        """
        if len(self.conversation_turns) <= self.recent_turns_to_keep:
            logger.info("Not enough turns to summarize")
            return
        
        # Split conversation into parts to summarize and parts to keep
        turns_to_summarize = self.conversation_turns[:-self.recent_turns_to_keep]
        recent_turns = self.conversation_turns[-self.recent_turns_to_keep:]
        
        # Convert turns to text for summarization
        conversation_text = self._turns_to_text(turns_to_summarize)
        
        try:
            # Create summarization prompt
            prompt = self._create_summarization_prompt(conversation_text)
            
            # Call model for summarization
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            response = self.model_client.messages.create(
                model=self.summarization_model,
                messages=messages,
                max_tokens=self.summary_max_tokens,
                temperature=0.1,
                timeout=30
            )
            
            new_summary = next((block.text for block in response.content if block.type == "text"), "")
            
            if new_summary:
                self.running_summary = new_summary
                self.conversation_turns = recent_turns
                
                # Invalidate token cache after summarization
                with self._token_count_lock:
                    self._token_count_cache = 0
                    self._last_token_calculation_turns = 0
                
                self.save_memory_state()
                logger.info(f"Background summarization completed. New summary length: {len(new_summary)}, turns reduced from {len(turns_to_summarize) + len(recent_turns)} to {len(recent_turns)}")
            else:
                logger.error("Summarization failed - no summary text received")
                
        except Exception as e:
            logger.error(f"Error during background summarization: {str(e)}")
            # Don't drop conversation turns if summarization fails
    
    def _turns_to_text(self, turns: List[Dict[str, Any]]) -> str:
        """Convert conversation turns to readable text for summarization."""
        text_parts = []
        
        for i, turn in enumerate(turns):
            user_msg = turn["user_message"]
            assistant_msg = turn.get("assistant_message")
            
            # Extract user message text
            user_text = self._extract_message_text(user_msg)
            text_parts.append(f"User: {user_text}")
            
            # Extract assistant message text if present
            if assistant_msg:
                assistant_text = self._extract_message_text(assistant_msg)
                text_parts.append(f"Assistant: {assistant_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_message_text(self, message: Dict[str, Any]) -> str:
        """Extract readable text from a message in Claude format."""
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
    
    def _create_summarization_prompt(self, conversation_text: str) -> str:
        """Create prompt for summarization."""
        existing_summary_part = ""
        if self.running_summary:
            existing_summary_part = f"\n\nExisting running summary:\n{self.running_summary}\n"
        
        prompt = f"""Update the existing running summary to include the new conversation below. The updated summary should:

1. Remain under {self.summary_max_tokens} tokens
2. Preserve important tasks, decisions, and open questions
3. Maintain key context and user preferences
4. Focus on actionable items and ongoing topics
5. Be concise but comprehensive

{existing_summary_part}

New conversation to incorporate:
{conversation_text}

Please provide an updated running summary that incorporates both the existing summary and the new conversation content:"""

        return prompt
    
    def get_context_for_inference(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get the context for model inference.
        Returns: (combined_context_text, recent_messages_list)
        """
        context_parts = []
        
        # Add system prompt if available
        if self.system_prompt:
            context_parts.append(f"System: {self.system_prompt}")
        
        # Add user profile if available
        if self.user_profile:
            context_parts.append(f"User Profile: {self.user_profile}")
        
        # Add running summary if available
        if self.running_summary:
            context_parts.append(f"Previous Conversation Summary: {self.running_summary}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Convert recent turns back to message format
        recent_messages = []
        for turn in self.conversation_turns:
            recent_messages.append(turn["user_message"])
            if turn["assistant_message"]:
                recent_messages.append(turn["assistant_message"])
        
        return combined_context, recent_messages
    
    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt."""
        self.system_prompt = system_prompt
        self.save_memory_state()
    
    def set_user_profile(self, user_profile: str):
        """Set the user profile information."""
        self.user_profile = user_profile
        self.save_memory_state()
    
    def clear_memory(self):
        """Clear all memory state."""
        self.running_summary = ""
        self.conversation_turns = []
        self.system_prompt = ""
        self.user_profile = ""
        
        # Clear background processing state
        with self._token_count_lock:
            self._token_count_cache = 0
            self._last_token_calculation_turns = 0
            if self._token_count_future:
                self._token_count_future.cancel()
                self._token_count_future = None
        
        self.save_memory_state()
        logger.info("Memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage."""
        current_tokens = self._calculate_current_tokens()
        
        # Check background processing status
        with self._token_count_lock:
            bg_calculation_status = "idle"
            if self._token_count_future:
                if self._token_count_future.running():
                    bg_calculation_status = "calculating"
                elif self._token_count_future.done():
                    bg_calculation_status = "completed"
                else:
                    bg_calculation_status = "pending"
        
        return {
            "total_tokens": current_tokens,
            "summary_tokens": self.estimate_tokens(self.running_summary),
            "conversation_turns": len(self.conversation_turns),
            "trigger_threshold": self.trigger_threshold,
            "max_context_tokens": self.max_context_tokens,
            "tokens_until_trigger": max(0, self.trigger_threshold - current_tokens),
            "summary_length": len(self.running_summary),
            "last_summarization": "Never" if not self.running_summary else "Has summary",
            "background_token_calculation": bg_calculation_status,
            "token_cache_valid": self._token_count_cache > 0
        }
    
    def shutdown(self):
        """Shutdown the memory manager and clean up background threads."""
        try:
            # Check if already shutdown
            if hasattr(self, '_shutdown') and self._shutdown:
                return
            
            self._shutdown = True
            
            # Cancel any pending operations
            if hasattr(self, '_token_count_lock'):
                with self._token_count_lock:
                    if hasattr(self, '_token_count_future') and self._token_count_future:
                        self._token_count_future.cancel()
                        self._token_count_future = None
            
            # Shutdown the executor with timeout to avoid hanging
            if hasattr(self, '_executor') and self._executor:
                try:
                    # Use shutdown with timeout to prevent hanging
                    self._executor.shutdown(wait=False)  # Don't wait for completion
                    logger.info("MemoryManager shutdown initiated")
                except Exception as e:
                    logger.warning(f"Error shutting down executor: {e}")
            
        except Exception as e:
            logger.error(f"Error during MemoryManager shutdown: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Only attempt shutdown if not already done
            if not (hasattr(self, '_shutdown') and self._shutdown):
                self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.shutdown()
