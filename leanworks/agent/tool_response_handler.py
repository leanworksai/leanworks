from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import logging

from leanworks.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class ToolResponseHandler(ABC):
    """Abstract base class for handling tool responses"""
    
    @abstractmethod
    def can_handle(self, response) -> bool:
        """Check if this handler can process the response"""
        pass
    
    @abstractmethod
    def handle(self, response, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle the response and return final text if available.
        
        Args:
            response: Claude API response object
            context: Context dict with conversation, tool_use, data_sources, streaming, etc.
        
        Returns:
            Final response text, or None if conversation should continue
        """
        pass

class ServerToolResponseHandler(ToolResponseHandler):
    """Handles responses from Claude built-in server tools"""
    
    def can_handle(self, response) -> bool:
        """
        Check if response contains server tool calls with results.
        Server tools typically return with stop_reason == "end_turn" and include
        tool_use blocks, tool_result blocks, and final text in the same response.
        """
        stop_reason = getattr(response, 'stop_reason', None)
        
        # Server tools complete in one turn with end_turn
        if stop_reason == "end_turn":
            # Check if response contains server tool calls
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type in ["server_tool_use", "tool_use"]:
                    tool_name = getattr(block, 'name', None)
                    if tool_name and ToolRegistry.is_server_tool(tool_name):
                        return True
        elif stop_reason == "pause_turn":
            # Server tools may also pause, check for server tool calls
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type in ["server_tool_use", "tool_use"]:
                    tool_name = getattr(block, 'name', None)
                    if tool_name and ToolRegistry.is_server_tool(tool_name):
                        return True
        
        return False
    
    def handle(self, response, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle server tool response - extract final text after tool results.
        Server tools complete with stop_reason == "end_turn" or "pause_turn".
        """
        conversation = context['conversation']
        data_sources = context['data_sources']
        stop_reason = getattr(response, 'stop_reason', None)
        
        # Extract tool names
        tool_names = self._extract_tool_names(response)
        
        # Add assistant message to conversation (includes tool_use, tool_result, and text blocks)
        conversation.add_assistant_message_with_tool_uses(response)
        
        # Track data sources
        for tool_name in tool_names:
            if ToolRegistry.is_server_tool(tool_name):
                if tool_name == "web_search":
                    data_sources.append("Web search results")
                else:
                    data_sources.append(f"{tool_name} results")
        
        # Handle pause_turn - need to continue the conversation
        if stop_reason == "pause_turn":
            logger.info("Server tool paused, will continue in next iteration")
            return None
        
        # Handle end_turn - extract final text
        if stop_reason == "end_turn":
            final_text = self._extract_final_text(response)
            if final_text:
                logger.info(f"Server tool response handled, extracted text (length: {len(final_text)})")
                # Streaming is handled in the main loop, not here
                return final_text
            else:
                logger.warning("Server tool executed but no final text found")
                return None
        
        # For other stop reasons, try to extract text anyway
        final_text = self._extract_final_text(response)
        if final_text:
            logger.info(f"Server tool response handled with stop_reason={stop_reason}, extracted text (length: {len(final_text)})")
            # Streaming is handled in the main loop, not here
            return final_text
        
        logger.warning(f"Server tool executed with stop_reason={stop_reason} but no final text found")
        return None
    
    def _extract_tool_names(self, response) -> List[str]:
        """Extract tool names from response"""
        tool_names = []
        for block in response.content:
            block_type = getattr(block, 'type', None)
            if block_type in ["tool_use", "server_tool_use"]:
                tool_name = getattr(block, 'name', None)
                if tool_name:
                    tool_names.append(tool_name)
        return tool_names
    
    def _extract_final_text(self, response) -> Optional[str]:
        """Extract all text blocks that come after all tool blocks and join them"""
        # Find index of last tool block
        last_tool_index = -1
        all_tool_block_types = ToolRegistry.SERVER_TOOL_BLOCK_TYPES | ToolRegistry.CLIENT_TOOL_BLOCK_TYPES
        
        for i, block in enumerate(response.content):
            block_type = getattr(block, 'type', None)
            if block_type in all_tool_block_types:
                last_tool_index = i
        
        # Get ALL text blocks after last tool block and join them
        text_blocks_after_tools = [
            block.text for i, block in enumerate(response.content)
            if getattr(block, 'type', None) == "text" and i > last_tool_index
        ]
        
        if text_blocks_after_tools:
            # Join all text blocks, not just return the last one
            return "".join(text_blocks_after_tools)
        
        # Fallback: return all text blocks joined
        all_text_blocks = [
            block.text for block in response.content
            if getattr(block, 'type', None) == "text"
        ]
        return "".join(all_text_blocks) if all_text_blocks else None

class ClientToolResponseHandler(ToolResponseHandler):
    """Handles responses from client-defined tools"""
    
    def can_handle(self, response) -> bool:
        """
        Check if response contains client tool calls.
        Client tools return with stop_reason == "tool_use" and expect execution.
        """
        stop_reason = getattr(response, 'stop_reason', None)
        
        if stop_reason == "tool_use":
            # Check if any tool_use blocks are for client tools
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type == "tool_use":
                    tool_name = getattr(block, 'name', None)
                    if tool_name and not ToolRegistry.is_server_tool(tool_name):
                        return True
        
        return False
    
    def handle(self, response, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle client tool response - execute tools and continue conversation.
        Client tools return with stop_reason == "tool_use" and require execution.
        """
        conversation = context['conversation']
        tool_use = context['tool_use']
        data_sources = context['data_sources']
        streaming = context.get('streaming', False)
        stop_reason = getattr(response, 'stop_reason', None)
        
        # Verify this is a tool_use stop reason
        if stop_reason != "tool_use":
            logger.warning(f"ClientToolResponseHandler called with stop_reason={stop_reason}, expected 'tool_use'")
        
        # Add assistant message with tool_use blocks
        conversation.add_assistant_message_with_tool_uses(response)
        
        # Show tool usage if streaming
        if streaming:
            self._show_tool_usage(response)
        
        # Get RAG storage tool if available
        rag_storage = None
        if hasattr(tool_use, '_get_rag_storage_tool'):
            rag_storage = tool_use._get_rag_storage_tool()
        
        # Execute client tools and add results to conversation
        tool_results = conversation.parse_and_format_tool_results_with_sources(
            response,
            tool_use.function_map,
            data_sources,
            rag_storage=rag_storage
        )

        # Register tool resources in working context
        memory_manager = context.get('memory_manager')
        if memory_manager and hasattr(memory_manager, 'working_context'):
            self._register_tool_resources(tool_results, memory_manager.working_context)

        conversation.add_tool_results(tool_results)
        
        # Show tool results summary if streaming is enabled
        if streaming:
            self._show_tool_results(tool_results)
        
        # Client tools require another API call to get final response
        # Return None to continue the loop
        return None

    def _register_tool_resources(self, tool_results, working_context):
        """
        Register tool-generated resources in working context.

        Args:
            tool_results: List of tool result dictionaries
            working_context: WorkingContext instance
        """
        for tool_result in tool_results:
            if tool_result.get("role") == "user" and isinstance(tool_result.get("content"), list):
                for content_block in tool_result["content"]:
                    if content_block.get("type") == "tool_result":
                        tool_name = content_block.get("tool_call_id", "").split(".")[0] if "." in content_block.get("tool_call_id", "") else None
                        result_content = content_block.get("content", "")

                        # Register DuckDB response resources
                        if tool_name == "save_data_to_duckdb" and isinstance(result_content, dict):
                            response_id = result_content.get("response_id")
                            if response_id:
                                working_context.register_resource(
                                    resource_id=response_id,
                                    resource_type='storage_ref',
                                    path=f'duckdb:response_id:{response_id}',
                                    metadata={
                                        'tool': 'duckdb',
                                        'operation': 'save_data',
                                        'created_at': 'tool_execution'
                                    }
                                )
                                logger.debug(f"Registered DuckDB resource in working context: {response_id}")

                        # Register file path resources (temp files, etc.)
                        if isinstance(result_content, str):
                            # Look for file paths in result content
                            import re
                            file_paths = re.findall(r'(/[\w/.-]+\.\w+|\.{0,2}/[\w/.-]+)', result_content)
                            for file_path in file_paths:
                                if '/tmp/' in file_path or 'temp_' in file_path:
                                    resource_id = f"temp_file_{hash(file_path) % 10000}"
                                    working_context.register_resource(
                                        resource_id=resource_id,
                                        resource_type='temp_file',
                                        path=file_path,
                                        metadata={
                                            'tool': tool_name,
                                            'source': 'tool_result'
                                        }
                                    )
                                    logger.debug(f"Registered temp file in working context: {file_path}")

                        # Register document IDs
                        if isinstance(result_content, str):
                            doc_ids = re.findall(r'\b(doc-|file-|task-|proj-)[a-zA-Z0-9]+\b', result_content)
                            for doc_id in doc_ids:
                                resource_id = f"doc_{doc_id}"
                                working_context.register_resource(
                                    resource_id=resource_id,
                                    resource_type='document_id',
                                    path=doc_id,
                                    metadata={
                                        'tool': tool_name,
                                        'source': 'tool_result'
                                    }
                                )
                                logger.debug(f"Registered document ID in working context: {doc_id}")
    
    def _show_tool_results(self, tool_results):
        """Display tool results information"""
        for tool_result in tool_results:
            if tool_result.get("role") == "user" and isinstance(tool_result.get("content"), list):
                for content_block in tool_result["content"]:
                    if content_block.get("type") == "tool_result":
                        result_content = content_block.get("content", "")
                        if result_content and not result_content.startswith("Error"):
                            result_preview = result_content[:200] + "..." if len(result_content) > 200 else result_content
                            print(f"✅ Tool result: {result_preview}")
                        elif result_content.startswith("Error"):
                            print(f"❌ Tool error: {result_content}")
                        print()
    
    def _show_tool_usage(self, response):
        """Display tool usage information"""
        for block in response.content:
            if getattr(block, 'type', None) == "tool_use":
                tool_name = getattr(block, 'name', 'unknown')
                tool_input = getattr(block, 'input', {})
                print(f"🔧 Using tool: {tool_name}")
                if tool_input:
                    key_params = []
                    for key, value in tool_input.items():
                        if isinstance(value, str) and len(value) > 100:
                            key_params.append(f"{key}: {value[:50]}...")
                        else:
                            key_params.append(f"{key}: {value}")
                    print(f"   Parameters: {', '.join(key_params)}")
                print()

class TextOnlyResponseHandler(ToolResponseHandler):
    """Handles text-only responses (no tool calls)"""
    
    def can_handle(self, response) -> bool:
        """
        Check if response is text-only with end_turn.
        This handles responses that don't require tool execution.
        """
        stop_reason = getattr(response, 'stop_reason', None)
        
        # Text-only responses complete with end_turn and no tool_use blocks
        if stop_reason == "end_turn":
            # Check that there are no tool_use blocks
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type in ["tool_use", "server_tool_use"]:
                    return False
            return True
        
        # Also handle other stop reasons that indicate completion without tools
        if stop_reason in ["max_tokens", "stop_sequence", "refusal", "model_context_window_exceeded"]:
            # Check that there are no tool_use blocks
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type in ["tool_use", "server_tool_use"]:
                    return False
            return True
        
        return False
    
    def handle(self, response, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle text-only response.
        Text-only responses complete with stop_reason == "end_turn" or other completion reasons.
        """
        conversation = context['conversation']
        stop_reason = getattr(response, 'stop_reason', None)
        
        text = next(
            (block.text for block in response.content if getattr(block, 'type', None) == "text"),
            None
        )
        
        if text:
            conversation.add_assistant_message(text)
            # Streaming is handled in the main loop, not here
            
            # Log stop reason for debugging
            if stop_reason != "end_turn":
                logger.info(f"Text-only response with stop_reason={stop_reason}")
        
        return text

class ToolResponseHandlerFactory:
    """Factory for selecting the appropriate response handler"""
    
    def __init__(self):
        # Order matters - check more specific handlers first
        self.handlers = [
            ServerToolResponseHandler(),
            ClientToolResponseHandler(),
            TextOnlyResponseHandler(),
        ]
    
    def get_handler(self, response) -> ToolResponseHandler:
        """Get the first handler that can process the response"""
        for handler in self.handlers:
            if handler.can_handle(response):
                logger.debug(f"Selected handler: {handler.__class__.__name__}")
                return handler
        raise ValueError("No handler found for response")

