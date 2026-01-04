from typing import Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Tool execution types"""
    CLIENT = "client"  # Executed on client side
    SERVER = "server"  # Executed by Anthropic on their servers

class ToolRegistry:
    """Centralized registry for all tools with their execution types"""
    
    # Server tools (executed by Anthropic)
    SERVER_TOOLS: Set[str] = {
        "web_search",
        "web_fetch", 
        "tool_search",
        "code_execution"
    }
    
    # Server tool block types in responses
    SERVER_TOOL_BLOCK_TYPES: Set[str] = {
        "server_tool_use",
        "web_search_tool_result",
        "tool_result"  # Generic tool_result can also be from server tools
    }
    
    # Client tool block types
    CLIENT_TOOL_BLOCK_TYPES: Set[str] = {
        "tool_use"  # Regular tool_use is for client tools
    }
    
    @classmethod
    def is_server_tool(cls, tool_name: str) -> bool:
        """Check if a tool is a server tool"""
        return tool_name in cls.SERVER_TOOLS
    
    @classmethod
    def get_tool_type(cls, tool_name: str) -> ToolType:
        """Get the execution type of a tool"""
        return ToolType.SERVER if cls.is_server_tool(tool_name) else ToolType.CLIENT
    
    @classmethod
    def is_server_tool_block(cls, block_type: str) -> bool:
        """Check if a block type indicates a server tool"""
        return block_type in cls.SERVER_TOOL_BLOCK_TYPES
    
    @classmethod
    def is_client_tool_block(cls, block_type: str) -> bool:
        """Check if a block type indicates a client tool"""
        return block_type in cls.CLIENT_TOOL_BLOCK_TYPES
    
    @classmethod
    def register_server_tool(cls, tool_name: str):
        """Dynamically register a new server tool"""
        cls.SERVER_TOOLS.add(tool_name)
        logger.info(f"Registered server tool: {tool_name}")

