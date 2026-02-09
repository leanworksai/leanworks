# Re-export core components for backward compatibility
from leanworks.agent.core import ChatAgent, ConversationManager, MemoryManager, WorkingContext
from leanworks.agent.utils import AgentHelpers, FactExtractor, LargeResponseHandler
from leanworks.agent.tools import toolkit, ToolRegistry, ToolResponseHandlerFactory
from leanworks.agent.session_manager import SessionManager  # If needed

__all__ = [
    'ChatAgent',
    'ConversationManager',
    'MemoryManager',
    'WorkingContext',
    'AgentHelpers',
    'FactExtractor',
    'LargeResponseHandler',
    'toolkit',
    'ToolRegistry',
    'ToolResponseHandlerFactory',
    'SessionManager',
]
