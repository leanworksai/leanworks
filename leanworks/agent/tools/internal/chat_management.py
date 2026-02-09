"""
Chat Management Tool - Domain-specific tool for chat/message operations via leanworks-hub API.
Replaces FirestoreTool for message-related operations.
"""
from typing import Dict, List, Any, Optional
from leanworks.agent.tools.base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class ChatManagementTool(BaseAPIClient):
    """Chat/message management operations via leanworks-hub API."""
    
    @property
    def query_messages_property(self):
        """Query chat messages with flexible filtering."""
        return {
            "type": "custom",
            "name": "query_messages",
            "description": """
Query chat messages from Firestore with flexible filtering.

Message Structure:
- chatId: The chat/conversation ID (e.g., 'ai-assistant-{user_id}', 'project-{projectId}', 'team-{teamId}')
- role: Message role ('user' or 'assistant')
- content: Message text content
- timestamp: ISO format timestamp
- userId: Email of the user who sent the message
- memberName: Display name of the message sender
- memberAvatar: Avatar initials of the message sender
- likes: Array of user emails who liked the message
- imageUrls: Array of image URLs attached to the message

Parameters:
- chatId (required): Chat ID to query messages from
- role: Optional filter by message role ('user' or 'assistant')
- afterTimestamp: Optional ISO timestamp string - only return messages after this timestamp
- limit: Maximum number of messages (default 50, max 200)
- orderBy: Order results by timestamp ('asc' for oldest first, 'desc' for newest first, default 'asc')

Examples:
- query_messages(chatId='ai-assistant-user@example.com', limit=20)
- query_messages(chatId='project-123', role='user')
- query_messages(chatId='team-456', afterTimestamp='2024-01-01T00:00:00Z', orderBy='desc')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID to query messages from (required)"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Optional: Filter by message role"
                    },
                    "afterTimestamp": {
                        "type": "string",
                        "description": "Optional: ISO timestamp - only return messages after this timestamp"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages (default 50, max 200)",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200
                    },
                    "orderBy": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Order by timestamp (default 'asc' for oldest first)"
                    }
                },
                "required": ["chatId"]
            }
        }
    
    def query_messages(self, chatId: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Query messages via API.
        
        Args:
            chatId: Chat ID to query
            **kwargs: Query parameters (role, afterTimestamp, limit, orderBy)
            
        Returns:
            List of message dictionaries
        """
        try:
            # Include chatId in params
            params = {"chatId": chatId, **kwargs}
            result = self._make_request('GET', f'/api/messages/{chatId}', params=params)
            logger.info(f"query_messages returned {len(result) if isinstance(result, list) else 'unknown'} messages")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_messages failed: {str(e)}")
            return {"error": str(e)}
