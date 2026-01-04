import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime

logger = logging.getLogger(__name__)

# AI Agent identifier - all messages are sent as 'lean'
AI_AGENT_ID = 'lean@leanworks.ai'
AI_AGENT_NAME = 'Lean'
AI_AGENT_AVATAR = 'L'


class FirestoreTool:
    """
    Firestore tool for querying messages and other Firestore collections.
    
    Provides read-only access to Firestore collections, primarily for accessing
    messages stored in orgs/{orgSlug}/messages collection.
    """
    
    def __init__(self, firestore_client, org_slug: str, user_id: str = None):
        """
        Initialize FirestoreTool with Firestore client and org context.
        
        Args:
            firestore_client: Firestore client instance
            org_slug: Organization slug (e.g., 'leanworks.ai')
            user_id: Optional user ID for filtering user-specific data
        """
        self.firestore_client = firestore_client
        self.org_slug = org_slug
        self.user_id = user_id
    
    @property
    def query_messages_property(self):
        description = f"""
        Query chat messages from Firestore for org `{self.org_slug}`.
        
        This tool provides read-only access to messages stored in Firestore.
        Messages are stored in the orgs/{self.org_slug}/messages collection.
        
        Message Structure:
        - chatId: The chat/conversation ID (e.g., 'ai-assistant-{{user_id}}', 'project-{{projectId}}', 'team-{{teamId}}')
        - role: Message role ('user' or 'assistant')
        - content: Message text content
        - timestamp: Firestore Timestamp (ISO format when returned)
        - userId: Email of the user who sent the message
        - projectId: Optional project ID if message is in a project channel
        - teamId: Optional team ID if message is in a team channel
        - memberName: Display name of the message sender
        - memberAvatar: Avatar initials of the message sender
        - likes: Array of user emails who liked the message
        - imageUrls: Array of image URLs attached to the message
        - citedContext: Optional cited context information
        
        Query Parameters:
        - chatId: Filter messages by chat ID (required for most queries)
        - userId: Optional filter by user ID (email)
        - projectId: Optional filter by project ID
        - teamId: Optional filter by team ID
        - role: Optional filter by message role ('user' or 'assistant')
        - limit: Maximum number of messages to return (default: 50, max: 200)
        - orderBy: Order results by 'timestamp' (default: 'asc' for oldest first, use 'desc' for newest first)
        - afterTimestamp: Optional ISO timestamp string - only return messages after this timestamp
        
        Examples:
        - Query all messages in a chat: {{"chatId": "ai-assistant-user@example.com"}}
        - Query user messages only: {{"chatId": "project-123", "role": "user"}}
        - Query recent messages: {{"chatId": "team-456", "limit": 10, "orderBy": "desc"}}
        - Query messages after a date: {{"chatId": "ai-assistant-user@example.com", "afterTimestamp": "2024-01-01T00:00:00Z"}}
        
        Security Notes:
        - For AI assistant conversations (chatId starts with 'ai-assistant-'), messages are automatically filtered by userId for privacy
        - Project and team channel access should be verified by the caller
        - This tool is read-only and cannot modify messages
        """
        return {
            "type": "custom",
            "name": "query_messages",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID to query messages from (e.g., 'ai-assistant-user@example.com', 'project-123', 'team-456')"
                    },
                    "userId": {
                        "type": "string",
                        "description": "Optional: Filter messages by user ID (email address)"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Optional: Filter messages by project ID"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Optional: Filter messages by team ID"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Optional: Filter messages by role ('user' or 'assistant')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200
                    },
                    "orderBy": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Order results by timestamp (default: 'asc' for oldest first, 'desc' for newest first)"
                    },
                    "afterTimestamp": {
                        "type": "string",
                        "description": "Optional: ISO timestamp string - only return messages after this timestamp (e.g., '2024-01-01T00:00:00Z')"
                    }
                },
                "required": ["chatId"]
            }
        }
    
    def query_messages(
        self,
        chatId: str,
        userId: Optional[str] = None,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 50,
        orderBy: str = "asc",
        afterTimestamp: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query messages from Firestore.
        
        Args:
            chatId: Chat ID to query
            userId: Optional user ID filter
            projectId: Optional project ID filter
            teamId: Optional team ID filter
            role: Optional role filter ('user' or 'assistant')
            limit: Maximum number of messages (default: 50, max: 200)
            orderBy: Order by timestamp ('asc' or 'desc', default: 'asc')
            afterTimestamp: Optional ISO timestamp string to filter messages after
            
        Returns:
            List of message dictionaries, or error dictionary
        """
        try:
            if not self.firestore_client or not self.org_slug:
                return {"error": "Firestore client or org_slug not initialized"}
            
            if not chatId:
                return {"error": "chatId is required"}
            
            # Validate limit
            limit = min(max(1, limit), 200)
            
            # Build collection path
            messages_path = f"orgs/{self.org_slug}/messages"
            messages_collection = self.firestore_client.collection(messages_path)
            
            # Start building query
            query = messages_collection.where(filter=FieldFilter('chatId', '==', chatId))
            
            # Security: For AI assistant conversations, filter by userId
            if chatId.startswith('ai-assistant-'):
                if self.user_id and not chatId.endswith(f"-{self.user_id}"):
                    # User trying to access another user's AI conversation - deny
                    return {"error": "Access denied: Cannot access another user's AI assistant conversation"}
                if self.user_id:
                    query = query.where(filter=FieldFilter('userId', '==', self.user_id.lower()))
            
            # Additional filters
            if userId:
                query = query.where(filter=FieldFilter('userId', '==', userId.lower()))
            
            if projectId:
                query = query.where(filter=FieldFilter('projectId', '==', projectId))
            
            if teamId:
                query = query.where(filter=FieldFilter('teamId', '==', teamId))
            
            if role:
                query = query.where(filter=FieldFilter('role', '==', role))
            
            if afterTimestamp:
                try:
                    # Parse ISO timestamp string
                    after_dt = datetime.fromisoformat(afterTimestamp.replace('Z', '+00:00'))
                    query = query.where(filter=FieldFilter('timestamp', '>', after_dt))
                except Exception as e:
                    logger.warning(f"Invalid afterTimestamp format: {afterTimestamp}, error: {e}")
            
            # Order by timestamp
            try:
                if orderBy == "desc":
                    query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
                else:
                    query = query.order_by('timestamp', direction=firestore.Query.ASCENDING)
            except Exception as e:
                # If index error, fetch without orderBy and sort in memory
                if 'index' in str(e).lower() or (hasattr(e, 'code') and e.code == 9):
                    logger.warning(f"Index not found for orderBy, will sort in memory: {e}")
                    # We'll sort in memory after fetching
                else:
                    raise
            
            # Limit results
            query = query.limit(limit)
            
            # Execute query
            logger.info(f"Querying Firestore messages: chatId={chatId}, limit={limit}, orderBy={orderBy}")
            snapshot = query.get()
            
            # Convert to list of dictionaries
            messages = []
            for doc in snapshot:
                data = doc.to_dict()
                message = {
                    'id': doc.id,
                    **data,
                    # Convert Firestore Timestamp to ISO string
                    'timestamp': data.get('timestamp').to_date().isoformat() if data.get('timestamp') and hasattr(data.get('timestamp'), 'to_date') else data.get('timestamp'),
                    'imageUrls': data.get('imageUrls') or [],
                    'likes': data.get('likes') or [],
                }
                messages.append(message)
            
            # Sort in memory if orderBy failed due to missing index
            if orderBy == "desc":
                messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            elif orderBy == "asc":
                messages.sort(key=lambda x: x.get('timestamp', ''))
            
            logger.info(f"Retrieved {len(messages)} messages from Firestore")
            return messages
            
        except Exception as e:
            logger.error(f"Error querying Firestore messages: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def send_message_property(self):
        description = f"""
        Send a message to a chat channel in Firestore for org `{self.org_slug}`.
        
        This tool sends messages that are attributed to the AI agent 'lean'. All messages sent through this tool will have userId='lean@leanworks.ai', memberName='Lean', and memberAvatar='L'.
        
        Message Structure:
        - chatId: The chat/conversation ID (required)
          - Format: 'project-{{projectId}}' for project channels
          - Format: 'ai-assistant-{{userId}}' for AI assistant conversations
          - Format: 'dm-{{email1}}-{{email2}}' for direct messages
        - content: Message text content (required)
        - projectId: Project ID if sending to project channel (optional, extracted from chatId if chatId starts with 'project-')
        - role: Message role ('user' or 'assistant', default: 'user')
        - citedContext: Optional cited context information
        - imageUrls: Optional array of image URLs
        
        Authorization:
        - For project channels: User must be a project member or owner (verified by caller)
        - For AI assistant: Already handled by existing security
        - For DMs: Both users must be in same org (verified by caller)
        
        Returns:
        - Success: Dictionary with message id and created message
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "send_message",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID (required) - format: 'project-{projectId}', 'ai-assistant-{userId}', or 'dm-{email1}-{email2}'"
                    },
                    "content": {
                        "type": "string",
                        "description": "Message text content (required)"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Project ID if sending to project channel (optional, extracted from chatId if chatId starts with 'project-')"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Message role (default: 'user')"
                    },
                    "citedContext": {
                        "type": "object",
                        "description": "Optional cited context information"
                    },
                    "imageUrls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional array of image URLs"
                    }
                },
                "required": ["chatId", "content"]
            }
        }
    
    def send_message(
        self,
        chatId: str,
        content: str,
        projectId: Optional[str] = None,
        role: str = "user",
        citedContext: Optional[Dict[str, Any]] = None,
        imageUrls: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to a chat channel.
        
        Args:
            chatId: Chat ID (required)
            content: Message text content (required)
            projectId: Project ID if sending to project channel (optional, extracted from chatId if chatId starts with 'project-')
            role: Message role (default: 'user')
            citedContext: Optional cited context information
            imageUrls: Optional array of image URLs
            
        Returns:
            Dictionary with message id and created message, or error dictionary
        """
        try:
            if not self.firestore_client or not self.org_slug:
                return {"error": "Firestore client or org_slug not initialized"}
            
            if not chatId or not content:
                return {"error": "chatId and content are required"}
            
            # Extract projectId from chatId if not provided
            actual_project_id = projectId
            
            if chatId.startswith('project-'):
                actual_project_id = chatId.replace('project-', '')
            
            # Note: Authorization checks for project membership should be done by the caller
            # For now, we'll just send the message with AI agent identity
            
            # Build message data
            message_data = {
                'chatId': chatId,
                'role': role,
                'content': content,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'userId': AI_AGENT_ID,  # Always use AI agent ID
                'projectId': actual_project_id,
                'memberName': AI_AGENT_NAME,  # Always use 'Lean'
                'memberAvatar': AI_AGENT_AVATAR,  # Always use 'L'
                'likes': [],  # Initialize likes as empty array
            }
            
            # Add citedContext if provided
            if citedContext:
                message_data['citedContext'] = citedContext
            
            # Add imageUrls if provided
            if imageUrls and isinstance(imageUrls, list) and len(imageUrls) > 0:
                message_data['imageUrls'] = imageUrls
            
            # Write to Firestore
            messages_path = f"orgs/{self.org_slug}/messages"
            write_result, doc_ref = self.firestore_client.collection(messages_path).add(message_data)
            
            # Get the created document to return
            message_doc = doc_ref.get()
            message_dict = message_doc.to_dict()
            message_dict['id'] = doc_ref.id
            message_dict['timestamp'] = message_dict.get('timestamp').to_date().isoformat() if message_dict.get('timestamp') and hasattr(message_dict.get('timestamp'), 'to_date') else message_dict.get('timestamp')
            message_dict['imageUrls'] = message_dict.get('imageUrls') or []
            message_dict['likes'] = message_dict.get('likes') or []
            
            logger.info(f"Message sent: chatId={chatId}, messageId={doc_ref.id}, userId={AI_AGENT_ID}")
            return {
                "success": True,
                "messageId": doc_ref.id,
                "message": message_dict
            }
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

