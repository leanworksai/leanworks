import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime

logger = logging.getLogger(__name__)


class FirestoreTool:
    """
    Firestore tool for querying messages and other Firestore collections.
    
    Provides read-only access to Firestore collections, primarily for accessing
    messages stored in orgs/{orgSlug}/messages collection.
    """
    
    def __init__(self, firestore_client, org_slug: str, user_id: str = None, secret_manager_client=None):
        """
        Initialize FirestoreTool with Firestore client and org context.
        
        Args:
            firestore_client: Firestore client instance
            org_slug: Organization slug (e.g., 'leanworks.ai')
            user_id: Optional user ID for filtering user-specific data
            secret_manager_client: Optional Secret Manager client for database access
        """
        self.firestore_client = firestore_client
        self.org_slug = org_slug
        self.user_id = user_id
        self.secret_manager_client = secret_manager_client
    
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
    

