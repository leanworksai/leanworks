import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime
import json
import re
from threading import Lock

logger = logging.getLogger(__name__)

# Cache for project members (key: (org_slug, project_id))
_project_members_cache: Dict[tuple, tuple] = {}  # (org_slug, project_id) -> (members_list, timestamp)
_cache_lock = Lock()
_cache_ttl = 300  # 5 minutes

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
    
    @property
    def send_message_property(self):
        description = f"""
        Send a message to a project channel in Firestore for org `{self.org_slug}`.
        
        This tool sends messages that are attributed to the AI agent 'lean'. All messages sent through this tool will have userId='lean@leanworks.ai', memberName='Lean', and memberAvatar='L'.
        
        IMPORTANT RESTRICTIONS:
        - This tool can send messages to project channels (chatId starting with 'project-') and AI assistant chats (chatId starting with 'ai-assistant-')
        - This tool CANNOT send messages to direct messages (DMs) - chatId starting with 'dm-' is not allowed
        - For direct messages: The AI should respond in the AI assistant chat instead of using this tool
        
        Message Structure:
        - chatId: The chat/conversation ID (required)
          - Format: 'project-{{projectId}}' for project channels
          - Format: 'ai-assistant-{{userId}}' for AI assistant conversations
          - DO NOT use 'dm-{{email1}}-{{email2}}' - DMs are not allowed
        - content: Message text content (required)
        - projectId: Project ID if sending to project channel (optional, extracted from chatId if chatId starts with 'project-')
        - role: Message role ('user' or 'assistant', default: 'user')
        - citedContext: Optional cited context information
        - imageUrls: Optional array of image URLs
        
        Usage Guidelines:
        - Use this tool to send messages to project channels or AI assistant chats
        - If the user asks to send a direct message to someone, respond in the AI assistant chat explaining that you can help compose the message but cannot send DMs directly
        
        Mentions:
        - Mentions can be included naturally in the message content using the format @user@example.com (e.g., "Hey @user@example.com, can you review this?")
        - Mentions are automatically extracted from the content and validated
        - Mentions are only allowed in project channels (chatId starting with 'project-')
        - All mentioned users must be project members
        
        Authorization:
        - For project channels: User must be a project member or owner (verified by caller)
        
        Returns:
        - Success: Dictionary with {{"success": true, "messageId": "...", "message": {{...}}, "status": "Message sent successfully"}}
        - Error: Dictionary with {{"error": "error message"}}
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
                        "description": "Chat ID (required) - format: 'project-{projectId}' for project channels, or 'ai-assistant-{userId}' for AI assistant chats. DO NOT use 'dm-' format - DMs are not allowed."
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
    
    def _get_project_members(self, project_id: str) -> List[str]:
        """
        Get list of project member emails from PostgreSQL.
        Handles different project visibility settings.
        Results are cached for 5 minutes to avoid repeated database queries.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of user emails who can access the project (members, owners, or all org members if visibility='all_members')
        """
        # Check cache first
        cache_key = (self.org_slug, project_id)
        current_time = datetime.now().timestamp()
        
        with _cache_lock:
            if cache_key in _project_members_cache:
                members, timestamp = _project_members_cache[cache_key]
                if current_time - timestamp < _cache_ttl:
                    logger.debug(f"Returning cached project members for {project_id}")
                    return members
                # Cache expired, remove it
                del _project_members_cache[cache_key]
        
        try:
            # Import here to avoid circular dependencies
            from app.services.database import query_org
            
            # First, get project visibility settings
            project_query = """
                SELECT 
                    p.visibility,
                    p.visible_to_members,
                    p.owner_email
                FROM projects p
                WHERE p.id = %s
            """
            
            project_results = query_org(self.org_slug, project_query, (project_id,))
            if not project_results:
                logger.warning(f"Project {project_id} not found")
                return []
            
            project = project_results[0]
            visibility = project.get('visibility') or 'all_members'
            owner_email = project.get('owner_email', '').lower()
            visible_to_members = project.get('visible_to_members')
            
            member_emails = set()
            
            # Always include owner
            if owner_email:
                member_emails.add(owner_email)
            
            # Handle different visibility settings
            if visibility == 'all_members':
                # Get all users in the org
                users_query = "SELECT email FROM users"
                user_results = query_org(self.org_slug, users_query)
                for row in user_results:
                    email = row.get('email', '').lower()
                    if email:
                        member_emails.add(email)
            elif visibility == 'specific_members':
                # Get explicit project members
                members_query = """
                    SELECT DISTINCT pm.user_email
                    FROM project_members pm
                    WHERE pm.project_id = %s
                """
                members_results = query_org(self.org_slug, members_query, (project_id,))
                for row in members_results:
                    email = row.get('user_email', '').lower()
                    if email:
                        member_emails.add(email)
                
                # Also include users from visible_to_members if it's a JSON array
                if visible_to_members:
                    if isinstance(visible_to_members, str):
                        try:
                            visible_to_members = json.loads(visible_to_members)
                        except:
                            pass
                    if isinstance(visible_to_members, list):
                        for email in visible_to_members:
                            if email:
                                member_emails.add(email.lower())
            else:
                # Default: get explicit project members
                members_query = """
                    SELECT DISTINCT pm.user_email
                    FROM project_members pm
                    WHERE pm.project_id = %s
                """
                members_results = query_org(self.org_slug, members_query, (project_id,))
                for row in members_results:
                    email = row.get('user_email', '').lower()
                    if email:
                        member_emails.add(email)
            
            members_list = list(member_emails)
            
            # Cache the result
            with _cache_lock:
                _project_members_cache[cache_key] = (members_list, current_time)
                # Clean up old cache entries (keep cache size reasonable)
                if len(_project_members_cache) > 100:
                    # Remove oldest entries
                    sorted_entries = sorted(_project_members_cache.items(), key=lambda x: x[1][1])
                    for key in sorted_entries[:20]:
                        del _project_members_cache[key[0]]
            
            return members_list
        except Exception as e:
            logger.warning(f"Error getting project members for project {project_id}: {str(e)}")
            # If we can't query, return empty list - validation will fail which is safer
            return []
    
    def _extract_mentions_from_content(self, content: str) -> List[str]:
        """
        Extract user email mentions from message content.
        Looks for mentions in the format @user@example.com (with @ prefix).
        
        Args:
            content: Message content text
            
        Returns:
            List of unique email addresses found in mentions (without the @ prefix)
        """
        if not content:
            return []
        
        # Pattern to match mentions: @user@example.com
        # The @ at the start indicates it's a mention, followed by an email address
        # Improved pattern: more robust email validation
        # Matches: @user@example.com, @user.name@example.co.uk, etc.
        # Excludes: URLs, other @ symbols that aren't email addresses
        mention_pattern = r'@([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)'
        
        # Find all mentions in the content
        mentions = re.findall(mention_pattern, content)
        
        # Normalize to lowercase and remove duplicates
        unique_mentions = list(set([email.lower() for email in mentions if email]))
        
        return unique_mentions
    
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
            content: Message text content (required). Can include mentions in the format @user@example.com (e.g., "Hey @user@example.com, can you review this?")
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
            
            # CRITICAL: Reject DM chatIds - this tool cannot send to DMs
            if chatId.startswith('dm-'):
                return {"error": "Cannot send messages to direct message (DM) channels. For direct messages, respond in the AI assistant chat instead."}
            
            # Extract projectId from chatId if not provided
            actual_project_id = projectId
            is_project_channel = False
            
            if chatId.startswith('project-'):
                actual_project_id = chatId.replace('project-', '')
                is_project_channel = True
            elif not chatId.startswith('ai-assistant-'):
                # If chatId doesn't start with 'project-' or 'ai-assistant-', it's invalid
                return {"error": "chatId must start with 'project-' for project channels or 'ai-assistant-' for AI assistant chats. This tool cannot send messages to DMs."}
            
            # Extract mentions from content (email addresses in the message)
            mentions = self._extract_mentions_from_content(content)
            
            # Validate mentions: only allowed in project channels
            if mentions:
                if not is_project_channel:
                    return {"error": "Mentions are only allowed in project channels (chatId must start with 'project-')"}
                
                if not actual_project_id:
                    return {"error": "Cannot validate mentions: projectId is required for project channels"}
                
                # Get project members
                project_members = self._get_project_members(actual_project_id)
                
                # Validate all mentioned users are project members
                invalid_mentions = []
                for mention_email in mentions:
                    if mention_email not in project_members:
                        invalid_mentions.append(mention_email)
                
                if invalid_mentions:
                    return {
                        "error": f"The following users are not project members and cannot be mentioned: {', '.join(invalid_mentions)}"
                    }
            
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
            
            # Add mentions if found in content (already validated)
            if mentions and len(mentions) > 0:
                message_data['mentions'] = mentions
            
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
            message_dict['mentions'] = message_dict.get('mentions') or []
            
            logger.info(f"Message sent: chatId={chatId}, messageId={doc_ref.id}, userId={AI_AGENT_ID}")
            return {
                "success": True,
                "messageId": doc_ref.id,
                "message": message_dict,
                "status": "Message sent successfully"
            }
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

