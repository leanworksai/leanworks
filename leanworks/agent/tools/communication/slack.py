import logging
from typing import List, Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)


class SlackTool:
    """
    Slack tool for accessing and managing Slack workspaces using Slack API.

    Provides methods for reading channels, messages, posting messages, and managing
    basic Slack operations. Uses Slack bot tokens for authentication.
    """

    def __init__(self, bot_token: str):
        """
        Initialize SlackTool with bot token.

        Args:
            bot_token: Slack bot token (starts with xoxb-)
        """
        self.bot_token = bot_token
        self.base_url = "https://slack.com/api"
        self.headers = {
            'Authorization': f'Bearer {bot_token}',
            'Content-Type': 'application/json; charset=utf-8'
        }

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make authenticated request to Slack API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., 'conversations.list')
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON data or None
        """
        url = f"{self.base_url}/{endpoint}"

        # Add headers to kwargs
        if 'headers' not in kwargs:
            kwargs['headers'] = self.headers
        else:
            kwargs['headers'].update(self.headers)

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()

        if response.content:
            result = response.json()
            if not result.get('ok', False):
                error = result.get('error', 'Unknown error')
                raise Exception(f"Slack API error: {error}")
            return result
        return None

    @property
    def list_channels_property(self):
        description = """
        List all channels in the Slack workspace that the bot has access to.

        Returns information about each channel including ID, name, type (public/private),
        member count, and creation date. The bot must be a member of private channels
        to see them.

        Parameters:
        - types: Comma-separated list of channel types to include (default: "public_channel,private_channel")
        - limit: Maximum number of channels to return (default: 100, max: 1000)

        Returns:
        A list of dictionaries, each containing:
        - id: Channel ID
        - name: Channel name
        - type: "public" or "private"
        - member_count: Number of members
        - created_at: ISO timestamp when channel was created
        - is_archived: Whether the channel is archived
        """
        return {
            "type": "custom",
            "name": "slack_list_channels",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "types": {
                        "type": "string",
                        "description": "Channel types to include (comma-separated: public_channel,private_channel,mpim,im)",
                        "default": "public_channel,private_channel"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of channels to return (default: 100, max: 1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    }
                }
            }
        }

    def list_channels(self, types: str = "public_channel,private_channel", limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        List channels in the Slack workspace.

        Args:
            types: Channel types to include
            limit: Maximum number of channels to return

        Returns:
            List of channel dictionaries, or error dictionary
        """
        logger.info(f"Listing Slack channels with types: {types}, limit: {limit}")

        try:
            # Validate limit
            limit = min(max(1, limit), 1000)

            # Make API request
            response = self._make_request("GET", "conversations.list", params={
                'types': types,
                'limit': limit
            })

            channels = response.get('channels', [])

            # Format the response
            formatted_channels = []
            for channel in channels:
                formatted_channels.append({
                    "id": channel.get('id'),
                    "name": channel.get('name'),
                    "type": "private" if channel.get('is_private') else "public",
                    "member_count": channel.get('num_members'),
                    "created_at": channel.get('created'),  # Unix timestamp
                    "is_archived": channel.get('is_archived', False)
                })

            logger.info(f"Listed {len(formatted_channels)} Slack channels")
            return formatted_channels

        except Exception as e:
            logger.error(f"Error listing Slack channels: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def get_channel_messages_property(self):
        description = """
        Get messages from a Slack channel.

        Retrieves recent messages from the specified channel. The bot must be a member
        of the channel to read messages. Returns message content, timestamps, and user information.

        Parameters:
        - channel_id: The ID of the channel to read messages from
        - limit: Maximum number of messages to return (default: 100, max: 200)

        Returns:
        A list of dictionaries, each containing:
        - ts: Message timestamp
        - user: User ID who sent the message
        - text: Message text content
        - type: Message type (usually "message")
        - thread_ts: Thread timestamp if message is in a thread
        """
        return {
            "type": "custom",
            "name": "slack_get_channel_messages",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel to read messages from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default: 100, max: 200)",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 100
                    }
                },
                "required": ["channel_id"]
            }
        }

    def get_channel_messages(self, channel_id: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Get messages from a Slack channel.

        Args:
            channel_id: The ID of the channel to read messages from
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries, or error dictionary
        """
        logger.info(f"Getting messages from Slack channel: {channel_id}, limit: {limit}")

        try:
            if not channel_id:
                return {"error": "channel_id is required"}

            # Validate limit
            limit = min(max(1, limit), 200)

            # Make API request
            response = self._make_request("GET", "conversations.history", params={
                'channel': channel_id,
                'limit': limit
            })

            messages = response.get('messages', [])

            # Format the response
            formatted_messages = []
            for message in messages:
                formatted_messages.append({
                    "ts": message.get('ts'),
                    "user": message.get('user'),
                    "text": message.get('text'),
                    "type": message.get('type'),
                    "thread_ts": message.get('thread_ts')
                })

            logger.info(f"Retrieved {len(formatted_messages)} messages from channel: {channel_id}")
            return formatted_messages

        except Exception as e:
            logger.error(f"Error getting Slack channel messages: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def post_message_property(self):
        description = """
        Post a message to a Slack channel.

        Sends a message to the specified channel. The bot must be a member of the channel
        and have permission to post messages. Can optionally post as a thread reply.

        Parameters:
        - channel_id: The ID of the channel to post to
        - text: The message text to post
        - thread_ts: Optional timestamp of the message to reply to (for threads)

        Returns:
        A dictionary containing:
        - ts: Timestamp of the posted message
        - channel: Channel ID where message was posted
        - message: The message object that was posted
        """
        return {
            "type": "custom",
            "name": "slack_post_message",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel to post the message to"
                    },
                    "text": {
                        "type": "string",
                        "description": "The message text to post"
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Optional timestamp of the message to reply to (creates a thread reply)"
                    }
                },
                "required": ["channel_id", "text"]
            }
        }

    def post_message(self, channel_id: str, text: str, thread_ts: str = None, **kwargs) -> Dict[str, Any]:
        """
        Post a message to a Slack channel.

        Args:
            channel_id: The ID of the channel to post to
            text: The message text to post
            thread_ts: Optional timestamp for thread replies

        Returns:
            Message posting result dictionary, or error dictionary
        """
        logger.info(f"Posting message to Slack channel: {channel_id}")

        try:
            if not channel_id or not text:
                return {"error": "channel_id and text are required"}

            # Prepare message payload
            payload = {
                'channel': channel_id,
                'text': text
            }

            if thread_ts:
                payload['thread_ts'] = thread_ts

            # Make API request
            response = self._make_request("POST", "chat.postMessage", json=payload)

            # Format the response
            result = {
                "ts": response.get('ts'),
                "channel": response.get('channel'),
                "message": response.get('message', {})
            }

            logger.info(f"Posted message to channel: {channel_id} with timestamp: {response.get('ts')}")
            return result

        except Exception as e:
            logger.error(f"Error posting Slack message: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def get_user_info_property(self):
        description = """
        Get information about a Slack user.

        Retrieves profile information for a specified user ID. The bot must have
        permission to view user information.

        Parameters:
        - user_id: The ID of the user to get information about

        Returns:
        A dictionary containing:
        - id: User ID
        - name: Username
        - real_name: Real name
        - display_name: Display name
        - email: Email address (if available)
        - is_bot: Whether the user is a bot
        - tz: Timezone
        """
        return {
            "type": "custom",
            "name": "slack_get_user_info",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to get information about"
                    }
                },
                "required": ["user_id"]
            }
        }

    def get_user_info(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get information about a Slack user.

        Args:
            user_id: The ID of the user to get information about

        Returns:
            User information dictionary, or error dictionary
        """
        logger.info(f"Getting Slack user info for: {user_id}")

        try:
            if not user_id:
                return {"error": "user_id is required"}

            # Make API request
            response = self._make_request("GET", "users.info", params={
                'user': user_id
            })

            user = response.get('user', {})

            # Format the response
            result = {
                "id": user.get('id'),
                "name": user.get('name'),
                "real_name": user.get('real_name'),
                "display_name": user.get('profile', {}).get('display_name'),
                "email": user.get('profile', {}).get('email'),
                "is_bot": user.get('is_bot', False),
                "tz": user.get('tz')
            }

            logger.info(f"Retrieved user info for: {user.get('name')}")
            return result

        except Exception as e:
            logger.error(f"Error getting Slack user info: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def create_channel_property(self):
        description = """
        Create a new public channel in the Slack workspace.

        Creates a new public channel with the specified name. The bot must have
        permission to create channels.

        Parameters:
        - name: The name of the channel to create (without # prefix)
        - is_private: Whether to create a private channel (default: false)

        Returns:
        A dictionary containing:
        - id: Channel ID
        - name: Channel name
        - type: "public" or "private"
        - created: Creation timestamp
        """
        return {
            "type": "custom",
            "name": "slack_create_channel",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the channel to create (without # prefix)"
                    },
                    "is_private": {
                        "type": "boolean",
                        "description": "Whether to create a private channel (default: false)",
                        "default": False
                    }
                },
                "required": ["name"]
            }
        }

    def create_channel(self, name: str, is_private: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Create a new channel in Slack.

        Args:
            name: The name of the channel to create
            is_private: Whether to create a private channel

        Returns:
            Channel creation result dictionary, or error dictionary
        """
        logger.info(f"Creating Slack channel: {name}, private: {is_private}")

        try:
            if not name:
                return {"error": "name is required"}

            # Prepare payload
            payload = {
                'name': name,
                'is_private': is_private
            }

            # Make API request
            response = self._make_request("POST", "conversations.create", json=payload)

            channel = response.get('channel', {})

            # Format the response
            result = {
                "id": channel.get('id'),
                "name": channel.get('name'),
                "type": "private" if channel.get('is_private') else "public",
                "created": channel.get('created')
            }

            logger.info(f"Created Slack channel: {name} with ID: {channel.get('id')}")
            return result

        except Exception as e:
            logger.error(f"Error creating Slack channel: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}