import logging
from typing import List, Dict, Any, Optional
from google.cloud import storage
from datetime import datetime, timedelta
import json
import re

logger = logging.getLogger(__name__)


class CloudStorageTool:
    """
    Cloud Storage tool for accessing images stored in Google Cloud Storage.
    
    Images are stored at: orgs/{orgSlug}/chat-images/{chatId}/{imageId}.jpg
    Provides signed URLs for accessing private images.
    """
    
    def __init__(self, storage_client, org_slug: str, bucket_name: str = "leanworks-prod", credential_path: str = "gcp_credential.json"):
        """
        Initialize CloudStorageTool with Storage client and org context.
        
        Args:
            storage_client: Google Cloud Storage client instance
            org_slug: Organization slug (e.g., 'leanworks.ai')
            bucket_name: GCS bucket name (default: 'leanworks-prod')
            credential_path: Path to GCP credential JSON file
        """
        self.storage_client = storage_client
        self.org_slug = org_slug
        self.bucket_name = bucket_name
        self.credential_path = credential_path
        
        # Image URL expiration time (default: 1 year)
        self.image_url_expiration_days = 365
    
    def _get_bucket(self):
        """Get or create the storage bucket."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            # Check if bucket exists
            if not bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist")
            return bucket
        except Exception as e:
            logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
            raise
    
    def _generate_signed_url(self, storage_path: str, expiration_days: int = None) -> str:
        """
        Generate a signed URL for accessing a file in Cloud Storage.
        
        Args:
            storage_path: Path to the file in the bucket
            expiration_days: Number of days until URL expires (default: 1 year)
            
        Returns:
            Signed URL string
        """
        try:
            expiration_days = expiration_days or self.image_url_expiration_days
            bucket = self._get_bucket()
            blob = bucket.blob(storage_path)
            
            # Calculate expiration time
            expires_in = expiration_days * 24 * 60 * 60 * 1000  # Convert to milliseconds
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)
            
            # Generate signed URL
            signed_url = blob.generate_signed_url(
                expiration=expires_at,
                method='GET'
            )
            
            return signed_url
        except Exception as e:
            logger.error(f"Error generating signed URL for {storage_path}: {e}")
            raise
    
    def _extract_storage_path(self, image_url_or_id: str, chat_id: str) -> Optional[str]:
        """
        Extract storage path from image URL or ID.
        
        Args:
            image_url_or_id: Either a signed URL or an image ID (UUID.jpg)
            chat_id: Chat ID for constructing path if image_id is provided
            
        Returns:
            Storage path or None if cannot be determined
        """
        # If it's already a storage path, return it
        if image_url_or_id.startswith('orgs/'):
            return image_url_or_id
        
        # If it's an imageId (UUID.jpg), construct the path
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$', image_url_or_id, re.IGNORECASE):
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"
        
        # Try to extract from signed URL
        try:
            from urllib.parse import urlparse
            url = urlparse(image_url_or_id)
            # Extract path from Google Cloud Storage signed URL
            # Format: https://storage.googleapis.com/bucket/path?signature=...
            if 'storage.googleapis.com' in url.netloc:
                path_match = url.path.split('/', 2)
                if len(path_match) >= 3:
                    return path_match[2]  # Path after /bucket/
        except Exception as e:
            logger.warning(f"Could not extract path from URL {image_url_or_id}: {e}")
        
        # Fallback: construct from chat_id
        if chat_id:
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"
        
        return None
    
    @property
    def get_image_url_property(self):
        description = f"""
        Get a signed URL for accessing an image from Cloud Storage for org `{self.org_slug}`.
        
        Images are stored in Google Cloud Storage at:
        orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg
        
        This tool generates signed URLs that provide temporary access to private images.
        Signed URLs expire after 1 year by default.
        
        Parameters:
        - imageId: The image ID (UUID.jpg format) or a signed URL that needs refreshing
        - chatId: The chat ID where the image was uploaded (required if imageId is provided)
        - expirationDays: Optional number of days until URL expires (default: 365)
        
        Examples:
        - Get URL for an image: {{"imageId": "123e4567-e89b-12d3-a456-426614174000.jpg", "chatId": "ai-assistant-user@example.com"}}
        - Refresh an expired URL: {{"imageId": "https://storage.googleapis.com/...", "chatId": "project-123"}}
        
        Returns:
        A dictionary with:
        - imageUrl: The signed URL for accessing the image
        - imageId: The image ID
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "get_image_url",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "imageId": {
                        "type": "string",
                        "description": "Image ID (UUID.jpg format) or existing signed URL that needs refreshing"
                    },
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID where the image was uploaded (required if imageId is a UUID)"
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URL expires (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["imageId", "chatId"]
            }
        }
    
    def get_image_url(
        self,
        imageId: str,
        chatId: str,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get a signed URL for accessing an image.
        
        Args:
            imageId: Image ID (UUID.jpg) or existing signed URL
            chatId: Chat ID where image was uploaded
            expirationDays: Optional expiration days (default: 365)
            
        Returns:
            Dictionary with imageUrl, imageId, and expiresAt, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}
            
            if not imageId or not chatId:
                return {"error": "imageId and chatId are required"}
            
            # Extract storage path
            storage_path = self._extract_storage_path(imageId, chatId)
            if not storage_path:
                return {"error": f"Could not determine storage path for imageId: {imageId}"}
            
            # Generate signed URL
            expiration_days = expirationDays or self.image_url_expiration_days
            signed_url = self._generate_signed_url(storage_path, expiration_days)
            
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)
            
            # Extract image ID from path if needed
            actual_image_id = storage_path.split('/')[-1]
            
            logger.info(f"Generated signed URL for image: {actual_image_id} in chat: {chatId}")
            
            return {
                "imageUrl": signed_url,
                "imageId": actual_image_id,
                "expiresAt": expires_at.isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Error getting image URL: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_chat_images_property(self):
        description = f"""
        List all images in a chat from Cloud Storage for org `{self.org_slug}`.
        
        Lists all images stored for a specific chat ID and returns signed URLs for accessing them.
        Images are stored at: orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg
        
        Parameters:
        - chatId: The chat ID to list images for
        - limit: Maximum number of images to return (default: 50, max: 200)
        - expirationDays: Number of days until URLs expire (default: 365)
        
        Returns:
        A list of dictionaries, each containing:
        - imageUrl: Signed URL for accessing the image
        - imageId: The image ID (filename)
        - uploadedAt: ISO timestamp when the image was uploaded (if available)
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "list_chat_images",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID to list images for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of images to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URLs expire (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["chatId"]
            }
        }
    
    def list_chat_images(
        self,
        chatId: str,
        limit: int = 50,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List all images in a chat.
        
        Args:
            chatId: Chat ID to list images for
            limit: Maximum number of images (default: 50, max: 200)
            expirationDays: Optional expiration days (default: 365)
            
        Returns:
            List of image dictionaries, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}
            
            if not chatId:
                return {"error": "chatId is required"}
            
            # Validate limit
            limit = min(max(1, limit), 200)
            
            # Build storage path prefix
            storage_prefix = f"orgs/{self.org_slug}/chat-images/{chatId}/"
            
            # List blobs in the prefix
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=storage_prefix, max_results=limit)
            
            images = []
            expiration_days = expirationDays or self.image_url_expiration_days
            
            for blob in blobs:
                # Skip if not an image file
                if not blob.name.endswith('.jpg'):
                    continue
                
                # Generate signed URL
                try:
                    signed_url = self._generate_signed_url(blob.name, expiration_days)
                    expires_at = datetime.utcnow() + timedelta(days=expiration_days)
                    
                    # Get upload time from blob metadata
                    uploaded_at = None
                    if blob.time_created:
                        uploaded_at = blob.time_created.isoformat() + "Z"
                    
                    image_id = blob.name.split('/')[-1]
                    
                    images.append({
                        "imageUrl": signed_url,
                        "imageId": image_id,
                        "uploadedAt": uploaded_at,
                        "expiresAt": expires_at.isoformat() + "Z"
                    })
                except Exception as e:
                    logger.warning(f"Error generating URL for {blob.name}: {e}")
                    continue
            
            logger.info(f"Listed {len(images)} images for chat: {chatId}")
            return images
            
        except Exception as e:
            logger.error(f"Error listing chat images: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

