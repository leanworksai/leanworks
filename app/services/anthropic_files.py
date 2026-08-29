"""
Anthropic Files API service for managing file uploads to Claude.
"""
import logging
import asyncio
from typing import Dict, List, Optional
from anthropic import APIError

logger = logging.getLogger(__name__)


class AnthropicFilesService:
    """
    Service for interacting with Claude's Files API.
    Handles file uploads, metadata retrieval, and file lifecycle management.
    """
    
    def __init__(self, model_client):
        """
        Initialize Anthropic Files Service.
        
        Args:
            model_client: Initialized Anthropic client instance
        """
        self.client = model_client
        self.beta_header = "files-api-2025-04-14"
    
    def validate_file(self, file, max_size_mb: int = 500) -> Dict:
        """
        Validate file before upload.
        
        Args:
            file: File object from request (has .filename, .content_type, .read() method)
            max_size_mb: Maximum file size in MB (default: 500 MB, Claude's limit)
            
        Returns:
            Dict with 'valid' (bool) and 'error' (str if invalid)
        """
        try:
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size_bytes = file.tell()
            file.seek(0)  # Reset to beginning
            
            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size_bytes > max_size_bytes:
                return {
                    "valid": False,
                    "error": f"File size ({file_size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
                }
            
            # Check file type
            content_type = file.content_type or ""
            filename = file.filename or ""
            
            # Supported MIME types for Claude Files API
            supported_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            supported_document_types = ["application/pdf", "text/plain"]
            supported_types = supported_image_types + supported_document_types
            
            if content_type not in supported_types:
                # Try to infer from filename extension
                file_ext = filename.lower().split('.')[-1] if '.' in filename else ""
                ext_to_mime = {
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                    'webp': 'image/webp',
                    'pdf': 'application/pdf',
                    'txt': 'text/plain'
                }
                
                inferred_mime = ext_to_mime.get(file_ext)
                if inferred_mime and inferred_mime in supported_types:
                    # Update content_type if we can infer it
                    file.content_type = inferred_mime
                    logger.info(f"Inferred MIME type {inferred_mime} from extension {file_ext}")
                else:
                    return {
                        "valid": False,
                        "error": f"File type '{content_type}' not supported. Supported types: {', '.join(supported_types)}"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            return {
                "valid": False,
                "error": f"Error validating file: {str(e)}"
            }
    
    def upload_file_sync(self, file_data: bytes, filename: str, mime_type: str) -> Dict:
        """
        Upload file to Claude Files API (synchronous version for use in executor).
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file
            
        Returns:
            Dict with file_id, filename, mime_type, size_bytes, created_at
            
        Raises:
            Exception: If upload fails
        """
        try:
            # Upload file using Anthropic Files API
            # Note: The beta header is handled automatically by the SDK when using .beta.files
            result = self.client.beta.files.upload(
                file=(filename, file_data, mime_type),
            )
            
            file_info = {
                "file_id": result.id,
                "filename": result.filename,
                "mime_type": result.mime_type,
                "size_bytes": result.size_bytes,
                "created_at": result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
            }
            
            logger.info("Successfully uploaded file to Claude Files API")
            return file_info
            
        except APIError as e:
            logger.error(f"Anthropic API error uploading file {filename}: {str(e)}")
            raise Exception(f"Failed to upload file to Claude: {str(e)}")
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {str(e)}")
            raise Exception(f"Failed to upload file: {str(e)}")
    
    async def upload_file(self, file_data: bytes, filename: str, mime_type: str) -> Dict:
        """
        Upload file to Claude Files API (async wrapper).
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file
            
        Returns:
            Dict with file_id, filename, mime_type, size_bytes, created_at
            
        Raises:
            Exception: If upload fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.upload_file_sync,
            file_data,
            filename,
            mime_type
        )
    
    def list_files_sync(self) -> List[Dict]:
        """
        List all uploaded files (synchronous version).
        
        Returns:
            List of file metadata dictionaries
        """
        try:
            files = self.client.beta.files.list()
            return [self._format_file(f) for f in files.data]
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise
    
    async def list_files(self) -> List[Dict]:
        """
        List all uploaded files (async wrapper).
        
        Returns:
            List of file metadata dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.list_files_sync)
    
    def get_file_metadata_sync(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file (synchronous version).
        
        Args:
            file_id: Claude file_id
            
        Returns:
            File metadata dictionary
        """
        try:
            file = self.client.beta.files.retrieve_metadata(file_id)
            return self._format_file(file)
        except Exception as e:
            logger.error(f"Error retrieving file metadata for {file_id}: {str(e)}")
            raise
    
    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file (async wrapper).
        
        Args:
            file_id: Claude file_id
            
        Returns:
            File metadata dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_file_metadata_sync, file_id)
    
    def delete_file_sync(self, file_id: str) -> bool:
        """
        Delete a file from Claude Files API (synchronous version).
        
        Args:
            file_id: Claude file_id
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.beta.files.delete(file_id)
            logger.info(f"Successfully deleted file: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Claude Files API (async wrapper).
        
        Args:
            file_id: Claude file_id
            
        Returns:
            True if successful, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.delete_file_sync, file_id)
    
    def _format_file(self, file) -> Dict:
        """
        Format file object from Anthropic API to dictionary.
        
        Args:
            file: File object from Anthropic API
            
        Returns:
            Formatted dictionary
        """
        return {
            "file_id": file.id,
            "filename": file.filename,
            "mime_type": file.mime_type,
            "size_bytes": file.size_bytes,
            "created_at": file.created_at.isoformat() if hasattr(file.created_at, 'isoformat') else str(file.created_at),
            "downloadable": getattr(file, 'downloadable', False)
        }
