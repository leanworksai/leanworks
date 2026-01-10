"""
Unit tests for Anthropic Files API service.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.services.anthropic_files import AnthropicFilesService


class TestAnthropicFilesService:
    """Test suite for AnthropicFilesService"""
    
    def test_validate_file_valid_image(self):
        """Test file validation with valid image file"""
        service = AnthropicFilesService(Mock())
        
        # Create mock file object
        file = Mock()
        file.filename = "test_image.png"
        file.content_type = "image/png"
        file.tell.return_value = 1024 * 1024  # 1 MB
        file.seek = Mock()
        
        result = service.validate_file(file, max_size_mb=500)
        assert result["valid"] is True
    
    def test_validate_file_valid_pdf(self):
        """Test file validation with valid PDF file"""
        service = AnthropicFilesService(Mock())
        
        file = Mock()
        file.filename = "document.pdf"
        file.content_type = "application/pdf"
        file.tell.return_value = 5 * 1024 * 1024  # 5 MB
        file.seek = Mock()
        
        result = service.validate_file(file, max_size_mb=500)
        assert result["valid"] is True
    
    def test_validate_file_too_large(self):
        """Test file validation with file exceeding size limit"""
        service = AnthropicFilesService(Mock())
        
        file = Mock()
        file.filename = "large_file.pdf"
        file.content_type = "application/pdf"
        file.tell.return_value = 600 * 1024 * 1024  # 600 MB (exceeds 500 MB limit)
        file.seek = Mock()
        
        result = service.validate_file(file, max_size_mb=500)
        assert result["valid"] is False
        assert "exceeds maximum" in result["error"].lower()
    
    def test_validate_file_invalid_type(self):
        """Test file validation with unsupported file type"""
        service = AnthropicFilesService(Mock())
        
        file = Mock()
        file.filename = "document.docx"
        file.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        file.tell.return_value = 1024 * 1024  # 1 MB
        file.seek = Mock()
        
        result = service.validate_file(file, max_size_mb=500)
        assert result["valid"] is False
        assert "not supported" in result["error"].lower()
    
    def test_validate_file_infers_mime_from_extension(self):
        """Test that MIME type is inferred from file extension when missing"""
        service = AnthropicFilesService(Mock())
        
        file = Mock()
        file.filename = "image.jpg"
        file.content_type = None  # Missing content type
        file.tell.return_value = 1024 * 1024
        file.seek = Mock()
        
        result = service.validate_file(file, max_size_mb=500)
        # Should infer image/jpeg from .jpg extension
        assert result["valid"] is True
        assert file.content_type == "image/jpeg"
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self):
        """Test successful file upload"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_result = Mock()
        mock_result.id = "file_123456"
        mock_result.filename = "test.pdf"
        mock_result.mime_type = "application/pdf"
        mock_result.size_bytes = 1024
        mock_result.created_at = "2025-01-01T00:00:00Z"
        
        mock_client.beta.files.upload.return_value = mock_result
        
        service = AnthropicFilesService(mock_client)
        
        file_data = b"test file content"
        result = await service.upload_file(file_data, "test.pdf", "application/pdf")
        
        assert result["file_id"] == "file_123456"
        assert result["filename"] == "test.pdf"
        assert result["mime_type"] == "application/pdf"
        assert result["size_bytes"] == 1024
        mock_client.beta.files.upload.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_api_error(self):
        """Test file upload with API error"""
        from anthropic import APIError
        
        mock_client = Mock()
        mock_client.beta.files.upload.side_effect = APIError(
            message="API error",
            response=Mock(),
            body=None
        )
        
        service = AnthropicFilesService(mock_client)
        
        with pytest.raises(Exception) as exc_info:
            await service.upload_file(b"test", "test.pdf", "application/pdf")
        
        assert "Failed to upload file" in str(exc_info.value)
    
    def test_format_file(self):
        """Test file formatting"""
        service = AnthropicFilesService(Mock())
        
        mock_file = Mock()
        mock_file.id = "file_123"
        mock_file.filename = "test.pdf"
        mock_file.mime_type = "application/pdf"
        mock_file.size_bytes = 1024
        mock_file.created_at = "2025-01-01T00:00:00Z"
        mock_file.downloadable = False
        
        result = service._format_file(mock_file)
        
        assert result["file_id"] == "file_123"
        assert result["filename"] == "test.pdf"
        assert result["mime_type"] == "application/pdf"
        assert result["size_bytes"] == 1024
        assert result["downloadable"] is False
