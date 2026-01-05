"""
Custom CloudStorage implementation for logging to Google Cloud Storage.
Replaces the removed leanworks.storage.CloudStorage class.
"""
import json
import logging
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class CloudStorage:
    """
    CloudStorage client for interacting with Google Cloud Storage.
    Provides methods for uploading and downloading blobs from GCS buckets.
    """
    
    def __init__(self, credentials_file: str = None, bucket: str = None, credentials=None):
        """
        Initialize CloudStorage client.
        
        Args:
            credentials_file (str, optional): Path to GCP service account credentials JSON file
            bucket (str): Name of the GCS bucket to use (required)
            credentials (optional): Pre-loaded service account credentials object
        """
        if not bucket:
            raise ValueError("bucket parameter is required")
        
        try:
            # Use provided credentials or load from file
            if credentials is not None:
                creds = credentials
            elif credentials_file:
                creds = service_account.Credentials.from_service_account_file(credentials_file)
            else:
                raise ValueError("Either credentials_file or credentials must be provided")
            
            # Initialize GCS client with credentials
            self.client = storage.Client(credentials=creds)
            
            # Get bucket reference
            self.bucket = self.client.bucket(bucket)
            self.bucket_name = bucket
            
            logger.info(f"CloudStorage initialized for bucket: {bucket}")
        except Exception as e:
            logger.error(f"Failed to initialize CloudStorage: {str(e)}")
            raise
    
    def download_blob_to_memory(self, blob_name: str) -> str:
        """
        Download a blob from GCS to memory and return as string.
        
        Args:
            blob_name (str): Name/path of the blob in the bucket
            
        Returns:
            str: Contents of the blob as a string
            
        Raises:
            Exception: If blob doesn't exist or download fails
        """
        try:
            blob = self.bucket.blob(blob_name)
            
            # Download blob content as bytes, then decode to string
            # This will raise google.cloud.exceptions.NotFound if blob doesn't exist
            content = blob.download_as_bytes()
            return content.decode('utf-8')
        except Exception as e:
            logger.debug(f"Error downloading blob {blob_name}: {str(e)}")
            raise
    
    def upload_blob_from_memory(self, contents: str, destination_blob_name: str):
        """
        Upload contents from memory to a blob in GCS.
        
        Args:
            contents (str): String contents to upload
            destination_blob_name (str): Name/path of the destination blob in the bucket
            
        Raises:
            Exception: If upload fails
        """
        try:
            blob = self.bucket.blob(destination_blob_name)
            
            # Upload string contents as bytes
            blob.upload_from_string(contents.encode('utf-8'), content_type='application/json')
            
            logger.debug(f"Successfully uploaded blob {destination_blob_name} to bucket {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error uploading blob {destination_blob_name}: {str(e)}")
            raise

