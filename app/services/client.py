"""
Client service for managing client information and tools
"""
import os
import logging
import traceback
import time
from typing import Tuple, Optional
from google.cloud import firestore, secretmanager
from anthropic import Anthropic
from app import get_firestore_client, get_secret_manager_client, get_project_id
from app.services.database import query_org
from app.utils.cache import get_cache, set_cache

logger = logging.getLogger(__name__)


def get_available_tools_from_postgres(org_slug: str) -> list:
    """Get available tools from integrations table in PostgreSQL for a given organization.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        
    Returns:
        List of available tool names (normalized to lowercase, using integration_id)
    """
    cache_key = f"available_tools:{org_slug}"
    cached_tools = get_cache(cache_key)
    
    if cached_tools is not None:
        return cached_tools
    
    try:
        # Query integrations table in the org's database
        integrations = query_org(
            org_slug,
            "SELECT integration_id, integration_name, connected FROM integrations WHERE connected = true"
        )
        
        available_tools = []
        for integration in integrations:
            # Get the tool/integration ID (prioritize integration_id over integration_name)
            tool_name = integration.get("integration_id") or integration.get("integration_name")
            if tool_name:
                # Normalize to lowercase to match leanworks package expectations
                available_tools.append(tool_name.lower())
        
        # If no tools found, fall back to environment variable
        if not available_tools:
            logger.info(f"No integrations found in PostgreSQL for org slug {org_slug}, falling back to environment variable")
            available_tools_str = os.environ.get("AVAILABLE_TOOLS", "")
            available_tools = available_tools_str.split(",") if available_tools_str else []
            available_tools = [tool.strip().lower() for tool in available_tools if tool.strip()]
        else:
            logger.info(f"Found {len(available_tools)} available tools from PostgreSQL for org slug {org_slug}: {available_tools}")
        
        # Cache the result
        set_cache(cache_key, available_tools)
        return available_tools
        
    except Exception as e:
        logger.error(f"Error querying integrations from PostgreSQL for org slug {org_slug}: {str(e)}")
        traceback.print_exc()
        # Fall back to environment variable on error
        available_tools_str = os.environ.get("AVAILABLE_TOOLS", "")
        available_tools = available_tools_str.split(",") if available_tools_str else []
        available_tools = [tool.strip().lower() for tool in available_tools if tool.strip()]
        return available_tools


def get_client_info(org_slug: str) -> Tuple[Optional[str], list]:
    """Get client info and available tools from PostgreSQL integrations table.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        
    Returns:
        Tuple of (client_name, available_tools)
    """
    try:
        # Client name is the slug (already sanitized)
        client_name = org_slug
        
        # Get available tools from PostgreSQL integrations table
        available_tools = get_available_tools_from_postgres(org_slug)
        
        return (client_name, available_tools)
    except Exception as e:
        logger.error(f"Error getting client info for org slug {org_slug}: {str(e)}")
        traceback.print_exc()
        return None, []


def get_cached_api_key(secret_name: str) -> Optional[str]:
    """Get API key with caching to reduce secret manager calls. Uses shared Secret Manager client."""
    cache_key = f"api_key:{secret_name}"
    cached_key = get_cache(cache_key)
    
    if cached_key:
        return cached_key
    
    secret_manager_client = get_secret_manager_client()
    project_id = get_project_id()
    
    if not secret_manager_client or not project_id:
        logger.error("Shared Secret Manager client or project_id not initialized")
        return None
    
    try:
        full_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = secret_manager_client.access_secret_version(name=full_name)
        api_key = response.payload.data.decode("UTF-8")
        
        if api_key:
            set_cache(cache_key, api_key)
        return api_key
    except Exception as e:
        logger.error(f"Error getting API key {secret_name}: {str(e)}")
        traceback.print_exc()
        return None


def get_cached_storage_client(client_name: str):
    """Get storage client with caching to reduce initialization overhead. Uses shared credentials."""
    from app.services.storage import CloudStorage
    from app import get_leanworks_credentials
    
    cache_key = f"storage:{client_name}"
    cached_client = get_cache(cache_key)
    
    if cached_client:
        return cached_client
    
    leanworks_credentials = get_leanworks_credentials()
    if not leanworks_credentials:
        logger.error("Shared credentials not initialized")
        raise RuntimeError("Shared credentials not initialized")
    
    try:
        storage_client = CloudStorage(credentials=leanworks_credentials, bucket="leanworks-prod")
        set_cache(cache_key, storage_client)
        return storage_client
    except Exception as e:
        logger.error(f"Error initializing storage client for client {client_name}: {str(e)}")
        traceback.print_exc()
        raise


async def initialize_clients_async(user_id: str, org_slug: str) -> Tuple[firestore.Client, secretmanager.SecretManagerServiceClient, Anthropic, list]:
    """Initialize all required clients asynchronously using shared infrastructure clients
    
    Args:
        user_id: User email address
        org_slug: Organization slug (e.g., 'leanworks')
    
    Returns:
        Tuple of (firestore_client, secret_manager_client, model_client, available_tools)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Initializing clients for user {user_id} in org slug {org_slug}")
        
        # Check if shared clients are initialized
        firestore_client = get_firestore_client()
        secret_manager_client = get_secret_manager_client()
        
        if not firestore_client or not secret_manager_client:
            raise RuntimeError("Shared Firestore or Secret Manager clients not initialized")
        
        # Get client info (queries PostgreSQL integrations table using org_slug)
        client_name, available_tools = get_client_info(org_slug)
        
        if not client_name:
            raise ValueError(f"Could not determine client for org slug: {org_slug}")
        
        # Get Claude API key using shared client
        # Try both naming conventions for Claude API key
        claude_api_key = None
        for secret_name in ["claude-api-key", "CLAUDE_API_KEY"]:
            try:
                claude_api_key = get_cached_api_key(secret_name)
                if claude_api_key:
                    break
            except Exception:
                continue
        
        if not claude_api_key:
            raise ValueError(f"claude-api-key or CLAUDE_API_KEY not found for org slug: {org_slug}")
        
        # Initialize Anthropic client (lightweight, per-request)
        model_client = Anthropic(api_key=claude_api_key)
        
        init_time = time.time() - start_time
        logger.info(f"Client initialization completed in {init_time:.3f}s for user: {user_id} in org slug: {org_slug}")
        
        return firestore_client, secret_manager_client, model_client, available_tools
    except Exception as e:
        logger.error(f"Error in initialize_clients_async for user {user_id} in org slug {org_slug}: {str(e)}")
        traceback.print_exc()
        raise

