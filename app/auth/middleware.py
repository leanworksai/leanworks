"""
Authentication middleware for API endpoints
"""
import json
import functools
import logging
import os
from quart import request, Response
from typing import Optional, Dict
from app import (
    get_firestore_client,
    get_secret_manager_client, 
    get_firebase_app, 
    is_firebase_available
)
from app.services.client import get_client_info, get_cached_api_key

logger = logging.getLogger(__name__)

# Check if API verification should be disabled (for development only)
VERIFY_API_KEY = os.environ.get("VERIFY_API_KEY", "true").lower() != "false"


async def verify_bearer_token(token: str) -> Optional[Dict]:
    """Verify Firebase Bearer token and return decoded token info"""
    if not is_firebase_available():
        return None
    
    firebase_app = get_firebase_app()
    if not firebase_app:
        return None
    
    try:
        from firebase_admin import auth
        # Try to verify as ID token first (normal flow)
        try:
            decoded_token = auth.verify_id_token(token)
            return decoded_token
        except Exception as id_token_error:
            # If ID token verification fails, try to verify as custom token
            # by decoding and checking the UID
            if 'custom token' in str(id_token_error).lower() or 'argument-error' in str(id_token_error).lower():
                try:
                    # Decode the JWT without verification first to get the UID
                    import jwt
                    parts = token.split('.')
                    if len(parts) == 3:
                        payload = jwt.decode(token, options={"verify_signature": False})
                        
                        # If it has a uid, it's likely a custom token
                        if payload.get('uid'):
                            # Verify the user exists and get their info
                            user_record = auth.get_user(payload['uid'])
                            
                            # Create a decoded token-like object
                            decoded_token = {
                                'uid': user_record.uid,
                                'email': user_record.email,
                                'email_verified': user_record.email_verified,
                            }
                            return decoded_token
                except Exception as custom_token_error:
                    logger.debug(
                        "Custom token handling failed (error_type=%s)",
                        type(custom_token_error).__name__,
                    )
                    return None
            return None
    except Exception as e:
        logger.debug(
            "Bearer token verification failed (error_type=%s)",
            type(e).__name__,
        )
        return None


def require_api_key(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Skip verification if VERIFY_API_KEY is false
        if not VERIFY_API_KEY:
            logger.warning("API key verification is disabled. This should only be used in development.")
            return await func(*args, **kwargs)
        
        # Check for Bearer token first (for leanworks-hub web)
        auth_header = request.headers.get('Authorization', '')
        bearer_token = None
        if auth_header.startswith('Bearer '):
            bearer_token = auth_header[7:]
        
        # Get API key from header (for existing clients)
        api_key = request.headers.get('X-API-Key')
        
        # Try Bearer token authentication first
        authenticated = False
        user_email = None
        
        if bearer_token and is_firebase_available():
            try:
                decoded_token = await verify_bearer_token(bearer_token)
                if decoded_token and decoded_token.get('email'):
                    authenticated = True
                    user_email = decoded_token['email'].lower()
                    logger.info(f"Bearer token authentication successful for user: {user_email}")
            except Exception as e:
                logger.debug(
                    "Bearer token authentication failed (error_type=%s)",
                    type(e).__name__,
                )
        
        # If Bearer token auth failed, try API key authentication
        if not authenticated:
            # For now, we'll get the API key per request since it depends on client_name
            # In a production setup, you might want to cache this or handle it differently
            data = await request.get_json()
            if data and data.get("user_id") and data.get("org_slug"):
                try:
                    user_id = data.get("user_id")
                    org_slug = data.get("org_slug")
                    if not user_id:
                        logger.error("user_id is None or empty")
                        return Response(response=json.dumps({"error": "user_id required"}), status=400, mimetype='application/json')
                    if not org_slug:
                        logger.error("org_slug is None or empty")
                        return Response(response=json.dumps({"error": "org_slug required"}), status=400, mimetype='application/json')
                    
                    logger.info(f"Getting client name for org slug: {org_slug}")
                    try:
                        client_name, _ = get_client_info(org_slug)
                        
                        if not client_name:
                            logger.error(f"Could not determine client for org slug: {org_slug}")
                            return Response(response=json.dumps({"error": "Could not determine client. Please check if the org_slug is valid and has proper configuration."}), status=400, mimetype='application/json')
                    except Exception as e:
                        logger.error(f"Error getting client info for org slug {org_slug}: {str(e)}")
                        return Response(response=json.dumps({"error": f"Error determining client: {str(e)}"}), status=500, mimetype='application/json')
                    
                    logger.info(f"Got client name: {client_name}")
                    try:
                        # Use shared Secret Manager client for API key retrieval
                        secret_manager_client = get_secret_manager_client()
                        if not secret_manager_client:
                            logger.error("Shared Secret Manager client not initialized")
                            return Response(response=json.dumps({"error": "Service configuration error"}), status=500, mimetype='application/json')
                        
                        # Try both naming conventions for API key
                        expected_api_key = None
                        for secret_name in ["api-key", "API_KEY"]:
                            try:
                                expected_api_key = get_cached_api_key(secret_name)
                                if expected_api_key:
                                    break
                            except Exception:
                                continue
                        
                        if not expected_api_key:
                            logger.error(f"API_KEY not found for client: {client_name}")
                            return Response(response=json.dumps({"error": f"API key not configured for client: {client_name}"}), status=500, mimetype='application/json')
                    except Exception as e:
                        logger.error(f"Error getting API key for client {client_name}: {str(e)}")
                        return Response(response=json.dumps({"error": f"Error retrieving API key: {str(e)}"}), status=500, mimetype='application/json')
                    
                    # Check if API key is valid
                    if api_key and api_key == expected_api_key:
                        authenticated = True
                        user_email = user_id.lower()
                        logger.info(f"API key authentication successful for user: {user_email}")
                    else:
                        logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
                        return Response(response=json.dumps({"error": "Unauthorized"}), status=401, mimetype='application/json')
                except Exception as e:
                    logger.error(f"Error validating API key: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return Response(response=json.dumps({"error": "Authentication error"}), status=500, mimetype='application/json')
            else:
                logger.warning("No user_id/org_slug provided for API key validation and no Bearer token")
                return Response(response=json.dumps({"error": "Authentication required. Provide either Bearer token or X-API-Key with user_id and org_slug in request body."}), status=401, mimetype='application/json')
        
        if not authenticated:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return Response(response=json.dumps({"error": "Unauthorized"}), status=401, mimetype='application/json')
        
        # Store authenticated user email in request context for use in the endpoint
        request.authenticated_user_email = user_email
        
        return await func(*args, **kwargs)
    return wrapper
