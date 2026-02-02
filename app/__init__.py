"""
Leanworks API Application
"""
import json
import logging
import os
import traceback
from quart import Quart
from quart_cors import cors
from google.cloud import firestore, secretmanager
from google.oauth2 import service_account

# Configure logging first
# Write to stdout instead of stderr so GKE doesn't treat all logs as errors
# GKE/Cloud Logging interprets stderr as errors regardless of log level
import sys
# Configure logging to write to stdout only (for GKE compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Suppress verbose loggers while keeping essential tool/response logs
logging.getLogger('leanworks.rag').setLevel(logging.WARNING)  # Covers query, chat, embedding, reranker, span_selection
logging.getLogger('leanworks.agent.tools.toolkit').setLevel(logging.WARNING)
logging.getLogger('leanworks.agent.tools.base_api_client').setLevel(logging.WARNING)
logging.getLogger('leanworks.agent.tools.internal.search').setLevel(logging.WARNING)
logging.getLogger('leanworks.agent.core.conversation').setLevel(logging.INFO)  # Keep for tool calls
logging.getLogger('leanworks.agent.core.chat').setLevel(logging.INFO)  # Keep for final response
logging.getLogger('leanworks.agent.core.memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Firebase Admin SDK for Bearer token authentication
try:
    import firebase_admin
    from firebase_admin import credentials, auth
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("Firebase Admin SDK not available. Bearer token authentication will be disabled.")

# Initialize Firebase Admin SDK if available
firebase_app = None
if FIREBASE_AVAILABLE:
    try:
        # Check if Firebase is already initialized
        try:
            firebase_app = firebase_admin.get_app()
            logger.info("Using existing Firebase Admin SDK instance")
        except ValueError:
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate("gcp_credential.json")
            firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Firebase Admin SDK: {str(e)}. Bearer token authentication will be disabled.")
        FIREBASE_AVAILABLE = False

# Initialize shared Leanworks infrastructure resources at startup
# These are NOT multi-tenant and can be shared across all requests
_leanworks_credentials = None
_project_id = None
_firestore_client = None
_secret_manager_client = None

def initialize_infrastructure():
    """Initialize shared GCP infrastructure clients"""
    global _leanworks_credentials, _project_id, _firestore_client, _secret_manager_client
    
    try:
        logger.info("Initializing shared Leanworks infrastructure resources...")
        
        # Check if credential file exists - use it if available, otherwise use ADC
        credential_file_exists = os.path.exists("gcp_credential.json")
        
        if credential_file_exists:
            # Use credential file if it exists (works in both local and Cloud Run)
            logger.info("Using service account file: gcp_credential.json")
            _leanworks_credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
            
            # Load project_id once from credentials JSON
            with open("gcp_credential.json", "r") as f:
                credential_data = json.load(f)
            _project_id = credential_data["project_id"]
        else:
            # Fallback to Application Default Credentials (ADC) if file doesn't exist
            logger.info("gcp_credential.json not found, using Application Default Credentials")
            from google.auth import default
            _leanworks_credentials, _project_id = default()
        
        # Initialize shared Firestore client (still needed for ChatAgent)
        _firestore_client = firestore.Client(
            credentials=_leanworks_credentials, 
            project=_project_id, 
            database="leanworks-prod"
        )
        
        # Initialize shared Secret Manager client
        _secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=_leanworks_credentials)
        
        # Initialize PostgreSQL database connection infrastructure
        from app.services.database import initialize_database
        initialize_database()
        
        logger.info(f"Shared Leanworks infrastructure initialized successfully (project: {_project_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize shared Leanworks infrastructure: {str(e)}")
        traceback.print_exc()
        return False

# Initialize infrastructure on import
initialize_infrastructure()

def initialize_docker_image():
    """Build Docker image for bash sessions in background."""
    import threading
    
    def build_image():
        """Background thread to build Docker image."""
        try:
            logger.info("Building Docker image for bash sessions (background)...")
            
            # Import ToolUse to access _ensure_custom_image method
            from leanworks.agent.tools.toolkit import ToolUse
            
            # Create a minimal instance just for image building
            # We only need the _ensure_custom_image method, not actual tools
            docker_builder = object.__new__(ToolUse)
            
            # Call the image build method
            image_name = ToolUse._ensure_custom_image(docker_builder)
            
            logger.info(f"Docker image ready: {image_name}")
        except Exception as e:
            # Don't fail - image will be built on first use instead
            logger.warning(f"Background Docker image build failed: {e}. Will build on first use.")
    
    # Start background thread
    thread = threading.Thread(target=build_image, daemon=True, name="DockerImageBuilder")
    thread.start()
    logger.info("Started background Docker image builder thread")

# Start background Docker image build
initialize_docker_image()

def get_leanworks_credentials():
    """Get shared Leanworks credentials"""
    return _leanworks_credentials

def get_project_id():
    """Get project ID"""
    return _project_id

def get_firestore_client():
    """Get shared Firestore client (still needed for ChatAgent)"""
    return _firestore_client

def get_secret_manager_client():
    """Get shared Secret Manager client"""
    return _secret_manager_client

def get_firebase_app():
    """Get Firebase app instance"""
    return firebase_app

def is_firebase_available():
    """Check if Firebase is available"""
    return FIREBASE_AVAILABLE

# Create Quart app
app = Quart(__name__)

# Configure CORS to allow requests from leanworks-hub
# Allow all origins in development, specific origins in production
allowed_origins = os.environ.get("CORS_ORIGINS", "https://leanworks.ai,https://hub.leanworks.ai,http://localhost:8080,http://localhost:5173").split(",")
cors(app, 
     allow_origin=allowed_origins, 
     allow_headers=["Content-Type", "Authorization", "X-API-Key"], 
     allow_methods=["GET", "POST", "OPTIONS"],
     allow_credentials=True)

