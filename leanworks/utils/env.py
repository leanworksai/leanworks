import json
import logging
import os
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "leanworks-474204"
DEFAULT_REGION = "us-west1"
DEFAULT_RESOURCE_NAME = "leanworks-prod"
DEV_RESOURCE_NAME = "leanworks-dev"


def get_environment() -> str:
    """Resolve environment name: local, dev, or prod."""
    explicit_env = os.environ.get("ENVIRONMENT") or os.environ.get("LEANWORKS_ENV")
    if explicit_env:
        normalized = explicit_env.strip().lower()
        if normalized in ("prod", "production"):
            return "prod"
        if normalized in ("dev", "staging"):
            return "dev"
        if normalized in ("local", "development"):
            return "local"
    if os.environ.get("NODE_ENV") == "development" or not os.environ.get("DB_HOST"):
        return "local"
    return "prod"


def is_local_dev() -> bool:
    """Return True if running in local development mode."""
    return get_environment() == "local"


def is_dev_environment() -> bool:
    """Return True if running in shared dev environment."""
    return get_environment() == "dev"


def is_prod_environment() -> bool:
    """Return True if running in production environment."""
    return get_environment() == "prod"


def is_local_or_dev() -> bool:
    """Return True if running in local or shared dev environment."""
    return get_environment() in ("local", "dev")


def validate_environment_config() -> Tuple[bool, List[str]]:
    """Validate environment configuration and return (is_valid, errors)."""
    errors = []
    env = get_environment()

    if env == "local":
        # Local should have dev credentials (no prefix in filename, but uses dev resources)
        if not os.path.exists("gcp_credential_dev.json"):
            errors.append("Local environment requires gcp_credential_dev.json")
    elif env == "dev":
        # Dev should have dev credentials or ADC
        if not os.path.exists("gcp_credential_dev.json"):
            logger.warning("Dev environment: gcp_credential_dev.json not found, will use ADC")
    elif env == "prod":
        # Prod uses base credential file name (no prefix/suffix)
        if not os.path.exists("gcp_credential.json"):
            logger.warning("Prod environment: gcp_credential.json not found, will use ADC")

    return len(errors) == 0, errors


def get_db_instance_name() -> str:
    """Get database instance name for current environment."""
    if is_local_or_dev():
        return DEV_RESOURCE_NAME
    return os.environ.get("DB_INSTANCE_NAME") or DEFAULT_RESOURCE_NAME


def get_db_name() -> str:
    """Get database name for current environment."""
    if is_local_or_dev():
        return DEV_RESOURCE_NAME
    return os.environ.get("DB_NAME") or DEFAULT_RESOURCE_NAME


def get_storage_bucket() -> str:
    """Get Cloud Storage bucket name for current environment."""
    if is_local_or_dev():
        return DEV_RESOURCE_NAME
    return os.environ.get("AUDIO_STORAGE_BUCKET") or DEFAULT_RESOURCE_NAME


def get_firestore_database_name() -> str:
    """Get Firestore database name for current environment."""
    if is_local_or_dev():
        return DEV_RESOURCE_NAME
    return os.environ.get("FIRESTORE_DATABASE_NAME") or DEFAULT_RESOURCE_NAME


def get_secret_name(base_name: str) -> str:
    """Prefix secret names for local or dev environments."""
    if is_local_or_dev():
        return f"dev-{base_name}"
    return base_name


def get_credential_path() -> str:
    """Get default credential file path for current environment."""
    if is_local_or_dev():
        return "gcp_credential_dev.json"
    return "gcp_credential.json"


def get_google_application_credentials() -> str:
    """Resolve GOOGLE_APPLICATION_CREDENTIALS with local/dev preference."""
    dev_path = "gcp_credential_dev.json"
    prod_path = "gcp_credential.json"
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if is_local_or_dev() and os.path.exists(dev_path):
        return dev_path
    if env_path:
        return env_path
    if os.path.exists(prod_path):
        return prod_path
    return dev_path if is_local_or_dev() else prod_path


def get_hub_url() -> str:
    """Resolve Leanworks Hub base URL based on environment."""
    env_url = os.environ.get("LEANWORKS_HUB_URL")
    if env_url:
        return env_url

    # Prefer in-cluster service when running on Kubernetes
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "http://leanworks-hub-service"

    env = get_environment()
    if env == "dev":
        return "https://dev.leanworks.ai"
    if env == "prod":
        return "https://hub.leanworks.ai"

    return "http://localhost:3001"


def resolve_credential_path() -> str:
    """Resolve credential path with a safe fallback when dev file is missing."""
    return get_google_application_credentials()


def _read_project_id_from_file(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            credential_data = json.load(f)
        return credential_data.get("project_id")
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read project_id from {path}: {e}")
        return None


def get_project_id(credential_path: Optional[str] = None) -> Optional[str]:
    """Get project_id from env or credentials file."""
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if env_project:
        return env_project
    path = credential_path or resolve_credential_path()
    if path and os.path.exists(path):
        return _read_project_id_from_file(path)
    return None


def get_cloud_sql_connection_name(project_id: Optional[str] = None, region: Optional[str] = None) -> str:
    """Get Cloud SQL connection name."""
    env_connection = os.environ.get("CLOUD_SQL_CONNECTION_NAME")
    if env_connection:
        return env_connection
    resolved_project_id = project_id or get_project_id() or DEFAULT_PROJECT_ID
    resolved_region = region or os.environ.get("DB_REGION") or DEFAULT_REGION
    instance_name = get_db_instance_name()
    return f"{resolved_project_id}:{resolved_region}:{instance_name}"


def get_cloud_sql_socket_path(project_id: Optional[str] = None, region: Optional[str] = None) -> str:
    """Get Cloud SQL Unix socket directory path."""
    return f"/cloudsql/{get_cloud_sql_connection_name(project_id, region)}"
