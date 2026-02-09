"""Session manager backend for bash sessions in Kubernetes."""
import logging
import os
import tempfile
from typing import Dict, Any, Optional
import uuid

try:
    import httpx
except ImportError:
    httpx = None

from .bash_backend import BashSessionBackend, BashSession

logger = logging.getLogger(__name__)


class SessionManagerBackend(BashSessionBackend):
    """Session manager backend that communicates with session manager service."""
    
    def __init__(
        self,
        manager_url: str = "http://bash-session-manager-service:8080",
        timeout: int = 30
    ):
        """
        Initialize session manager backend.
        
        Args:
            manager_url: URL to session manager service
            timeout: HTTP request timeout in seconds
        """
        if httpx is None:
            raise ImportError("httpx library is required for SessionManagerBackend")
        
        self.manager_url = manager_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        logger.info(f"SessionManagerBackend initialized with manager URL: {self.manager_url}")
    
    def create_session(self, session_id: str) -> BashSession:
        """Create bash session via session manager service."""
        try:
            # Generate unique session ID if not provided
            if not session_id:
                session_id = f"session-{str(uuid.uuid4())[:12]}"
            
            # Create session via API
            response = self.client.post(
                f"{self.manager_url}/sessions",
                json={"session_id": session_id},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Create local temp directory for session
            local_session_temp = os.path.join(tempfile.gettempdir(), f"session_{session_id}")
            os.makedirs(local_session_temp, exist_ok=True)
            
            logger.info(f"Created session {session_id} via session manager (PID: {data['pid']})")
            
            return BashSession(
                session_id=session_id,
                backend_id=data['session_id'],
                workspace_path=data['workspace_path'],
                session_temp_dir=local_session_temp,
                backend_type="session_manager"
            )
        
        except Exception as e:
            logger.error(f"Error creating session via session manager: {e}")
            raise
    
    def execute_command(
        self,
        session: BashSession,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute command in session via session manager service."""
        try:
            response = self.client.post(
                f"{self.manager_url}/sessions/{session.session_id}/exec",
                json={
                    "command": command,
                    "timeout": timeout
                },
                timeout=timeout + 5  # Add buffer for service processing
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.debug(f"Command executed in session {session.session_id}: return_code={result.get('return_code')}")
            
            return {
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "return_code": result.get("return_code", 0)
            }
        
        except httpx.TimeoutException:
            logger.error(f"Command timeout in session {session.session_id}")
            return {
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": 124
            }
        except Exception as e:
            logger.error(f"Command execution failed in session {session.session_id}: {e}")
            return {
                "output": "",
                "error": str(e),
                "return_code": 1
            }
    
    def check_session_health(self, session: BashSession) -> bool:
        """Check if session is healthy via session manager service."""
        try:
            response = self.client.get(
                f"{self.manager_url}/sessions/{session.session_id}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            is_healthy = data.get("healthy", False)
            
            if not is_healthy:
                logger.warning(f"Session {session.session_id} is not healthy")
            
            return is_healthy
        
        except Exception as e:
            logger.warning(f"Health check failed for session {session.session_id}: {e}")
            return False
    
    def cleanup_session(self, session: BashSession) -> None:
        """Delete session via session manager service."""
        try:
            response = self.client.delete(
                f"{self.manager_url}/sessions/{session.session_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Cleanup initiated for session {session.session_id}")
        
        except Exception as e:
            logger.warning(f"Cleanup failed for session {session.session_id}: {e}")
    
    def close(self) -> None:
        """Close HTTP client connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("SessionManagerBackend HTTP client closed")
        except Exception as e:
            logger.warning(f"Failed to close HTTP client: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
