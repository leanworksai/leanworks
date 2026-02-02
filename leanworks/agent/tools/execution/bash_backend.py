"""
Abstract interface for bash session backends (Docker, Kubernetes).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BashSessionBackend(ABC):
    """Abstract base class for bash session backends."""
    
    @abstractmethod
    def create_session(self, session_id: str) -> 'BashSession':
        """Create a new bash session."""
        pass
    
    @abstractmethod
    def execute_command(self, session: 'BashSession', command: str, timeout: int) -> Dict[str, Any]:
        """Execute command in session. Returns dict with output, error, return_code."""
        pass
    
    @abstractmethod
    def check_session_health(self, session: 'BashSession') -> bool:
        """Check if session is still healthy."""
        pass
    
    @abstractmethod
    def cleanup_session(self, session: 'BashSession') -> None:
        """Clean up session resources."""
        pass


class BashSession:
    """Represents a bash session (Docker container or Kubernetes pod)."""
    
    def __init__(self, session_id: str, backend_id: str, workspace_path: str, 
                 session_temp_dir: str, backend_type: str):
        """
        Initialize BashSession.
        
        Args:
            session_id: Unique session identifier
            backend_id: Container name (Docker) or pod name (Kubernetes)
            workspace_path: Path to workspace in container/pod
            session_temp_dir: Local temp directory for session files
            backend_type: 'docker' or 'kubernetes'
        """
        self.session_id = session_id
        self.backend_id = backend_id
        self.workspace_path = workspace_path
        self.session_temp_dir = session_temp_dir
        self.backend_type = backend_type
