"""
Session lifecycle management.
"""
import logging
import subprocess
from typing import Dict, Any, Optional
from .isolation import ProcessIsolation
from .persistence import SessionPersistence

logger = logging.getLogger(__name__)


class Session:
    """Represents and manages a single bash session."""
    
    def __init__(
        self,
        session_id: str,
        pid: int,
        workspace_path: str,
        cgroup_path: str = "",
        persistence: Optional[SessionPersistence] = None
    ):
        """
        Initialize session.
        
        Args:
            session_id: Unique session identifier
            pid: Process ID of bash process
            workspace_path: Path to workspace directory
            cgroup_path: Path to cgroup (optional)
            persistence: SessionPersistence instance
        """
        self.session_id = session_id
        self.pid = pid
        self.workspace_path = workspace_path
        self.cgroup_path = cgroup_path
        self.persistence = persistence
    
    def execute_command(
        self,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a command in the session.
        
        Args:
            command: Bash command to execute
            timeout: Command timeout in seconds
        
        Returns:
            Dict with output, error, return_code
        """
        try:
            # Update access time
            if self.persistence:
                self.persistence.update_session_access(self.session_id)
            
            # Execute command directly in workspace directory
            # No need for nsenter since the session process was created in this workspace
            exec_cmd = [
                'bash', '-c', f'cd {self.workspace_path} && {command}'
            ]
            
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timeout in session {self.session_id} after {timeout}s")
            return {
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": 124
            }
        except Exception as e:
            logger.error(f"Command execution failed in session {self.session_id}: {e}")
            return {
                "output": "",
                "error": str(e),
                "return_code": -1
            }
    
    def is_healthy(self) -> bool:
        """
        Check if session is still running.
        
        Returns:
            True if session process is alive
        """
        try:
            import os
            os.kill(self.pid, 0)  # Check without killing
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False
    
    def cleanup(self) -> bool:
        """
        Cleanup session resources.
        
        Returns:
            True if successful
        """
        try:
            # Kill process
            ProcessIsolation.kill_process(self.pid)
            
            # Delete persistent data
            if self.persistence:
                self.persistence.delete_session(self.session_id)
            
            logger.info(f"Cleaned up session {self.session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to cleanup session {self.session_id}: {e}")
            return False


class SessionManager:
    """Manages collection of sessions."""
    
    def __init__(self, storage_path: str = "/var/sessions"):
        """
        Initialize session manager.
        
        Args:
            storage_path: Base path for session storage
        """
        self.persistence = SessionPersistence(storage_path)
        self.sessions: Dict[str, Session] = {}
        self.total_created = 0
        self.total_cleaned = 0
        
        # Recover sessions on startup
        self._recover_sessions()
    
    def _recover_sessions(self) -> None:
        """Recover active sessions from persistent storage on startup."""
        try:
            active_sessions = self.persistence.recover_sessions_on_startup()
            for session_data in active_sessions:
                session_id = session_data["session_id"]
                self.sessions[session_id] = Session(
                    session_id=session_id,
                    pid=session_data["pid"],
                    workspace_path=session_data["workspace_path"],
                    cgroup_path=session_data.get("cgroup_path", ""),
                    persistence=self.persistence
                )
            logger.info(f"Recovered {len(self.sessions)} sessions")
        except Exception as e:
            logger.error(f"Failed to recover sessions: {e}")
    
    def create_session(self, session_id: str) -> Session:
        """
        Create a new bash session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Session object
        
        Raises:
            Exception if session creation fails
        """
        try:
            # Get workspace directory
            workspace_path = self.persistence.get_session_workspace(session_id)
            
            # Create isolated process
            isolation = ProcessIsolation(str(workspace_path))
            pid, cgroup_path = isolation.create_isolated_process(
                ['bash', '-i'],
                cpu_limit="1000m",
                memory_limit="512M"
            )
            
            # Save session metadata
            self.persistence.save_session(
                session_id,
                pid,
                str(workspace_path),
                cgroup_path
            )
            
            # Create session object
            session = Session(
                session_id=session_id,
                pid=pid,
                workspace_path=str(workspace_path),
                cgroup_path=cgroup_path,
                persistence=self.persistence
            )
            
            self.sessions[session_id] = session
            self.total_created += 1
            
            logger.info(f"Created session {session_id} with PID {pid}")
            return session
        
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session object or None if not found
        """
        return self.sessions.get(session_id)
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Cleanup specific session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if successful
        """
        session = self.sessions.get(session_id)
        if session:
            success = session.cleanup()
            if success:
                del self.sessions[session_id]
                self.total_cleaned += 1
            return success
        return False
    
    def cleanup_stale_sessions(
        self,
        max_age_seconds: int = 7200,
        max_idle_seconds: int = 3600,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Cleanup stale sessions.
        
        Args:
            max_age_seconds: Sessions older than this are stale
            max_idle_seconds: Sessions idle longer than this are stale
            force: Force cleanup regardless of age/idle time
        
        Returns:
            Cleanup statistics
        """
        try:
            stale_sessions = self.persistence.get_stale_sessions(
                max_age_seconds=max_age_seconds,
                max_idle_seconds=max_idle_seconds
            )
            
            cleaned = []
            failed = []
            
            for session_data in stale_sessions:
                session_id = session_data["session_id"]
                if self.cleanup_session(session_id):
                    cleaned.append(session_id)
                else:
                    failed.append(session_id)
            
            return {
                "cleaned": len(cleaned),
                "failed": len(failed),
                "sessions_cleaned": cleaned,
                "sessions_failed": failed,
                "total_sessions_before": len(stale_sessions),
                "total_sessions_after": len(self.sessions)
            }
        
        except Exception as e:
            logger.error(f"Failed to cleanup stale sessions: {e}")
            return {
                "cleaned": 0,
                "failed": 0,
                "sessions_cleaned": [],
                "sessions_failed": [],
                "error": str(e)
            }
    
    def cleanup_specific_sessions(self, session_ids: list) -> Dict[str, Any]:
        """
        Cleanup specific sessions by ID.
        
        Args:
            session_ids: List of session IDs to clean
        
        Returns:
            Cleanup statistics
        """
        cleaned = []
        failed = []
        
        for session_id in session_ids:
            if self.cleanup_session(session_id):
                cleaned.append(session_id)
            else:
                failed.append(session_id)
        
        return {
            "cleaned": len(cleaned),
            "failed": len(failed),
            "sessions_cleaned": cleaned,
            "sessions_failed": failed,
            "total_sessions_before": len(cleaned) + len(failed),
            "total_sessions_after": len(self.sessions)
        }
    
    def list_sessions(self) -> list:
        """
        List all active sessions.
        
        Returns:
            List of session metadata dicts
        """
        return self.persistence.list_sessions()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.
        
        Returns:
            Statistics dict
        """
        stats = self.persistence.get_session_stats()
        stats["total_created"] = self.total_created
        stats["total_cleaned"] = self.total_cleaned
        return stats
