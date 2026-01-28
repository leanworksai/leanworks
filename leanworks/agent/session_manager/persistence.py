"""
Session state persistence to survive pod restarts.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionPersistence:
    """Manages session state persistence using JSON files."""
    
    def __init__(self, storage_path: str = "/var/sessions"):
        """
        Initialize persistence layer.
        
        Args:
            storage_path: Base path for session storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.storage_path / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def save_session(
        self,
        session_id: str,
        pid: int,
        workspace_path: str,
        cgroup_path: str = ""
    ) -> None:
        """
        Save session metadata to persist state.
        
        Args:
            session_id: Unique session identifier
            pid: Process ID
            workspace_path: Path to session workspace
            cgroup_path: Path to cgroup (optional)
        """
        try:
            metadata = {
                "session_id": session_id,
                "pid": pid,
                "workspace_path": workspace_path,
                "cgroup_path": cgroup_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            metadata_file = self.metadata_dir / f"{session_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved session {session_id} metadata to {metadata_file}")
        
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load session metadata from persistent storage.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session metadata dict or None if not found
        """
        try:
            metadata_file = self.metadata_dir / f"{session_id}.json"
            
            if not metadata_file.exists():
                logger.info(f"Session {session_id} not found")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded session {session_id} metadata")
            return metadata
        
        except Exception as e:
            logger.error(f"Failed to load session metadata: {e}")
            return None
    
    def update_session_access(self, session_id: str) -> None:
        """
        Update last_used timestamp for session.
        
        Args:
            session_id: Session identifier
        """
        try:
            metadata = self.load_session(session_id)
            if metadata:
                metadata["last_used"] = datetime.now(timezone.utc).isoformat()
                metadata_file = self.metadata_dir / f"{session_id}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Failed to update session access time: {e}")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session metadata and workspace.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if successful
        """
        try:
            # Delete metadata file
            metadata_file = self.metadata_dir / f"{session_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Deleted session {session_id} metadata")
            
            # Delete workspace directory
            workspace_dir = self.storage_path / session_id
            if workspace_dir.exists():
                import shutil
                shutil.rmtree(workspace_dir)
                logger.info(f"Deleted session {session_id} workspace")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[Dict]:
        """
        List all active sessions.
        
        Returns:
            List of session metadata dicts
        """
        try:
            sessions = []
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        session = json.load(f)
                        sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to load session from {metadata_file}: {e}")
            
            logger.info(f"Listed {len(sessions)} sessions")
            return sessions
        
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def get_session_workspace(self, session_id: str) -> Path:
        """
        Get workspace path for session, creating it if needed.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Path to session workspace
        """
        workspace_dir = self.storage_path / session_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir
    
    def recover_sessions_on_startup(self) -> List[Dict]:
        """
        On startup, check which sessions are still valid.
        
        Returns:
            List of active sessions (PIDs still running)
        """
        import subprocess
        
        active_sessions = []
        
        try:
            all_sessions = self.list_sessions()
            
            for session in all_sessions:
                pid = session.get("pid")
                session_id = session.get("session_id")
                
                if pid:
                    # Check if process still exists
                    try:
                        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if exists
                        active_sessions.append(session)
                        logger.info(f"Session {session_id} (PID {pid}) still active")
                    except (ProcessLookupError, PermissionError):
                        # Process no longer exists
                        logger.info(f"Session {session_id} (PID {pid}) no longer running, cleaning up")
                        self.delete_session(session_id)
            
            logger.info(f"Recovered {len(active_sessions)} active sessions on startup")
            return active_sessions
        
        except Exception as e:
            logger.error(f"Failed to recover sessions on startup: {e}")
            return []
    
    def get_stale_sessions(
        self,
        max_age_seconds: int = 7200,
        max_idle_seconds: int = 3600
    ) -> List[Dict]:
        """
        Get sessions that match cleanup criteria.
        
        Args:
            max_age_seconds: Sessions older than this are stale
            max_idle_seconds: Sessions idle longer than this are stale
        
        Returns:
            List of stale session metadata dicts
        """
        try:
            stale_sessions = []
            now = datetime.now(timezone.utc)
            
            all_sessions = self.list_sessions()
            
            for session in all_sessions:
                created_at = datetime.fromisoformat(session["created_at"])
                last_used = datetime.fromisoformat(session["last_used"])
                
                age_seconds = (now - created_at).total_seconds()
                idle_seconds = (now - last_used).total_seconds()
                
                if age_seconds > max_age_seconds or idle_seconds > max_idle_seconds:
                    stale_sessions.append(session)
            
            logger.info(f"Found {len(stale_sessions)} stale sessions")
            return stale_sessions
        
        except Exception as e:
            logger.error(f"Failed to get stale sessions: {e}")
            return []
    
    def get_session_stats(self) -> Dict:
        """
        Get statistics about sessions.
        
        Returns:
            Stats dictionary
        """
        try:
            all_sessions = self.list_sessions()
            active_count = 0
            idle_count = 0
            ages = []
            
            now = datetime.now(timezone.utc)
            
            for session in all_sessions:
                try:
                    pid = session.get("pid")
                    if pid:
                        try:
                            os.kill(pid, 0)
                            active_count += 1
                        except (ProcessLookupError, PermissionError):
                            idle_count += 1
                    
                    created_at = datetime.fromisoformat(session["created_at"])
                    age = (now - created_at).total_seconds()
                    ages.append(age)
                
                except Exception:
                    pass
            
            return {
                "total_sessions": len(all_sessions),
                "active_sessions": active_count,
                "idle_sessions": idle_count,
                "oldest_session_age_seconds": max(ages) if ages else 0,
                "average_session_age_seconds": sum(ages) / len(ages) if ages else 0
            }
        
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
