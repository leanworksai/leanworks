"""
FastAPI application for session manager service.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from session import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session manager instance
session_manager: Optional[SessionManager] = None


# Pydantic models for API
class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    session_id: Optional[str] = None


class ExecuteCommandRequest(BaseModel):
    """Request to execute a command in a session."""
    command: str
    timeout: Optional[int] = 30


class CleanupRequest(BaseModel):
    """Request to cleanup sessions."""
    max_age_seconds: Optional[int] = 7200
    max_idle_seconds: Optional[int] = 3600
    force: Optional[bool] = False
    session_ids: Optional[List[str]] = None


class CreateSessionResponse(BaseModel):
    """Response from session creation."""
    session_id: str
    pid: int
    workspace_path: str


class ExecuteCommandResponse(BaseModel):
    """Response from command execution."""
    output: str
    error: str
    return_code: int


class HealthCheckResponse(BaseModel):
    """Response from health check."""
    healthy: bool


class SessionListItem(BaseModel):
    """Session list item."""
    session_id: str
    pid: int
    status: str
    created_at: str
    last_used: str


class CleanupResponse(BaseModel):
    """Response from cleanup."""
    cleaned: int
    failed: int
    sessions_cleaned: List[str]
    sessions_failed: List[str]
    total_sessions_before: int
    total_sessions_after: int


class StatsResponse(BaseModel):
    """Response from stats endpoint."""
    total_sessions: int
    active_sessions: int
    idle_sessions: int
    total_created: int
    total_cleaned: int
    oldest_session_age_seconds: float
    average_session_age_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    global session_manager
    
    # Startup
    storage_path = os.environ.get("SESSION_STORAGE_PATH", "/var/sessions")
    session_manager = SessionManager(storage_path=storage_path)
    logger.info(f"Session manager started with storage at {storage_path}")
    
    yield
    
    # Shutdown - cleanup all sessions
    logger.info("Session manager shutting down, cleaning up sessions...")
    if session_manager:
        for session in session_manager.list_sessions():
            session_manager.cleanup_session(session["session_id"])


# Create FastAPI app
app = FastAPI(
    title="Bash Session Manager",
    description="Manages multiple bash sessions with process isolation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> Dict:
    """
    Create a new bash session.
    
    Returns:
        Session information including session_id, pid, workspace_path
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())[:12]
        
        # Create session
        session = session_manager.create_session(session_id)
        
        return CreateSessionResponse(
            session_id=session.session_id,
            pid=session.pid,
            workspace_path=session.workspace_path
        )
    
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/exec", response_model=ExecuteCommandResponse)
async def execute_command(
    session_id: str,
    request: ExecuteCommandRequest
) -> Dict:
    """
    Execute a command in a session.
    
    Args:
        session_id: Session identifier
        request: Command and timeout
    
    Returns:
        Command output, error, and return code
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if not session.is_healthy():
            raise HTTPException(status_code=410, detail=f"Session {session_id} is not running")
        
        result = session.execute_command(request.command, timeout=request.timeout)
        
        return ExecuteCommandResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute command in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/health", response_model=HealthCheckResponse)
async def check_session_health(session_id: str) -> Dict:
    """
    Check health of a session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Health status
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return HealthCheckResponse(healthy=session.is_healthy())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check session health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def cleanup_session(session_id: str) -> Dict:
    """
    Cleanup a specific session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Cleanup result
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        success = session_manager.cleanup_session(session_id)
        
        if success:
            return {"status": "cleaned", "session_id": session_id}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to cleanup session {session_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionListItem])
async def list_sessions() -> List[Dict]:
    """
    List all active sessions.
    
    Returns:
        List of session information
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        sessions = session_manager.list_sessions()
        return [
            SessionListItem(
                session_id=s["session_id"],
                pid=s["pid"],
                status=s.get("status", "active"),
                created_at=s["created_at"],
                last_used=s["last_used"]
            )
            for s in sessions
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup", response_model=CleanupResponse)
async def cleanup_sessions(request: CleanupRequest) -> Dict:
    """
    Cleanup stale or specific sessions.
    
    Args:
        request: Cleanup parameters
    
    Returns:
        Cleanup statistics
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        # Cleanup specific sessions if IDs provided
        if request.session_ids:
            result = session_manager.cleanup_specific_sessions(request.session_ids)
        else:
            # Cleanup stale sessions
            result = session_manager.cleanup_stale_sessions(
                max_age_seconds=request.max_age_seconds or 7200,
                max_idle_seconds=request.max_idle_seconds or 3600,
                force=request.force or False
            )
        
        return CleanupResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> Dict:
    """
    Get session manager statistics.
    
    Returns:
        Statistics including session counts and performance metrics
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        stats = session_manager.get_stats()
        return StatsResponse(**stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
