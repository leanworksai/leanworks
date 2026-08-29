"""Docker-based bash session backend for local development."""
import subprocess
import tempfile
import os
import uuid
import time
import re
import logging
from typing import Dict, Any
from .bash_backend import BashSessionBackend, BashSession

logger = logging.getLogger(__name__)


class DockerBashBackend(BashSessionBackend):
    """Docker backend for bash sessions in local development."""
    
    def __init__(self, image_name: str = "leanworks-bash-session:latest", 
                 ensure_image_fn=None):
        """
        Initialize Docker backend.
        
        Args:
            image_name: Docker image to use for bash sessions
            ensure_image_fn: Optional function to ensure image exists
        """
        self.image_name = image_name
        self.ensure_image_fn = ensure_image_fn
    
    def create_session(self, session_id: str) -> BashSession:
        """Create a Docker container bash session."""
        # Generate unique container name
        container_name = f"bash-session-{uuid.uuid4().hex[:12]}"
        
        # Create session-specific temp directory on host
        session_temp_dir = os.path.join(tempfile.gettempdir(), f"session_{session_id or 'default'}")
        os.makedirs(session_temp_dir, exist_ok=True)
        container_mount_path = '/workspace'
        
        try:
            # Ensure custom image exists if callback provided
            image_name = self.image_name
            if self.ensure_image_fn:
                image_name = self.ensure_image_fn()
            
            # Create and start Docker container
            create_cmd = [
                'docker', 'run', '-d',
                '--name', container_name,
                '--rm',  # Auto-remove when stopped
                '--network', 'none',  # No network access for security
                '--memory', '512m',  # Limit memory
                '--cpus', '1.0',  # Limit to 1 CPU
                '--pids-limit', '100',  # Limit number of processes
                '--read-only',  # Read-only root filesystem
                '--tmpfs', '/tmp:rw,noexec,nosuid,size=100m',  # Writable /tmp
                '--tmpfs', '/home:rw,noexec,nosuid,size=100m',  # Writable /home
                '-v', f'{session_temp_dir}:{container_mount_path}:rw',  # Mount session dir
                image_name,
                'sleep', 'infinity'
            ]
            
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(
                    "Failed to create Docker container (stderr_chars=%d)",
                    len(result.stderr),
                )
                raise Exception(f"Docker create failed: {result.stderr}")
            
            container_id = result.stdout.strip()
            
            # Wait for container to be fully running
            for attempt in range(5):
                time.sleep(0.2)
                check_cmd = ['docker', 'inspect', '--format', '{{.State.Running}}', container_name]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
                
                if check_result.returncode == 0 and check_result.stdout.strip() == 'true':
                    logger.info(f"Container {container_name} is running and ready")
                    break
                
                if attempt == 4:
                    raise Exception(f"Container started but not running after 1s")
            
            logger.info(f"Created Docker container {container_name} ({container_id[:12]})")
            
            # Create BashSession object with Docker-specific attributes
            session = BashSession(
                session_id=session_id or 'default',
                backend_id=container_name,
                workspace_path=container_mount_path,
                session_temp_dir=session_temp_dir,
                backend_type='docker'
            )
            # Store Docker-specific info
            session.container_id = container_id
            session.container_workspace_path = container_mount_path
            
            return session
        
        except FileNotFoundError:
            logger.error("Docker is not installed or not in PATH")
            raise Exception("Docker is not installed or not in PATH")
        except subprocess.TimeoutExpired:
            logger.error("Docker container creation timed out after 10 seconds")
            raise Exception("Docker container creation timed out")
        except Exception as e:
            logger.error(f"Error creating Docker container: {e}")
            raise
    
    def execute_command(self, session: BashSession, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command via docker exec."""
        try:
            # Translate file paths if needed
            translated_command = self._translate_path_for_container(command, session)
            
            # Pre-check: If command references /workspace paths, ensure they exist on host
            if "/workspace" in translated_command:
                try:
                    parts = translated_command.split()
                    missing = False
                    for part in parts:
                        if part.startswith("/workspace"):
                            host_path = os.path.join(session.session_temp_dir, os.path.relpath(part, "/workspace"))
                            if not os.path.exists(host_path):
                                missing = True
                                break
                    if missing:
                        return {"output": "", "error": "Referenced file does not exist on host", "return_code": 1}
                except Exception:
                    pass
            
            # Execute command in Docker container
            exec_cmd = [
                'docker', 'exec',
                session.backend_id,
                'sh', '-c', f'cd /workspace && {translated_command}'
            ]
            
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return_code = result.returncode
            if return_code != 0:
                logger.info("Bash command failed (return_code=%d)", return_code)
            
            return {
                "output": result.stdout,
                "error": result.stderr,
                "return_code": return_code
            }
        
        except subprocess.TimeoutExpired:
            logger.info(f"Bash command timed out after {timeout} seconds")
            # Try to kill the command
            try:
                subprocess.run(['docker', 'exec', session.backend_id, 'pkill', '-9', 'sh'],
                             capture_output=True, timeout=5)
            except:
                pass
            
            return {
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": 124
            }
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "output": "",
                "error": str(e),
                "return_code": -1
            }
    
    def check_session_health(self, session: BashSession) -> bool:
        """Check if container is running."""
        try:
            check_cmd = ['docker', 'inspect', '--format', '{{.State.Running}}', session.backend_id]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            
            if check_result.returncode == 0 and check_result.stdout.strip() == 'true':
                logger.debug(f"Container {session.backend_id} is healthy")
                return True
            else:
                logger.warning(f"Container {session.backend_id} is not running")
                return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def cleanup_session(self, session: BashSession) -> None:
        """Stop and remove Docker container."""
        try:
            # Stop container
            subprocess.run(['docker', 'stop', session.backend_id],
                         capture_output=True, timeout=10)
            
            # Remove container (--rm flag should handle this, but be explicit)
            subprocess.run(['docker', 'rm', '-f', session.backend_id],
                         capture_output=True, timeout=10)
            
            logger.info(f"Cleaned up Docker container {session.backend_id}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _translate_path_for_container(self, command: str, session: BashSession) -> str:
        """
        Translate file paths in command from host session directory to container workspace path.
        
        Args:
            command: The bash command with potential file paths
            session: BashSession instance with mount info
            
        Returns:
            Command with translated paths
        """
        if not hasattr(session, 'session_temp_dir') or not hasattr(session, 'container_workspace_path'):
            return command
        
        session_temp_dir = session.session_temp_dir
        container_workspace_path = session.container_workspace_path
        
        # Normalize paths for comparison
        session_temp_dir_norm = os.path.normpath(session_temp_dir)
        
        # Replace session directory paths with workspace paths
        if session_temp_dir_norm in command:
            # Escape special regex characters in the session directory path
            escaped_temp_dir = re.escape(session_temp_dir_norm)
            
            def replace_session_path(match):
                matched_path = match.group(0)
                # Get the relative path from session directory
                if matched_path.startswith(session_temp_dir_norm):
                    rel_path = matched_path[len(session_temp_dir_norm):].lstrip(os.sep)
                    if rel_path:
                        # Construct container workspace path
                        container_path = os.path.join(container_workspace_path, rel_path).replace('\\', '/')
                    else:
                        container_path = container_workspace_path
                    return container_path
                return matched_path
            
            # Replace session directory paths
            translated_command = re.sub(
                escaped_temp_dir + r'[^\s"\'<>|&;()]*',
                replace_session_path,
                command
            )
            return translated_command
        
        return command
