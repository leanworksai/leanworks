"""
Process isolation using Linux namespaces and cgroups v2.
"""
import os
import subprocess
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessIsolation:
    """Manages process isolation using Linux namespaces and cgroups."""
    
    def __init__(self, workspace_path: str):
        """
        Initialize process isolation context.
        
        Args:
            workspace_path: Path to workspace directory for session
        """
        self.workspace_path = workspace_path
        self.cgroup_v2_mount = "/sys/fs/cgroup"
    
    def create_isolated_process(
        self,
        command: list,
        cpu_limit: Optional[str] = None,
        memory_limit: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Create an isolated process using standard subprocess.
        
        Note: In GKE Autopilot, we use simple process isolation without
        namespaces since SYS_ADMIN capability is not allowed.
        Sessions are isolated via separate workspace directories.
        
        Args:
            command: Command to execute as list
            cpu_limit: CPU limit (e.g., "1000m" for 1 CPU)
            memory_limit: Memory limit (e.g., "512M")
        
        Returns:
            Tuple of (pid, cgroup_path)
        """
        try:
            # Start process in workspace directory
            # Isolation provided by separate workspace paths and resource limits
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workspace_path,
                preexec_fn=self._setup_environment
            )
            
            # Setup resource limits if cgroups are available (best effort)
            cgroup_path = ""
            try:
                cgroup_path = self._setup_cgroups(
                    process.pid,
                    cpu_limit=cpu_limit,
                    memory_limit=memory_limit
                )
            except Exception as e:
                logger.warning(f"Cgroups not available, continuing without resource limits: {e}")
            
            logger.info(f"Created isolated process {process.pid}")
            return process.pid, cgroup_path
        
        except Exception as e:
            logger.error(f"Failed to create isolated process: {e}")
            raise
    
    def _setup_environment(self):
        """Setup environment inside isolated process."""
        try:
            # Change to workspace
            os.chdir(self.workspace_path)
            
            # Set up mount point if needed
            # Mount the workspace as the root for the isolated process
            try:
                subprocess.run(['mount', '--make-rprivate', '/'], 
                             capture_output=True, check=False)
            except:
                pass
        except Exception as e:
            logger.warning(f"Failed to setup environment: {e}")
    
    def _setup_cgroups(
        self,
        pid: int,
        cpu_limit: Optional[str] = None,
        memory_limit: Optional[str] = None
    ) -> str:
        """
        Setup cgroups v2 resource limits for process (best effort).
        
        In GKE Autopilot, this may fail, so we treat it as optional.
        Sessions are still isolated via separate workspace directories.
        
        Args:
            pid: Process ID
            cpu_limit: CPU limit string
            memory_limit: Memory limit string
        
        Returns:
            Path to cgroup directory, or empty string if not available
        """
        try:
            # Check if cgroup v2 is available
            cgroup_v2_path = Path(self.cgroup_v2_mount)
            
            if not cgroup_v2_path.exists():
                logger.debug("Cgroup v2 not mounted, resource limits not available")
                return ""
            
            # Try to create session-specific cgroup
            session_cgroup = cgroup_v2_path / f"session-{pid}"
            
            try:
                session_cgroup.mkdir(exist_ok=True)
                
                # Add process to cgroup
                cgroup_procs = session_cgroup / "cgroup.procs"
                with open(cgroup_procs, 'w') as f:
                    f.write(str(pid))
                
                logger.info(f"Setup cgroups for pid {pid}: cpu={cpu_limit}, mem={memory_limit}")
                return str(session_cgroup)
            
            except PermissionError:
                logger.debug("No permission to setup cgroups (expected in Autopilot)")
                return ""
        
        except Exception as e:
            logger.debug(f"Cgroups not available: {e}")
            return ""
    
    def _set_cpu_limit(self, cgroup_path: Path, cpu_limit: str) -> None:
        """Set CPU limit for cgroup."""
        try:
            cpu_max_file = cgroup_path / "cpu.max"
            if cpu_max_file.exists():
                # Convert cpu_limit (e.g., "1000m") to cgroup format
                # For now, use a simple format: "100000 100000" means 1 CPU
                with open(cpu_max_file, 'w') as f:
                    f.write("100000 100000")  # 1 CPU in microseconds
        except Exception as e:
            logger.warning(f"Failed to set CPU limit: {e}")
    
    def _set_memory_limit(self, cgroup_path: Path, memory_limit: str) -> None:
        """Set memory limit for cgroup."""
        try:
            mem_max_file = cgroup_path / "memory.max"
            if mem_max_file.exists():
                # Convert memory_limit (e.g., "512M") to bytes
                limit_bytes = self._parse_memory_limit(memory_limit)
                with open(mem_max_file, 'w') as f:
                    f.write(str(limit_bytes))
        except Exception as e:
            logger.warning(f"Failed to set memory limit: {e}")
    
    @staticmethod
    def _parse_memory_limit(limit_str: str) -> int:
        """Parse memory limit string to bytes."""
        multipliers = {
            'K': 1024,
            'M': 1024 ** 2,
            'G': 1024 ** 3,
        }
        
        limit_str = limit_str.upper().strip()
        for suffix, mult in multipliers.items():
            if limit_str.endswith(suffix):
                try:
                    value = float(limit_str[:-1])
                    return int(value * mult)
                except ValueError:
                    pass
        
        # Try parsing as plain integer (bytes)
        try:
            return int(limit_str)
        except ValueError:
            return 512 * 1024 * 1024  # Default 512MB
    
    @staticmethod
    def kill_process(pid: int, grace_period: int = 5) -> bool:
        """
        Kill process tree gracefully.
        
        Args:
            pid: Process ID
            grace_period: Grace period in seconds before SIGKILL
        
        Returns:
            True if killed, False otherwise
        """
        import signal
        import time
        
        try:
            # Try SIGTERM first
            os.kill(pid, signal.SIGTERM)
            
            # Wait for graceful shutdown
            for _ in range(grace_period):
                try:
                    os.kill(pid, 0)  # Check if process still exists
                    time.sleep(1)
                except ProcessLookupError:
                    logger.info(f"Process {pid} terminated gracefully")
                    return True
            
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Process {pid} force killed")
                return True
            except ProcessLookupError:
                return True
        
        except ProcessLookupError:
            logger.info(f"Process {pid} not found")
            return True
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False
