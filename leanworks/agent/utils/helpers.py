"""
Helper utilities for the agent module.
"""
import json
import sys
import time
import logging
from typing import Optional
from leanworks.utils.env import resolve_credential_path, get_project_id

logger = logging.getLogger(__name__)


class AgentHelpers:
    """Helper class containing utility functions for agent operations."""
    
    @staticmethod
    def get_project_id_from_credentials(credential_path: Optional[str] = None) -> str:
        """Read project_id from GCP credential file.
        
        Args:
            credential_path: Path to GCP credential JSON file
            
        Returns:
            str: The project_id from the credential file
            
        Raises:
            FileNotFoundError: If credential file doesn't exist
            KeyError: If project_id is not found in credential file
        """
        resolved_path = credential_path or resolve_credential_path()
        project_id = get_project_id(resolved_path)
        if project_id:
            return project_id
        try:
            with open(resolved_path, "r") as f:
                credential_data = json.load(f)
            project_id = credential_data.get("project_id")
            if not project_id:
                raise KeyError(f"project_id not found in {resolved_path}")
            return project_id
        except FileNotFoundError:
            logger.error(f"Credential file not found: {resolved_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {resolved_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read project_id from {resolved_path}: {e}")
            raise
    
    @staticmethod
    def stream_text(text: str, delay: float = 0.02) -> None:
        """Stream text output with a typewriter effect.
        
        Args:
            text: Text to stream
            delay: Delay between characters in seconds
        """
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()  # Add newline at the end
