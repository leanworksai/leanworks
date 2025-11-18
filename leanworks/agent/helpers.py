"""
Helper utilities for the agent module.
"""
import json
import sys
import time
import logging

logger = logging.getLogger(__name__)


class AgentHelpers:
    """Helper class containing utility functions for agent operations."""
    
    @staticmethod
    def get_project_id_from_credentials(credential_path: str = "gcp_credential.json") -> str:
        """Read project_id from GCP credential file.
        
        Args:
            credential_path: Path to GCP credential JSON file
            
        Returns:
            str: The project_id from the credential file
            
        Raises:
            FileNotFoundError: If credential file doesn't exist
            KeyError: If project_id is not found in credential file
        """
        try:
            with open(credential_path, "r") as f:
                credential_data = json.load(f)
            project_id = credential_data.get("project_id")
            if not project_id:
                raise KeyError(f"project_id not found in {credential_path}")
            return project_id
        except FileNotFoundError:
            logger.error(f"Credential file not found: {credential_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {credential_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read project_id from {credential_path}: {e}")
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

