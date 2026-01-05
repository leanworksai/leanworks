"""
Main entry point for the Leanworks API application
"""
import logging
from app import app
from app.api import routes  # Import routes to register them

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting async API server on port 8081")
    app.run(host="0.0.0.0", debug=True, port=8081)

