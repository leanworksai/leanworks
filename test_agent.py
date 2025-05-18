from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    # Initialize the chat agent
    storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
    secret_client = GCPSecretLoader("gcp_credential.json", client_name="leanworks")
    model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
    agent = ChatAgent(
        storage_client,
        secret_client,
        model_client,
        user_id="zhuyanfu0712@gmail.com",
        session_id="frfe8384",
        clear_conversation=True
    )
    
    # Process a user message
    user_message = "show me all projects"
    response = agent.process_message(user_message)
    
    return response

if __name__ == "__main__":
    main()


