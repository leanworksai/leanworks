from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
import logging
import traceback

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    try:
        # Initialize the chat agent
        storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
        secret_client = GCPSecretLoader("gcp_credential.json", client_name="leanworks")
        model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
        agent = ChatAgent(
            storage_client,
            secret_client,
            model_client,
            user_id="zhuyanfu0712@gmail.com",
            session_id="hde29897",
            clear_conversation=False  # Change to True to reset conversation each time
        )
        
        # Process a user message
        user_message = "give me a summary of product development project’s progress since May 1st, including others’ updates"
        response = agent.process_message(user_message)
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


