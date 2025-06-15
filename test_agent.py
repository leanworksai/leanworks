from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
from google.cloud import bigquery
import logging
import traceback
from leanworks.setting import get_client_name
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    try:
        bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
        client_name = get_client_name(bq_client, "yanfu@leanworks.ai")
        storage_client = CloudStorage("gcp_credential.json", bucket=client_name)
        secret_client = GCPSecretLoader("gcp_credential.json", client_name=client_name)
        model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))

        class BigQueryClient:
            def __init__(self, bq_client, client_name):
                self.bq_client = bq_client
                self.client_name = client_name
                
        bq_client_wrapper = BigQueryClient(bq_client, client_name)
        # Initialize the chat agent with BigQuery client
        agent = ChatAgent(
            storage_client,
            secret_client,
            model_client,
            bq_client_wrapper,
            user_id="yanfu@leanworks.ai",
            session_id="cdj976r6f",
            clear_conversation=False  # Change to True to reset conversation each time
        )
        
        # Process a user message
        user_message = "what is vijay's speed in completing tasks?"
        # cited_context = "task_id: 0722343a-464f-4a60-9ebf-ac6774755ff7"
        response = agent.process_message(user_message)
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


