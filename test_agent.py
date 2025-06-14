from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
from google.cloud import bigquery
import logging
import traceback

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BigQueryClient:
    """Wrapper class for BigQuery client that includes dataset_id"""
    def __init__(self, credentials_path, dataset_id):
        self.client = bigquery.Client.from_service_account_json(credentials_path)
        self.dataset_id = dataset_id
    
    def query(self, query_string):
        """Execute a BigQuery query"""
        return self.client.query(query_string)


def main():
    try:
        # Initialize clients
        storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
        secret_client = GCPSecretLoader("gcp_credential.json", client_name="leanworks")
        model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
        
        # Initialize BigQuery client with service account
        bq_client = BigQueryClient("gcp_credential.json", dataset_id="leanworks")
        
        # Initialize the chat agent with BigQuery client
        agent = ChatAgent(
            storage_client,
            secret_client,
            model_client,
            bq_client,
            user_id="yanfu@leanworks.ai",
            session_id="dhe2o86",
            clear_conversation=False  # Change to True to reset conversation each time
        )
        
        # Process a user message
        user_message = "summarize email response from sara"
        # cited_context = "task_id: 0722343a-464f-4a60-9ebf-ac6774755ff7"
        response = agent.process_message(user_message)
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


