from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
from google.cloud import bigquery
import logging
import traceback
from leanworks.setting import get_client_info
from gitlab import Gitlab
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    user_id = "bharathkumar.l@sbnasoftware.com"
    try:
        bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
        client_name, _ = get_client_info(bq_client, user_id)
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
            storage_client=storage_client,
            secret_client=secret_client,
            model_client=model_client,
            bq_client_wrapper=bq_client_wrapper,
            user_id=user_id,
            session_id="dheo3gft",
            clear_conversation=True,
            tools=["gitlab"]
        )
        
        # Process a user message
        user_message = '''
         How many tickets are there in CCX active milestone

'''
        # cited_context = "task_id: 0722343a-464f-4a60-9ebf-ac6774755ff7"
        response = agent.process_message(user_message, deep_research=False)
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


