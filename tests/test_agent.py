from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
from google.cloud import bigquery
import logging
import traceback
import time
from leanworks.setting import get_client_info
from gitlab import Gitlab
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    user_id = "bharathkumar.l@sbnasoftware.com"
    # user_id = "yanfu@leanworks.ai"
    
    print("=" * 80)
    print("🚀 AGENT PERFORMANCE TESTING")
    print("=" * 80)
    
    try:
        # Time the overall setup
        setup_start = time.time()
        
        print("📋 Setting up clients...")
        bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
        client_name, _ = get_client_info(bq_client, user_id)
        storage_client = CloudStorage("gcp_credential.json", bucket=client_name)
        secret_client = GCPSecretLoader("gcp_credential.json", client_name=client_name)
        model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
        
        class BigQueryClient:
            def __init__(self, bq_client, client_name):
                self.bq_client = bq_client
                self.client_name = client_name
                self.table_schemas = {}
                
        bq_client_wrapper = BigQueryClient(bq_client, client_name)
        
        # Time the agent initialization
        print("🤖 Initializing ChatAgent with lazy loading...")
        agent_init_start = time.time()
        
        agent = ChatAgent(
            storage_client=storage_client,
            secret_client=secret_client,
            model_client=model_client,
            bq_client_wrapper=bq_client_wrapper,
            user_id=user_id,
            session_id="jfwp3gy9",
            clear_conversation=True,
            tools=["gitlab"]  # Only load what we need
        )
        
        agent_init_time = time.time() - agent_init_start
        setup_time = time.time() - setup_start
        
        print(f"✅ Agent initialized in {agent_init_time:.2f}s (total setup: {setup_time:.2f}s)")
        print()
        
        # Process a user message with timing
        user_message = '''
         How many tickets are there in the the CCXAI group?
'''
        
        print("💬 Processing user message:")
        print(f"   Query: {user_message.strip()}")
        print(f"   Thinking mode: True (evaluation enabled)")
        print(f"   Streaming mode: True (shows tools and streams response)")
        print()
        
        # Time the response processing
        response_start = time.time()
        
        response = agent.process_message(user_message, thinking=True, streaming=True)
        
        response_time = time.time() - response_start
        total_time = time.time() - setup_start
        
        print("=" * 80)
        print("⏱️  PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"🔧 Setup Time:           {setup_time:.2f}s")
        print(f"🤖 Agent Init Time:      {agent_init_time:.2f}s") 
        print(f"💭 Response Time:        {response_time:.2f}s")
        print(f"🎯 Total Time:           {total_time:.2f}s")
        print("=" * 80)
        
        # Print response details
        if response:
            print(f"📝 Response Length:      {len(response.get('content', ''))} characters")
            print(f"📊 Data Sources Used:    {len(response.get('data_sources', []))}")
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


