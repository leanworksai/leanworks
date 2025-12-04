from leanworks.agent.chat import ChatAgent
from google.cloud import firestore, secretmanager
from google.oauth2 import service_account
from anthropic import Anthropic
import logging
import traceback
import time
import asyncio
import json
from typing import Dict, Tuple, Optional
import re
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def initialize_clients_async(user_id: str, org_name: str, tools: Optional[list] = None) -> Tuple[firestore.Client, secretmanager.SecretManagerServiceClient, Anthropic, list]:
    """Initialize all required clients asynchronously"""
    start_time = time.time()
    
    try:
        logger.info(f"Initializing clients for user {user_id} in org {org_name}")
        
        # Initialize clients in parallel using asyncio.gather
        loop = asyncio.get_event_loop()
        
        # Load credentials and get project_id
        def init_clients():
            credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
            with open("gcp_credential.json", "r") as f:
                credential_data = json.load(f)
            project_id = credential_data["project_id"]
            
            # Initialize Firestore client
            firestore_client = firestore.Client(credentials=credentials, project=project_id, database="leanworks-prod")
            
            # Initialize Secret Manager client
            secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=credentials)
            
            return firestore_client, secret_manager_client
        
        # Run client initializations in executor to avoid blocking
        logger.info(f"Initializing Firestore and Secret Manager clients for org: {org_name}")
        firestore_client, secret_manager_client = await loop.run_in_executor(None, init_clients)
        
        # Get Claude API key (project_id will be read from credential file in ChatAgent)
        def get_secret(name):
            with open("gcp_credential.json", "r") as f:
                credential_data = json.load(f)
            project_id = credential_data["project_id"]
            full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
            response = secret_manager_client.access_secret_version(name=full_name)
            return response.payload.data.decode("UTF-8")
        
        claude_api_key = await loop.run_in_executor(None, lambda: get_secret("claude-api-key"))
        if not claude_api_key:
            raise ValueError(f"claude-api-key not found for org: {org_name}")
        
        # Initialize Anthropic client
        model_client = Anthropic(api_key=claude_api_key)
        
        # Tools are now passed explicitly - no database fetching
        if tools is None:
            # No tools provided - use only default/internal tools (search, postgres, duckdb)
            tools = []
            logger.info("No tools provided - using only default internal tools (search, postgres, duckdb)")
        else:
            logger.info(f"Using {len(tools)} tools from provided list: {tools}")
        
        init_time = time.time() - start_time
        logger.info(f"Client initialization completed in {init_time:.3f}s for user: {user_id}")
        
        return firestore_client, secret_manager_client, model_client, tools
    except Exception as e:
        logger.error(f"Error in initialize_clients_async for user {user_id}: {str(e)}")
        traceback.print_exc()
        raise

async def main_async():
    """Async version of main function with async client initialization"""
    # user_id = "bharathkumar.l@sbnasoftware.com"
    user_id = "yanfu@leanworks.ai"
    org_name = "leanworks.ai"
    
    print("=" * 80)
    print("🚀 AGENT PERFORMANCE TESTING")
    print("=" * 80)
    
    try:
        # Time the overall setup
        setup_start = time.time()
        
        print("⚡ Initializing clients with parallel execution...")
        client_init_start = time.time()
        
        # Use async client initialization (no BigQuery setup needed)
        firestore_client, secret_manager_client, model_client, tools = await initialize_clients_async(user_id, org_name)
        
        client_init_time = time.time() - client_init_start
        
        # Time the agent initialization
        print("🤖 Initializing ChatAgent...")
        agent_init_start = time.time()

        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_name=org_name,
            session_id="hf38r89r",
            clear_conversation=True,
            tools=tools
        )
        
        agent_init_time = time.time() - agent_init_start
        setup_time = time.time() - setup_start
        
        print(f"✅ Agent initialized in {agent_init_time:.2f}s (total setup: {setup_time:.2f}s)")
        print(f"📊 Detailed timing breakdown:")
        print(f"   - Client initialization (parallel): {client_init_time:.3f}s")
        print(f"   - Agent initialization: {agent_init_time:.3f}s")
        print()
        
        # Process a user message with timing
        user_message = '''
        what's the latest progress of web development?
'''
        
        print("💬 Processing user message:")
        print(f"   Query: {user_message.strip()}")
        print(f"   Thinking mode: False (evaluation disabled for speed)")
        print(f"   Streaming mode: True (shows tools and streams response)")
        print()
        
        # Time the response processing
        response_start = time.time()
        
        response = agent.process_message(user_message, thinking=False, streaming=True)
        
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

def print_full_tool_result(tool_name: str, result: any):
    """Print tool call results in a readable format without truncation"""
    print(f"\n{'='*80}")
    print(f"🔧 TOOL CALL RESULT: {tool_name}")
    print(f"{'='*80}")
    
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"\n📋 {key.upper()}:")
            if isinstance(value, (list, dict)):
                print(json.dumps(value, indent=2, ensure_ascii=False))
            else:
                print(str(value))
    elif isinstance(result, list):
        print(f"📋 RESULTS ({len(result)} items):")
        for i, item in enumerate(result):
            print(f"\n--- Item {i+1} ---")
            if isinstance(item, dict):
                print(json.dumps(item, indent=2, ensure_ascii=False))
            else:
                print(str(item))
    else:
        print(f"📋 RESULT:")
        print(str(result))
    
    print(f"\n{'='*80}\n")


def main():
    """Synchronous wrapper for the async main function"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    main()


