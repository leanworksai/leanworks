from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
import logging
import traceback
import time
import asyncio
import json
from typing import Dict, Tuple, Optional
from leanworks.setting import get_client_info
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

async def initialize_clients_async(user_id: str) -> Tuple[CloudStorage, GCPSecretLoader, Anthropic, any, list]:
    """Initialize all required clients asynchronously"""
    start_time = time.time()
    
    try:
        # Get client info
        logger.info(f"Fetching client info for user: {user_id}")
        result = get_client_info(user_id)
        if result is None:
            logger.error(f"get_client_info returned None for user_id: {user_id}")
            raise ValueError(f"Could not get client info for user_id: {user_id}")
        
        # Validate that result is a tuple with at least 2 elements
        if not isinstance(result, tuple) or len(result) < 2:
            logger.error(f"get_client_info returned invalid result type: {type(result)} for user_id: {user_id}")
            raise ValueError(f"Invalid client info format for user_id: {user_id}")
        
        domain, available_tools = result
        
        if not domain:
            raise ValueError(f"Could not determine domain for user_id: {user_id}")
        
        # Extract client_name from domain by removing all non-alphanumeric characters
        client_name = re.sub(r'[^a-zA-Z0-9]', '', domain)
        
        # Initialize clients in parallel using asyncio.gather
        loop = asyncio.get_event_loop()
        
        # Run client initializations in executor to avoid blocking
        logger.info(f"Initializing storage and secret clients for domain: {domain}")
        storage_client, secret_client = await asyncio.gather(
            loop.run_in_executor(None, lambda: CloudStorage("gcp_credential.json", bucket=client_name)),
            loop.run_in_executor(None, lambda: GCPSecretLoader("gcp_credential.json"))
        )
        
        # Get Claude API key
        claude_api_key = await loop.run_in_executor(None, lambda: secret_client.get("claude-api-key"))
        if not claude_api_key:
            raise ValueError(f"claude-api-key not found for domain: {domain}")
        
        # Initialize Anthropic client
        model_client = Anthropic(api_key=claude_api_key)
        
        # Create Firestore client wrapper
        class FirestoreClientWrapper:
            def __init__(self, domain, client_name):
                self.domain = domain
                self.client_name = client_name  # For backward compatibility
        
        firestore_client_wrapper = FirestoreClientWrapper(domain, client_name)
        
        init_time = time.time() - start_time
        logger.info(f"Client initialization completed in {init_time:.3f}s for user: {user_id}")
        
        return storage_client, secret_client, model_client, firestore_client_wrapper, available_tools
    except Exception as e:
        logger.error(f"Error in initialize_clients_async for user {user_id}: {str(e)}")
        traceback.print_exc()
        raise

async def main_async():
    """Async version of main function with async client initialization"""
    # user_id = "bharathkumar.l@sbnasoftware.com"
    user_id = "yanfu@leanworks.ai"
    
    print("=" * 80)
    print("🚀 AGENT PERFORMANCE TESTING")
    print("=" * 80)
    
    try:
        # Time the overall setup
        setup_start = time.time()
        
        print("⚡ Initializing clients with parallel execution...")
        client_init_start = time.time()
        
        # Use async client initialization (no BigQuery setup needed)
        storage_client, secret_client, model_client, firestore_client_wrapper, tools = await initialize_clients_async(user_id)
        
        client_init_time = time.time() - client_init_start
        
        # Time the agent initialization
        print("🤖 Initializing ChatAgent...")
        agent_init_start = time.time()

        agent = ChatAgent(
            storage_client=storage_client,
            secret_client=secret_client,
            model_client=model_client,
            firestore_client_wrapper=firestore_client_wrapper,
            user_id=user_id,
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
        summarize the ai backend progress using github commits for last two weeks
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


