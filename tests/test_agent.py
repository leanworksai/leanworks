from leanworks.agent.chat import ChatAgent
from google.cloud import firestore, secretmanager
from google.oauth2 import service_account
from anthropic import Anthropic
import logging
import traceback
import time
import asyncio
import json
import subprocess
import socket
import os
import atexit
from typing import Dict, Tuple, Optional
import re
logger = logging.getLogger(__name__)

# Global reference to Cloud SQL Proxy process for cleanup
_cloud_sql_proxy_process = None

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_cloud_sql_proxy(credential_path: str = "gcp_credential.json") -> subprocess.Popen:
    """
    Start Cloud SQL Proxy for local development if not already running.
    
    Returns:
        subprocess.Popen: The proxy process, or None if already running
    """
    global _cloud_sql_proxy_process
    
    # Check if we're in k8s (has /cloudsql directory)
    if os.path.exists("/cloudsql"):
        logger.info("Running in k8s environment, Cloud SQL Proxy not needed")
        return None
    
    # Check if proxy is already running on port 5432
    if is_port_in_use(5432):
        logger.info("Port 5432 already in use, assuming Cloud SQL Proxy is running")
        return None
    
    # Read project ID from credentials
    try:
        with open(credential_path, "r") as f:
            credential_data = json.load(f)
        project_id = credential_data.get("project_id", "leanworks-474204")
    except Exception as e:
        logger.warning(f"Could not read project_id from {credential_path}: {e}")
        project_id = "leanworks-474204"
    
    region = os.getenv("DB_REGION", "us-west1")
    instance = os.getenv("DB_INSTANCE", "leanworks-prod")
    instance_connection = f"{project_id}:{region}:{instance}"
    
    logger.info(f"Starting Cloud SQL Proxy for {instance_connection}...")
    
    # Try cloud-sql-proxy (newer) first, then cloud_sql_proxy (older)
    proxy_commands = [
        ["cloud-sql-proxy", instance_connection, "--port", "5432", f"--credentials-file={credential_path}"],
        ["cloud_sql_proxy", f"-instances={instance_connection}=tcp:5432", f"-credential_file={credential_path}"],
    ]
    
    for cmd in proxy_commands:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait a bit for the proxy to start
            time.sleep(2)
            
            # Check if it's running
            if process.poll() is None and is_port_in_use(5432):
                logger.info(f"Cloud SQL Proxy started successfully (PID: {process.pid})")
                _cloud_sql_proxy_process = process
                
                # Register cleanup on exit
                atexit.register(stop_cloud_sql_proxy)
                
                return process
            else:
                # Process died, try next command
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    logger.debug(f"Proxy command failed: {cmd[0]} - {stderr}")
                    
        except FileNotFoundError:
            logger.debug(f"Proxy command not found: {cmd[0]}")
            continue
        except Exception as e:
            logger.debug(f"Error starting proxy with {cmd[0]}: {e}")
            continue
    
    logger.warning("Could not start Cloud SQL Proxy - database features may not work")
    return None

def stop_cloud_sql_proxy():
    """Stop the Cloud SQL Proxy if we started it."""
    global _cloud_sql_proxy_process
    if _cloud_sql_proxy_process:
        logger.info("Stopping Cloud SQL Proxy...")
        try:
            _cloud_sql_proxy_process.terminate()
            _cloud_sql_proxy_process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error stopping Cloud SQL Proxy: {e}")
        _cloud_sql_proxy_process = None
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def initialize_clients_async(user_id: str, org_slug: str, tools: Optional[list] = None) -> Tuple[firestore.Client, secretmanager.SecretManagerServiceClient, Anthropic, list]:
    """Initialize all required clients asynchronously"""
    start_time = time.time()
    
    try:
        logger.info(f"Initializing clients for user {user_id} in org {org_slug}")
        
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
        logger.info(f"Initializing Firestore and Secret Manager clients for org: {org_slug}")
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
            raise ValueError(f"claude-api-key not found for org: {org_slug}")
        
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
    org_slug = "leanworksai_mj6bu7m8"
    
    print("=" * 80)
    print("🚀 AGENT PERFORMANCE TESTING")
    print("=" * 80)
    
    # Auto-start Cloud SQL Proxy for local development
    print("🔌 Checking Cloud SQL Proxy...")
    start_cloud_sql_proxy()
    
    try:
        # Time the overall setup
        setup_start = time.time()
        
        print("⚡ Initializing clients with parallel execution...")
        client_init_start = time.time()
        
        # Use async client initialization (no BigQuery setup needed)
        firestore_client, secret_manager_client, model_client, tools = await initialize_clients_async(user_id, org_slug, ["jira", "github"])
        
        client_init_time = time.time() - client_init_start
        
        # Time the agent initialization
        print("🤖 Initializing ChatAgent...")
        agent_init_start = time.time()

        agent = ChatAgent(
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            model_client=model_client,
            user_id=user_id,
            org_slug=org_slug,
            session_id="fhr3p9gf7g",
            clear_conversation=False,  # Set to False to test loading previous conversations
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
       plz use yanfuzhu94 to search
'''
        
        print("💬 Processing user message:")
        print(f"   Query: {user_message.strip()}")
        print(f"   Thinking mode: False (evaluation disabled for speed)")
        print(f"   Streaming mode: True (shows tools and streams response)")
        print()
        
        # Print memories before processing
        print("🔍 MEMORIES BEFORE PROCESSING:")
        print_all_memories(agent)
        
        # Time the response processing
        response_start = time.time()
        
        response = agent.process_message(user_message, thinking=False, streaming=True)
        
        # Print memories after processing
        print("🔍 MEMORIES AFTER PROCESSING:")
        print_all_memories(agent)
        
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

def print_all_memories(agent):
    """Print all memories being used by the agent"""
    if not agent.memory_manager:
        print("\n" + "=" * 80)
        print("🧠 MEMORY STATUS: Memory manager not enabled")
        print("=" * 80 + "\n")
        return
    
    print("\n" + "=" * 80)
    print("🧠 MEMORY INFORMATION")
    print("=" * 80)
    
    # Get memory context and recent messages
    memory_context, recent_messages = agent.memory_manager.get_context_for_inference()
    
    # Print memory statistics
    stats = agent.memory_manager.get_memory_stats()
    print("\n📊 MEMORY STATISTICS:")
    print(f"   Total Tokens:           {stats.get('total_tokens', 0):,}")
    print(f"   Summary Tokens:         {stats.get('summary_tokens', 0):,}")
    print(f"   Conversation Turns:     {stats.get('conversation_turns', 0)}")
    print(f"   Summary Length:         {stats.get('summary_length', 0):,} characters")
    print(f"   Trigger Threshold:      {stats.get('trigger_threshold', 0):,} tokens")
    print(f"   Max Context Tokens:     {stats.get('max_context_tokens', 0):,} tokens")
    print(f"   Tokens Until Trigger:   {stats.get('tokens_until_trigger', 0):,} tokens")
    print(f"   Last Summarization:     {stats.get('last_summarization', 'Never')}")
    
    # Print memory context (system prompt + user profile + running summary)
    print("\n📝 MEMORY CONTEXT (System Prompt + User Profile + Running Summary):")
    print("-" * 80)
    if memory_context:
        # Truncate if too long for readability
        context_preview = memory_context[:2000] + "..." if len(memory_context) > 2000 else memory_context
        print(context_preview)
        if len(memory_context) > 2000:
            print(f"\n... (truncated, full length: {len(memory_context):,} characters)")
    else:
        print("(No memory context available)")
    
    # Print recent messages
    print(f"\n💬 RECENT MESSAGES ({len(recent_messages)} messages):")
    print("-" * 80)
    if recent_messages:
        for i, msg in enumerate(recent_messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", [])
            
            # Extract text from content
            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown_tool")
                            text_parts.append(f"[Used tool: {tool_name}]")
                        elif block.get("type") == "tool_result":
                            result_preview = str(block.get("content", ""))[:100]
                            text_parts.append(f"[Tool result: {result_preview}...]")
            
            message_text = " ".join(text_parts)
            # Truncate long messages
            if len(message_text) > 500:
                message_text = message_text[:500] + "..."
            
            print(f"\n   Message {i} ({role.upper()}):")
            print(f"   {message_text}")
    else:
        print("(No recent messages)")
    
    # Print running summary separately if it exists
    if agent.memory_manager.running_summary:
        print(f"\n📚 RUNNING SUMMARY ({len(agent.memory_manager.running_summary):,} characters):")
        print("-" * 80)
        summary_preview = agent.memory_manager.running_summary[:1000] + "..." if len(agent.memory_manager.running_summary) > 1000 else agent.memory_manager.running_summary
        print(summary_preview)
        if len(agent.memory_manager.running_summary) > 1000:
            print(f"\n... (truncated, full length: {len(agent.memory_manager.running_summary):,} characters)")
    
    # Print user profile if it exists
    user_profile = agent.memory_manager.get_user_profile_dict()
    if user_profile:
        print(f"\n👤 USER PROFILE:")
        print("-" * 80)
        print(json.dumps(user_profile, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Synchronous wrapper for the async main function"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    main()


