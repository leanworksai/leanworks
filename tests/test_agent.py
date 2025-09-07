from leanworks.agent.chat import ChatAgent
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from anthropic import Anthropic
from google.cloud import bigquery
import logging
import traceback
import time
import asyncio
from typing import Dict, Tuple, Optional
from leanworks.setting import get_client_info
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Performance optimization: Add caching for frequently accessed data
class ClientCache:
    def __init__(self, max_size=100, ttl_seconds=300):  # 5 minutes TTL
        self.cache: Dict[str, Tuple[any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: any):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

# Global caches
client_info_cache = ClientCache()
storage_client_cache = ClientCache()
secret_client_cache = ClientCache()

# Performance optimization: Cached client initialization functions
def get_cached_client_info(bq_client, user_id: str) -> Tuple[str, list, str]:
    """Get client info with caching to reduce database calls"""
    cache_key = f"client_info:{user_id}"
    cached_result = client_info_cache.get(cache_key)
    
    if cached_result:
        logger.info(f"Using cached client info for user: {user_id}")
        return cached_result
    
    logger.info(f"Fetching fresh client info for user: {user_id}")
    try:
        result = get_client_info(bq_client, user_id)
        if result is None:
            logger.error(f"get_client_info returned None for user_id: {user_id}")
            return None, [], ""
        
        # Validate that result is a tuple with at least 2 elements
        if not isinstance(result, tuple) or len(result) < 2:
            logger.error(f"get_client_info returned invalid result type: {type(result)} for user_id: {user_id}")
            return None, [], ""
        
        client_info_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error getting client info for user {user_id}: {str(e)}")
        traceback.print_exc()
        return None, [], ""

def get_cached_storage_client(client_name: str) -> CloudStorage:
    """Get storage client with caching to reduce initialization overhead"""
    cache_key = f"storage:{client_name}"
    cached_client = storage_client_cache.get(cache_key)
    
    if cached_client:
        logger.info(f"Using cached storage client for client: {client_name}")
        return cached_client
    
    logger.info(f"Initializing fresh storage client for client: {client_name}")
    try:
        storage_client = CloudStorage("gcp_credential.json", bucket=client_name)
        storage_client_cache.set(cache_key, storage_client)
        return storage_client
    except Exception as e:
        logger.error(f"Error initializing storage client for client {client_name}: {str(e)}")
        traceback.print_exc()
        raise

def get_cached_secret_client(client_name: str) -> GCPSecretLoader:
    """Get secret client with caching to reduce initialization overhead"""
    cache_key = f"secret:{client_name}"
    cached_client = secret_client_cache.get(cache_key)
    
    if cached_client:
        logger.info(f"Using cached secret client for client: {client_name}")
        return cached_client
    
    logger.info(f"Initializing fresh secret client for client: {client_name}")
    try:
        secret_client = GCPSecretLoader("gcp_credential.json", client_name=client_name)
        secret_client_cache.set(cache_key, secret_client)
        return secret_client
    except Exception as e:
        logger.error(f"Error initializing secret client for client {client_name}: {str(e)}")
        traceback.print_exc()
        raise

# Performance optimization: Batch client initialization
async def initialize_clients_async(bq_client, user_id: str) -> Tuple[CloudStorage, GCPSecretLoader, Anthropic, any, list, str]:
    """Initialize all required clients asynchronously with caching"""
    start_time = time.time()
    
    try:
        # Get client info (cached)
        client_name, available_tools, additional_context = get_cached_client_info(bq_client, user_id)
        
        if not client_name:
            raise ValueError(f"Could not determine client for user_id: {user_id}")
        
        # Initialize clients in parallel using asyncio.gather
        loop = asyncio.get_event_loop()
        
        # Run client initializations in executor to avoid blocking
        storage_client, secret_client = await asyncio.gather(
            loop.run_in_executor(None, get_cached_storage_client, client_name),
            loop.run_in_executor(None, get_cached_secret_client, client_name)
        )
        
        # Get Claude API key (cached)
        claude_api_key = await loop.run_in_executor(None, lambda: secret_client.get("CLAUDE_API_KEY"))
        if not claude_api_key:
            raise ValueError(f"CLAUDE_API_KEY not found for client: {client_name}")
        
        # Initialize Anthropic client
        model_client = Anthropic(api_key=claude_api_key)
        
        # Create BigQuery client wrapper
        class BigQueryClient:
            def __init__(self, bq_client, client_name):
                self.bq_client = bq_client
                self.client_name = client_name
        
        bq_client_wrapper = BigQueryClient(bq_client, client_name)
        
        init_time = time.time() - start_time
        logger.info(f"Client initialization completed in {init_time:.3f}s for user: {user_id}")
        
        return storage_client, secret_client, model_client, bq_client_wrapper, available_tools, additional_context
    except Exception as e:
        logger.error(f"Error in initialize_clients_async for user {user_id}: {str(e)}")
        traceback.print_exc()
        raise

async def main_async():
    """Async version of main function with optimized client initialization"""
    # user_id = "bharathkumar.l@sbnasoftware.com"
    user_id = "yanfu@leanworks.ai"
    
    print("=" * 80)
    print("🚀 AGENT PERFORMANCE TESTING (OPTIMIZED)")
    print("=" * 80)
    
    try:
        # Time the overall setup
        setup_start = time.time()
        
        print("📋 Setting up base BigQuery client...")
        bq_setup_start = time.time()
        bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
        bq_setup_time = time.time() - bq_setup_start
        
        print("⚡ Initializing clients with caching and parallel execution...")
        client_init_start = time.time()
        
        # Use optimized async client initialization
        storage_client, secret_client, model_client, bq_client_wrapper, tools, additional_context = await initialize_clients_async(bq_client, user_id)
        
        client_init_time = time.time() - client_init_start
        
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
            tools=tools,  # Only load what we need
            additional_context=additional_context
        )
        
        agent_init_time = time.time() - agent_init_start
        setup_time = time.time() - setup_start
        
        print(f"✅ Agent initialized in {agent_init_time:.2f}s (total setup: {setup_time:.2f}s)")
        print(f"📊 Detailed timing breakdown:")
        print(f"   - BigQuery client setup: {bq_setup_time:.3f}s")
        print(f"   - Client initialization (cached/parallel): {client_init_time:.3f}s")
        print(f"   - Agent initialization: {agent_init_time:.3f}s")
        print()
        
        # Process a user message with timing
        user_message = '''
         Based on the project timeline, which tasks require my attention?
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
        
        # Print cache statistics
        print("\n📈 CACHE STATISTICS:")
        print(f"   - Client info cache size: {len(client_info_cache.cache)}")
        print(f"   - Storage client cache size: {len(storage_client_cache.cache)}")
        print(f"   - Secret client cache size: {len(secret_client_cache.cache)}")
        
        return response
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

# Performance optimization: Add cache management functions
def clear_all_caches():
    """Clear all caches (useful for debugging and maintenance)"""
    client_info_cache.clear()
    storage_client_cache.clear()
    secret_client_cache.clear()
    print("✅ All caches cleared")

def get_cache_stats():
    """Get cache statistics for monitoring"""
    stats = {
        "client_info_cache_size": len(client_info_cache.cache),
        "storage_client_cache_size": len(storage_client_cache.cache),
        "secret_client_cache_size": len(secret_client_cache.cache)
    }
    return stats

def run_performance_comparison():
    """Run a comparison between cached and non-cached initialization"""
    print("=" * 80)
    print("🔬 PERFORMANCE COMPARISON TEST")
    print("=" * 80)
    
    user_id = "yanfu@leanworks.ai"
    
    # Test 1: Non-cached initialization
    print("📊 Test 1: Non-cached initialization")
    clear_all_caches()
    
    start_time = time.time()
    bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
    client_name, tools, additional_context = get_client_info(bq_client, user_id)
    storage_client = CloudStorage("gcp_credential.json", bucket=client_name)
    secret_client = GCPSecretLoader("gcp_credential.json", client_name=client_name)
    model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
    
    class BigQueryClient:
        def __init__(self, bq_client, client_name):
            self.bq_client = bq_client
            self.client_name = client_name
    
    bq_client_wrapper = BigQueryClient(bq_client, client_name)
    non_cached_time = time.time() - start_time
    
    print(f"   Non-cached initialization time: {non_cached_time:.3f}s")
    
    # Test 2: Cached initialization (second run)
    print("📊 Test 2: Cached initialization (second run)")
    start_time = time.time()
    
    # Use cached functions
    client_name, tools, additional_context = get_cached_client_info(bq_client, user_id)
    storage_client = get_cached_storage_client(client_name)
    secret_client = get_cached_secret_client(client_name)
    model_client = Anthropic(api_key=secret_client.get("CLAUDE_API_KEY"))
    bq_client_wrapper = BigQueryClient(bq_client, client_name)
    
    cached_time = time.time() - start_time
    
    print(f"   Cached initialization time: {cached_time:.3f}s")
    
    # Calculate improvement
    improvement = ((non_cached_time - cached_time) / non_cached_time) * 100
    print(f"   Performance improvement: {improvement:.1f}% faster")
    print(f"   Time saved: {non_cached_time - cached_time:.3f}s")
    
    # Show cache stats
    stats = get_cache_stats()
    print(f"   Cache stats: {stats}")
    
    return non_cached_time, cached_time, improvement

def main():
    """Synchronous wrapper for the async main function"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    # Run performance comparison first
    run_performance_comparison()
    print("\n" + "=" * 80)
    
    # Run the main optimized test
    main()


