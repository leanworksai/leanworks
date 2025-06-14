from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from leanworks.rag.chat import Chat
from openai import OpenAI
import logging
import traceback
import json
import uuid

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    try:
        # Initialize storage and secret clients
        storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
        secret_client = GCPSecretLoader("gcp_credential.json", "leanworks")
        embedding_model_api_key = secret_client.get("GEMINI_API_KEY")
        model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
        
        # Initialize Chat retriever with query rewrite capabilities
        chat_retriever = Chat(
            pinecone_api_key=secret_client.get("PINECONE_API_KEY"),
            index_host=secret_client.get("PINECONE_INDEX_HOST"),
            storage_client=storage_client,
            embedding_model_api_key=embedding_model_api_key,
            model_client=model_client,
            user_id="zhuyanfu0712@gmail.com",
            session_id=str(uuid.uuid4())
        )
        
        # Test query rewrite functionality
        test_query = "have we heard back from Sara in email?"
        
        print(f"Testing query rewrite for: '{test_query}'")
        print("=" * 60)
        
        # Test 1: Query rewriting
        print("\n1. Testing Query Rewriting:")
        rewrites = chat_retriever.rewrite_query(test_query, num_rewrites=3)
        print(f"Original query: {test_query}")
        print("Rewritten queries:")
        for i, rewrite in enumerate(rewrites, 1):
            print(f"  {i}. {rewrite}")
        
        # Test 2: Direct node retrieval with multiple queries
        print("\n2. Testing Direct Node Retrieval with Multiple Queries:")
        nodes = chat_retriever.retrieve_nodes(rewrites[:3], top_k=10)
        print(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes")
        
        # Test 3: Postprocessing with reranking
        print("\n3. Testing Postprocessing with Reranking:")
        context, sources = chat_retriever.postprocess_nodes(
            nodes, 
            test_query, 
            use_reranker=True, 
            rerank_top_k=5
        )
        
        print(f"Processed contexts: {len(context)}")
        print(f"Sources: {sources}")
        print(f"Context: {context}")
        
        return context
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + "="*60)
        print("Test completed successfully!")
    else:
        print("\n" + "="*60)
        print("Test failed!") 