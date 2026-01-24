from google.cloud import secretmanager
from google.oauth2 import service_account
from leanworks.rag.chat import Chat
from leanworks.rag.vectordb import PineconeHybridIndex
from leanworks.rag.embedding import GoogleEmbedding
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
        # Load credentials and initialize secret manager
        credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
        secret_client = secretmanager.SecretManagerServiceClient(credentials=credentials)

        # Load project ID
        with open("gcp_credential.json", "r") as f:
            credential_data = json.load(f)
        project_id = credential_data["project_id"]

        # Get secrets
        def get_secret(name):
            full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
            response = secret_client.access_secret_version(name=full_name)
            return response.payload.data.decode("UTF-8")

        model_client = OpenAI(api_key=get_secret("claude-api-key"), base_url="https://api.anthropic.com/v1")

        # Initialize embedding model
        embedding_model = GoogleEmbedding(get_secret("gemini-api-key"))

        # Initialize vector database client
        vectordb_client = PineconeHybridIndex(
            pinecone_key=get_secret("pinecone-api-key"),
            embedding_model_client=embedding_model
        )
        
        # Load hybrid indexes
        vectordb_client.load_hybrid_index(
            dense_index_name=secret_client.client_name + "-dense",
            sparse_index_name=secret_client.client_name + "-sparse"
        )
        
        # Initialize Chat retriever with query rewrite capabilities
        chat_retriever = Chat(
            vectordb_client=vectordb_client,
            storage_client=storage_client,
            model_client=model_client,
            user_id="yanfu@leanworks.ai",
            session_id=str(uuid.uuid4())
        )
        
        # Test query rewrite functionality
        test_query = "find github commits from yanfu from last 7 days"

        
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
        
        # Extract time filters from the test query
        filters = chat_retriever.extract_time_filters(test_query, model_client)
        print(f"Extracted filters: {filters}")
        
        nodes = chat_retriever.retrieve_nodes(rewrites[:3], top_k=10, filters=filters)
        print(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes")
        if hasattr(nodes, 'matches') and nodes.matches:
            for i, match in enumerate(nodes.matches):
                metadata = getattr(match, 'metadata', {})
                if isinstance(metadata, dict):
                    chunk_text = metadata.get('chunk_text', 'No chunk text available')
                else:
                    chunk_text = getattr(metadata, 'chunk_text', 'No chunk text available')
                print(f"Node {i+1}: {chunk_text}")
        else:
            print("No nodes retrieved")
        
        # Test 3: Postprocessing with reranking
        print("\n3. Testing Postprocessing with Reranking:")
        # Initialize read_document_ids set for deduplication
        read_document_ids = set()
        context, sources = chat_retriever.postprocess_nodes(
            nodes, 
            test_query, 
            use_reranker=True, 
            rerank_top_k=5,
            read_document_ids=read_document_ids
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