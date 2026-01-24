from leanworks.rag.chat import Chat, AsyncChat
from leanworks.rag.vectordb import PineconeHybridIndex
from leanworks.rag.embedding import GoogleEmbedding
from google import genai
import uuid
from openai import OpenAI
from google.cloud import secretmanager
from google.oauth2 import service_account
import time
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# query = "what is the project progress?"
# query = "go to market"
# query = "are there resource challenge in Android app development?"
# query = "based on our customer interviews, who has the most potential to be our customer?"
# query = "how was the interview with Alan"
query = "give me a summary of iOS app progress so far"
def test_sync_chat():
    # Load credentials and initialize secret manager
    credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
    secret_client = secretmanager.SecretManagerServiceClient(credentials=credentials)

    # Load project ID
    with open("gcp_credential.json", "r") as f:
        import json
        credential_data = json.load(f)
    project_id = credential_data["project_id"]

    # Get secrets
    def get_secret(name):
        full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
        response = secret_client.access_secret_version(name=full_name)
        return response.payload.data.decode("UTF-8")

    embedding_model_api_key = get_secret("gemini-api-key")
    model_client = OpenAI(api_key=get_secret("claude-api-key"), base_url="https://api.anthropic.com/v1")
    # session_id = str(uuid.uuid4())
    session_id = "cwjhh[fp984]"

    # Initialize embedding model
    embedding_model = GoogleEmbedding(embedding_model_api_key)

    # Initialize vector database client
    vectordb_client = PineconeHybridIndex(
        pinecone_key=get_secret("pinecone-api-key"),
        embedding_client=embedding_model
    )

    # Load hybrid indexes
    vectordb_client.load_hybrid_index(
        dense_index_name=get_secret("dense-index-name") if get_secret("dense-index-name") else "leanworks-dense",
        sparse_index_name=get_secret("sparse-index-name") if get_secret("sparse-index-name") else "leanworks-sparse"
    )
    
    # Initialize RAG
    chat = Chat(
        vectordb_client=vectordb_client,
        storage_client=storage_client,
        embedding_model_api_key=embedding_model_api_key,
        model_client=model_client,
        user_id="zhuyanfu0712@gmail.com",
        session_id=session_id
    )
    start_time = time.time()
    response = chat.get_response(query)
    elapsed_time = time.time() - start_time
    print(f"Sync Chat Response (took {elapsed_time:.2f} seconds):")
    print(response["content"])
    print("\n" + "-"*50 + "\n")

async def test_async_chat():
    # Load credentials and initialize secret manager
    credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
    secret_client = secretmanager.SecretManagerServiceClient(credentials=credentials)

    # Load project ID
    with open("gcp_credential.json", "r") as f:
        import json
        credential_data = json.load(f)
    project_id = credential_data["project_id"]

    # Get secrets
    def get_secret(name):
        full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
        response = secret_client.access_secret_version(name=full_name)
        return response.payload.data.decode("UTF-8")

    embedding_model_api_key = get_secret("gemini-api-key")
    model_client = OpenAI(api_key=get_secret("claude-api-key"), base_url="https://api.anthropic.com/v1")
    session_id = str(uuid.uuid4())
    # session_id = "deu2tp892fhg"

    # Initialize embedding model
    embedding_model = GoogleEmbedding(embedding_model_api_key)

    # Initialize vector database client
    vectordb_client = PineconeHybridIndex(
        pinecone_key=get_secret("pinecone-api-key"),
        embedding_client=embedding_model
    )

    # Load hybrid indexes
    vectordb_client.load_hybrid_index(
        dense_index_name=get_secret("dense-index-name") if get_secret("dense-index-name") else "leanworks-dense",
        sparse_index_name=get_secret("sparse-index-name") if get_secret("sparse-index-name") else "leanworks-sparse"
    )
    
    # Initialize AsyncRAG
    async_chat = AsyncChat(
        vectordb_client=vectordb_client,
        storage_client=storage_client,
        embedding_model_api_key=embedding_model_api_key,
        model_client=model_client,
        user_id="zhuyanfu0712@gmail.com",
        session_id=session_id
    )
    start_time = time.time()
    response = await async_chat.async_get_response(query)
    elapsed_time = time.time() - start_time
    print(f"Async Chat Response (took {elapsed_time:.2f} seconds):")
    print(response["content"])

async def main():
    # Run synchronous test
    test_sync_chat()
    
    # Run asynchronous test
    await test_async_chat()

if __name__ == "__main__":
    asyncio.run(main())