from leanworks.rag.chat import Chat, AsyncChat
from leanworks.storage.gcs import CloudStorage
from google import genai
import uuid
from openai import OpenAI
from leanworks.secret import GCPSecretLoader
import time
import pandas as pd
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
query = "are there resource challenge in Android app development?"
# query = "based on our customer interviews, who has the most potential to be our customer?"
# query = "how was the interview with Alan"
# query = "give me a summary of iOS app progress so far"
def test_sync_chat():
    storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
    secret_client = GCPSecretLoader("gcp_credential.json", "leanworks")
    embedding_model_api_key=secret_client.get("GEMINI_API_KEY")
    model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
    session_id = str(uuid.uuid4())

    # Initialize RAG
    chat = Chat(
        pinecone_api_key=secret_client.get("PINECONE_API_KEY"),
        index_host=secret_client.get("PINECONE_INDEX_HOST"),
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
    storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
    secret_client = GCPSecretLoader("gcp_credential.json", "leanworks")
    embedding_model_api_key=secret_client.get("GEMINI_API_KEY")
    model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
    # session_id = str(uuid.uuid4())
    session_id = "deu2tp892fhg"

    # Initialize AsyncRAG
    async_chat = AsyncChat(
        pinecone_api_key=secret_client.get("PINECONE_API_KEY"),
        index_host=secret_client.get("PINECONE_INDEX_HOST"),
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
    # await test_async_chat()

if __name__ == "__main__":
    asyncio.run(main())