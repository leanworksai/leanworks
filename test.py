from leanworks.rag.chat import Chat
from leanworks.storage.gcs import CloudStorage
from google import genai
import uuid
from openai import OpenAI
from leanworks.secret import GCPSecretLoader
import logging
import os

# Configure logging to write to both console and file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "test.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG application")
    storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
    secret_client = GCPSecretLoader("gcp_credential.json", "leanworks")
    
    logger.info("Loading API keys and initializing clients")
    embedding_model_client = genai.Client(api_key=secret_client.get("GEMINI_API_KEY"))
    model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
    session_id = str(uuid.uuid4())
    logger.info(f"Generated session ID: {session_id}")

    # Initialize RAG
    logger.info("Initializing Chat")
    chat = Chat(
        pinecone_api_key=secret_client.get("PINECONE_API_KEY"),
        index_host=secret_client.get("PINECONE_INDEX_HOST"),
        storage_client=storage_client,
        embedding_model_client=embedding_model_client,
        model_client=model_client,
        user_id="zhuyanfu0712@gmail.com",
        session_id=session_id
    )
    query = "list all learnings from our customer interviews"
    logger.info(f"Sending query: {query}")
    response1 = chat.get_response(query)
    logger.info("Query complete, printing response")
    print(response1["content"])

if __name__ == "__main__":
    main()
