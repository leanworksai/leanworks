import asyncio
import logging
import os
import json
from leanworks.rag.chat import Chat
from leanworks.rag.advanced_chat import VerificationChain
from google.generativeai import configure, GenerativeModel
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example usage of Chain of Verification
async def main():
    # Configure API keys (replace with your own keys)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_host = os.environ.get("PINECONE_INDEX_HOST")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not all([pinecone_api_key, pinecone_index_host, openai_api_key]):
        logger.error("Missing required API keys")
        return
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Initialize Google Generative AI for embeddings
    configure(api_key=google_api_key)
    embedding_model = GenerativeModel("embedding-001")
    
    # Initialize Cloud Storage client (mock for example)
    storage_client = None  # Replace with actual storage client
    
    # Initialize Chat module
    chat = Chat(
        pinecone_api_key=pinecone_api_key,
        index_host=pinecone_index_host,
        storage_client=storage_client,
        embedding_model_client=embedding_model,
        model_client=openai_client,
        use_reranker=True,
        use_verifier=True
    )
    
    # Initialize Chain of Verification
    verification_chain = VerificationChain(
        chat=chat,
        model_client=openai_client,
        embedding_model_client=embedding_model,
        max_claims=5,
        verification_confidence_threshold=0.7
    )
    
    # Example query
    query = "When did SpaceX first land a Falcon 9 on a droneship, and what was the droneship's name?"
    
    # Process query through Chain of Verification
    print(f"Processing query: {query}")
    print("This may take a minute as it performs multiple steps...")
    
    result = await verification_chain.process_query(
        query=query,
        generation_model="gpt-4o",
        include_memory=False,
        top_k=12,
        rerank_top_k=5
    )
    
    # Print results
    print("\n==== CHAIN OF VERIFICATION RESULT ====")
    print(f"\nFinal Answer:\n{result['content']}")
    print(f"\nData Sources: {result['data_sources']}")
    
    verification_meta = result.get("verification_meta", {})
    print(f"\nWas Corrected: {verification_meta.get('was_corrected', False)}")
    print(f"Claims Found: {verification_meta.get('claims_found', 0)}")
    
    # Optionally print full verification results
    if verification_meta.get("verification_results"):
        print("\nVerification Results Summary:")
        for claim in verification_meta.get("verification_results", []):
            verification = claim.get("verification", {})
            print(f"  {claim.get('id')}: {verification.get('status')} (confidence: {verification.get('confidence', 0.0):.2f})")
    
    print("\n====================================")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main()) 