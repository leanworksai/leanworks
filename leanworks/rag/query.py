import json
import logging
import asyncio
from typing import List
from leanworks.setting import OTHER_MODEL, QUERY_REWRITE_MODEL_SYSTEM_PROMPT

# Set up logging
logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    QueryRewriter class for rewriting queries to improve retrieval recall.
    Provides both synchronous and asynchronous functionality.
    """
    
    def __init__(self, model_client):
        """
        Initialize QueryRewriter with model client.
        
        Args:
            model_client: Initialized client for LLM generation
        """
        self.model_client = model_client
        logger.info("QueryRewriter initialized successfully")
    
    def rewrite_query(self, query: str, num_rewrites: int = 3, model: str = OTHER_MODEL) -> List[str]:
        """
        Rewrite the original query into multiple diverse variants to improve retrieval recall.
        
        Args:
            query: The original user query
            num_rewrites: Number of query rewrites to generate
            model: The model to use for generating rewrites
            
        Returns:
            List of rewritten queries
        """
        logger.info(f"Generating {num_rewrites} rewrites for query: '{query}'")
        
        user_prompt = f"Original Query: {query}\nNumber of rewrites: {num_rewrites}"
        
        try:
            response = self.model_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": QUERY_REWRITE_MODEL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Use some temperature for diversity
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            logger.debug(f"MQR response: {result}")
            
            # Parse JSON response
            try:
                rewrites_data = json.loads(result)
                rewrites = rewrites_data.get("rewrites", [])
                
                # Ensure we have at least one rewrite
                if not rewrites:
                    logger.warning("No rewrites received from model, using original query")
                    return [query]
                
                return [query] + rewrites
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return [query]
                
        except Exception as e:
            logger.error(f"Error generating query rewrites: {str(e)}")
            # Return original query if rewriting fails
            return [query]
            
    async def async_rewrite_query(self, query: str, num_rewrites: int = 3, model: str = OTHER_MODEL) -> List[str]:
        """
        Asynchronous version of rewrite_query that generates query rewrites without blocking.
        
        Args:
            query: The original user query
            num_rewrites: Number of query rewrites to generate
            model: The model to use for generating rewrites
            
        Returns:
            List of rewritten queries
        """
        logger.info(f"Asynchronously generating {num_rewrites} rewrites for query: '{query}'")
        
        user_prompt = f"Original Query: {query}\nNumber of rewrites: {num_rewrites}"
        
        try:
            # Run the model call in executor to make it non-blocking
            loop = asyncio.get_event_loop()
            
            # For consistent results with sync version, use the exact same parameters
            response = await loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": QUERY_REWRITE_MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,  # Use same temperature as sync version
                    response_format={"type": "json_object"}
                )
            )
            
            result = response.choices[0].message.content
            logger.debug(f"MQR response: {result}")
            
            # Parse JSON response with same logic as sync version
            try:
                rewrites_data = json.loads(result)
                rewrites = rewrites_data.get("rewrites", [])
                
                # Ensure we have at least one rewrite
                if not rewrites:
                    logger.warning("No rewrites received from model, using original query")
                    return [query]
                
                return rewrites
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return [query]
                
        except Exception as e:
            logger.error(f"Error generating query rewrites: {str(e)}")
            # Return original query if rewriting fails
            return [query]
        
