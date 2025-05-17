import asyncio
import datetime
from leanworks.storage import CloudStorage
from leanworks.secret import GCPSecretLoader
from openai import OpenAI
from leanworks.rag.chat import AsyncChat
import logging

logger = logging.getLogger(__name__)

class SearchTool:
    """
    Tool that uses the Leanworks API to search for information when other tools
    cannot provide sufficient context.
    """
    storage_client = CloudStorage("gcp_credential.json", bucket="leanworks")
    secret_client = GCPSecretLoader("gcp_credential.json", "leanworks")
    embedding_model_api_key=secret_client.get("GEMINI_API_KEY")
    model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
    chat = AsyncChat(
        pinecone_api_key=secret_client.get("PINECONE_API_KEY"),
        index_host=secret_client.get("PINECONE_INDEX_HOST"),
        storage_client=storage_client,
        embedding_model_api_key=embedding_model_api_key,
        model_client=model_client
    )
        
    @property
    def search_knowledge_property(self):
        description = """
        It will search for relevant documents using the team's knowledge base, based on the query. 
        The response will be a list of documents ordered by relevance to the query, most relevant first.
        Those documents can be used as context to answer the user's question.
        """
        return {
            "type": "custom",
            "name": "search_knowledge",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the knowledge base"
                    }
                },
                "required": ["query"]
            }
        }

    async def async_search_knowledge(self, query: str):
        # Retrieve context
        context = []
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Task 1: Generate query rewrites if needed
            rewrites_task = asyncio.create_task(self.chat.async_rewrite_query(query))
            tasks.append(rewrites_task)
            
            # Task 2: Extract time filters if needed
            filters_task = asyncio.create_task(self.chat.async_extract_time_filters(query))
            tasks.append(filters_task)
                
            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)
                
            # Prepare all queries for retrieval
            all_queries = [query]
            try:
                rewrites = rewrites_task.result()
                all_queries.extend(rewrites)
                logger.info(f"Query rewrites for '{query}': {rewrites}")
            except Exception as e:
                logger.error(f"Error getting query rewrites: {str(e)}")
                    
            # Get time filters for retrieval
            try:
                filters = filters_task.result()
                logger.info(f"Time filters for '{query}': {filters}")
            except Exception as e:
                    logger.error(f"Error getting time filters: {str(e)}")
            
            # Retrieve nodes (running in executor since retrieve_nodes is not async)
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(
                None, 
                lambda: self.chat.retrieve_nodes(all_queries, top_k=10, filters=filters)
            )
            logger.info(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes for query: '{query}'")
            
            # Use async postprocessing with non-blocking reranking
            context, data_sources = await self.chat.async_postprocess_nodes(
                nodes, 
                query, 
                apply_filters=True, 
                use_reranker=True, 
                rerank_top_k=8
            )
            logger.info(f"Postprocessed to {len(context)} context items for query: '{query}'")
        except Exception as e:
            logger.error(f"Error in async context retrieval: {str(e)}")
        
        formatted_context = ""
        # Add document context
        for i, ctx in enumerate(context):
            # Convert timestamp to ISO UTC format if available
            timestamp_str = ""
            if ctx.get("timestamp"):
                try:
                    # Convert timestamp to ISO format
                    timestamp = datetime.datetime.fromtimestamp(ctx["timestamp"], tz=datetime.timezone.utc)
                    timestamp_str = f" (from {timestamp.isoformat()})"
                except (TypeError, ValueError):
                    logger.warning(f"Failed to convert timestamp: {ctx.get('timestamp')}")
                    # If conversion fails, don't include timestamp
                    pass
            
            # Add source information if available
            source_str = ""
            if ctx.get("data_source"):
                source_str = f" - Source: {ctx['data_source']}"
            
            # Add recency indicator - earlier items are more recent
            recency_indicator = f"[DOCUMENT - Date: {timestamp_str}{source_str}]: "
            formatted_context += recency_indicator + ctx["context"] + "\n\n"
        
        # Return just the formatted context without the search quality reflection
        return formatted_context
        
    def search_knowledge(self, query: str):
        """
        Synchronous wrapper for the async search_knowledge method.
        This allows the method to be called from synchronous code.
        """
        try:
            logger.info(f"Executing search_knowledge with query: {query}")
            # Get or create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method in the event loop
            result = loop.run_until_complete(self.async_search_knowledge(query))
            
            # Remove the search quality reflection header
            if result.startswith("Search results (ordered by relevance, most relevant first):"):
                result = result.replace("Search results (ordered by relevance, most relevant first):\n", "", 1)
            
            return result
        except Exception as e:
            logger.error(f"Error in synchronous search_knowledge: {str(e)}")
            return f"Error occurred during knowledge search: {str(e)}"