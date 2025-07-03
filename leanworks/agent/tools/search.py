import asyncio
import datetime
from openai import OpenAI
from leanworks.rag.chat import AsyncChat
from leanworks.rag.vectordb import PineconeHybridIndex
from leanworks.rag.embedding import GoogleEmbedding
import logging
from leanworks.setting import RETRIEVE_TOP_K, RERANK_TOP_K

logger = logging.getLogger(__name__)

class SearchResult:
    """
    Custom class to hold search results with both formatted context and data sources.
    This class behaves like a string for backward compatibility while also storing metadata.
    """
    def __init__(self, formatted_context: str, data_sources: list = None):
        self.formatted_context = formatted_context
        self._search_data_sources = data_sources or []
    
    def __str__(self):
        return self.formatted_context
    
    def __repr__(self):
        return f"SearchResult(context_length={len(self.formatted_context)}, sources={len(self._search_data_sources)})"

class SearchTool:
    """
    Tool that uses the Leanworks API to search for information when other tools
    cannot provide sufficient context.
    """
    def __init__(self, storage_client, secret_client):
        
        model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
        embedding_model_client = GoogleEmbedding(secret_client.get("GEMINI_API_KEY"))
        
        # Initialize vector database client
        vectordb_client = PineconeHybridIndex(
            pinecone_key=secret_client.get("PINECONE_API_KEY"),
            embedding_model_client=embedding_model_client
        )
        
        # Load hybrid indexes
        vectordb_client.load_hybrid_index(
            dense_index_name=secret_client.client_name + "-dense",
            sparse_index_name=secret_client.client_name + "-sparse"
        )
        
        self.chat = AsyncChat(
            vectordb_client=vectordb_client,
            storage_client=storage_client,
            model_client=model_client
        )
        
    @property
    def search_knowledge_property(self):
        description = """
        Search for relevant documents using the team's knowledge base, based on the query. The response will be a list of documents ordered by relevance to the query, most relevant first.
        You MUST ALWAYS use this tool as the fallback when any of these conditions occur:
        - Other tools return empty or insufficient results
        - You have ANY uncertainty about the completeness of your answer
        - More detailed information is needed to answer the question
        You might need to use this tool multiple times with different queries to fully answer the question.
        NEVER skip this tool if the above conditions are met.
        Do not reflect on the quality of the returned search results in your response
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

    async def async_search_knowledge(self, query: str, read_document_ids: set = None):
        # Retrieve context
        context = []
        data_sources = []
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
                lambda: self.chat.retrieve_nodes(all_queries, top_k=RETRIEVE_TOP_K, filters=filters)
            )
            logger.info(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes for query: '{query}'")
            
            # Use async postprocessing with non-blocking reranking and deduplication
            context, data_sources = await self.chat.async_postprocess_nodes(
                nodes, 
                query, 
                apply_filters=True, 
                use_reranker=True, 
                rerank_top_k=RERANK_TOP_K,
                read_document_ids=read_document_ids
            )
            print(f"context: {context}")
            logger.info(f"Postprocessed to {len(context)} context items for query: '{query}'")
            logger.info(f"Retrieved data sources: {data_sources}")
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
        
        # Return both formatted context and data sources
        return {
            "formatted_context": formatted_context,
            "data_sources": data_sources
        }
        
    def search_knowledge(self, query: str, read_document_ids: set = None):
        """
        Synchronous wrapper for the async search_knowledge method.
        This allows the method to be called from synchronous code.
        
        Args:
            query: The search query
            read_document_ids: Set of document IDs already read to skip duplicates
        """
        try:
            logger.info(f"Executing search_knowledge with query: {query}")
            logger.info(f"Received read_document_ids: {read_document_ids}")
            logger.info(f"read_document_ids is None: {read_document_ids is None}")
            if read_document_ids is not None:
                logger.info(f"read_document_ids length: {len(read_document_ids)}")
                logger.info(f"read_document_ids id: {id(read_document_ids)}")
            # Get or create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method in the event loop
            result = loop.run_until_complete(self.async_search_knowledge(query, read_document_ids))
            
            # Extract the formatted context for backward compatibility
            formatted_context = result["formatted_context"]
            
            # Remove the search quality reflection header if present
            if formatted_context.startswith("Search results (ordered by relevance, most relevant first):"):
                formatted_context = formatted_context.replace("Search results (ordered by relevance, most relevant first):\n", "", 1)
            
            # Return SearchResult object that behaves like a string but also stores data sources
            return SearchResult(formatted_context, result["data_sources"])
            
        except Exception as e:
            logger.error(f"Error in synchronous search_knowledge: {str(e)}")
            error_message = f"Error occurred during knowledge search: {str(e)}"
            return SearchResult(error_message, [])