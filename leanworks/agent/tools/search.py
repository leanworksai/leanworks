import asyncio
import datetime
from openai import OpenAI
import logging
from leanworks.setting import RETRIEVE_TOP_K, RERANK_TOP_K
from leanworks.rag.embedding import GoogleEmbedding
from leanworks.rag.vectordb import PineconeHybridIndex
from leanworks.rag.chat import AsyncChat


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
    def __init__(self, storage_client, secret_client, read_document_ids: set | None = None):
        
        model_client = OpenAI(api_key=secret_client.get("CLAUDE_API_KEY"), base_url="https://api.anthropic.com/v1")
        
        # Use the module-level imports directly
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
        # Shared deduplication set used across searches
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
    
    def _convert_unix_timestamps_in_text(self, text: str) -> str:
        """Convert Unix timestamps in text to ISO format for better readability."""
        import re
        from datetime import datetime
        
        def replace_timestamp(match):
            prefix = match.group(1)  # The part before the timestamp
            timestamp_str = match.group(2)  # The actual timestamp
            try:
                # Convert Unix timestamp to datetime
                unix_timestamp = float(timestamp_str)
                # Handle both seconds and milliseconds
                if unix_timestamp > 1e10:  # Likely milliseconds
                    unix_timestamp = unix_timestamp / 1000
                dt = datetime.fromtimestamp(unix_timestamp)
                iso_format = dt.isoformat()
                return f"{prefix}{iso_format}"
            except (ValueError, OSError):
                return match.group(0)  # Return original if conversion fails
        
        # Pattern to match Unix timestamps in various contexts
        # Matches patterns like: timestamp is 1756087205.146079, [0].timestamp is 1756087205.146079, etc.
        unix_pattern = r'(\w*[Tt]imestamp\s*is\s*)(\d{10,13}(?:\.\d+)?)'
        text = re.sub(unix_pattern, replace_timestamp, text)
        
        return text
        
    @property
    def search_documents_property(self):
        description = """
        Search for relevant documents using the team's knowledge base, based on the query. The response will be a list of documents ordered by relevance to the query, most relevant first.
        You should use this tool as when any of these conditions occur:
        - Other tools are not suitable to answer the question
        - Other tools return empty, error or insufficient results
        - You have used the search_documents tool before and the answer is still not satisfactory. 
        - You have ANY uncertainty about the quality of your answer
        - More detailed information is needed to answer the question
        - You need to perform search for details of a specific data source when the corresponding tool result is too large to display
        When you use this tool, find what are the missing information from the last response (if any) and try to search (call search_documents tool) with a different query 
        so that it can surface more information to help refine your answer.
        You might need to use this tool multiple times with different queries to fully answer the question.
        NEVER skip this tool if the above conditions are met.
        Do not reflect on the quality of the returned search results in your response
        """
        return {
            "type": "custom",
            "name": "search_documents",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the knowledge base"
                    },
                    "data_source": {
                        "type": "string",
                        "description": "Optional data source name to filter documents. Can only be one of the following: confluence, jira, gitlab_issue, gitlab_commits, github_commits, slack, teams, notion, google_doc, google_sheet, servicenow"
                    }
                },
                "required": ["query"]
            }
        }

    async def async_search_documents(self, query: str, data_source: str = None):
        # Retrieve context
        context = []
        data_sources = []
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Task 1: Generate query rewrites if needed
            rewrites_task = asyncio.create_task(self.chat.async_rewrite_query(query))
            tasks.append(rewrites_task)
                
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
                    
            
            # Build filters from explicit arguments
            filters = {}
            if data_source:
                filters["data_source"] = {"$eq": data_source}
            # Note: Timestamp filtering removed as timestamp fields are no longer used in context structure
            # Timestamp information is now extracted from context text when needed for display
            logger.info(f"Search filters: {filters}")
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
                rerank_top_k=RERANK_TOP_K,
                read_document_ids=self.read_document_ids
            )
            logger.info(f"Postprocessed to {len(context)} context items for query: '{query}'")
            logger.info(f"Retrieved data sources: {data_sources}")
        except Exception as e:
            logger.error(f"Error in async context retrieval: {str(e)}")
            # Return only the error message without full details
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

        formatted_context = ""
        # Add document context
        for ctx in context:
            # Extract timestamp from context text if available
            timestamp_str = ""
            extracted_timestamp = self.chat._extract_timestamp_from_context(ctx.get("context", ""))
            if extracted_timestamp:
                timestamp_str = f" (from {extracted_timestamp})"
            
            # Add source information if available
            source_str = ""
            if ctx.get("data_source"):
                source_str = ctx['data_source']
            
            # Convert Unix timestamps in the context text to ISO format for better readability
            context_text = self._convert_unix_timestamps_in_text(ctx.get("context", ""))
            
            title = f"DOCUMENT - Date: {timestamp_str}, Source: {source_str}, Doc ID: {ctx['doc_id']}"
            formatted_context += f"{title}\n{context_text}\n\n"
        # Return both formatted context and data sources
        return {
            "formatted_context": formatted_context,
            "data_sources": data_sources
        }
        
    def search_documents(self, query: str, data_source: str = None):
        """
        Synchronous wrapper for the async search_documents method.
        This allows the method to be called from synchronous code.
        
        Args:
            query: The search query
            data_source: Optional data source name to filter
        """
        try:
            logger.info(f"Executing search_documents with query: {query}")
            logger.info(f"Using shared read_document_ids length: {len(self.read_document_ids)}")
            
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an event loop, so we need to use a thread executor
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create a new event loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.async_search_documents(
                            query=query,
                            data_source=data_source
                        ))
                    finally:
                        new_loop.close()
                
                # Run the async method in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result(timeout=30)  # 30 second timeout
                    
            except RuntimeError:
                # No event loop running, we can safely create one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If there's no event loop in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async method in the event loop
                result = loop.run_until_complete(self.async_search_documents(
                    query=query,
                    data_source=data_source
                ))
            
            # If async layer returned an error, surface it directly
            if isinstance(result, dict) and "error" in result:
                error_message = f"Error: {result['error']}"
                return SearchResult(error_message, [])

            # Extract the formatted context for backward compatibility
            formatted_context = result["formatted_context"]
            
            # Remove the search quality reflection header if present
            if formatted_context.startswith("Search results (ordered by relevance, most relevant first):"):
                formatted_context = formatted_context.replace("Search results (ordered by relevance, most relevant first):\n", "", 1)
            
            # Return SearchResult object that behaves like a string but also stores data sources
            return SearchResult(formatted_context, result["data_sources"])
            
        except Exception as e:
            logger.error(f"Error in synchronous search_documents: {str(e)}")
            # Return only the error message without full details
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            error_message = f"Error occurred during documents search: {error_msg}"
            return SearchResult(error_message, [])