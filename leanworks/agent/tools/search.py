import asyncio
import datetime
from openai import OpenAI
import logging
from leanworks.setting import RETRIEVE_TOP_K, RERANK_TOP_K, APPLY_FILTERS
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
                    },
                    "start_timestamp": {
                        "type": "string",
                        "description": "Optional start of time range in ISO 8601 (e.g., 2025-06-01T00:00:00Z)"
                    },
                    "end_timestamp": {
                        "type": "string",
                        "description": "Optional end of time range in ISO 8601 (e.g., 2025-06-30T23:59:59Z)"
                    }
                },
                "required": ["query"]
            }
        }

    async def async_search_documents(self, query: str, data_source: str = None, start_timestamp: str | int | None = None, end_timestamp: str | int | None = None):
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
            # Build timestamp filter
            def _parse_to_unix_seconds(value):
                try:
                    # Allow ints/floats or numeric strings directly
                    if isinstance(value, (int, float)):
                        return int(value)
                    if isinstance(value, str):
                        stripped = value.strip()
                        # Numeric string
                        if stripped.isdigit():
                            return int(stripped)
                        # ISO 8601 parsing; accept trailing 'Z'
                        iso_str = stripped.replace('Z', '+00:00') if stripped.endswith('Z') else stripped
                        dt = datetime.datetime.fromisoformat(iso_str)
                        # If naive, assume UTC
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=datetime.timezone.utc)
                        return int(dt.timestamp())
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp '{value}': {e}")
                return None

            ts_filter = {}
            if start_timestamp is not None:
                parsed_start = _parse_to_unix_seconds(start_timestamp)
                if parsed_start is not None:
                    ts_filter["$gte"] = parsed_start
            if end_timestamp is not None:
                parsed_end = _parse_to_unix_seconds(end_timestamp)
                if parsed_end is not None:
                    ts_filter["$lte"] = parsed_end
            if ts_filter:
                filters["timestamp"] = ts_filter

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
                read_document_ids=self.read_document_ids
            )
            print(f"context: {context}")
            logger.info(f"Postprocessed to {len(context)} context items for query: '{query}'")
            logger.info(f"Retrieved data sources: {data_sources}")
        except Exception as e:
            logger.error(f"Error in async context retrieval: {str(e)}")
            return {"error": f"async context retrieval failed: {str(e)}"}

        formatted_context = ""
        # Add document context
        for ctx in context:
            # Add source information if available
            source_str = ""
            if ctx.get("data_source"):
                source_str = ctx['data_source']
            
            title = f"DOCUMENT - Date: {ctx['timestamp']}, Source: {source_str}, Doc ID: {ctx['doc_id']}"
            formatted_context += f"{title}\n{ctx['context']}\n\n"
        
        # Return both formatted context and data sources
        return {
            "formatted_context": formatted_context,
            "data_sources": data_sources
        }
        
    def search_documents(self, query: str, data_source: str = None, start_timestamp: str | int | None = None, end_timestamp: str | int | None = None):
        """
        Synchronous wrapper for the async search_documents method.
        This allows the method to be called from synchronous code.
        
        Args:
            query: The search query
            data_source: Optional data source name to filter
            start_timestamp: Optional start of time range (Unix timestamp)
            end_timestamp: Optional end of time range (Unix timestamp)
            read_document_ids: Set of document IDs already read to skip duplicates
        """
        try:
            logger.info(f"Executing search_documents with query: {query}")
            logger.info(f"Using shared read_document_ids length: {len(self.read_document_ids)}")
            # Get or create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method in the event loop
            result = loop.run_until_complete(self.async_search_documents(
                query=query,
                data_source=data_source,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
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
            error_message = f"Error occurred during documents search: {str(e)}"
            return SearchResult(error_message, [])