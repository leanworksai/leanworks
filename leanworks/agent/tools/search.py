import asyncio
from datetime import datetime, timezone
from anthropic import Anthropic
import logging
from typing import List, Dict, Any
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
    
    def __contains__(self, item):
        """Support 'in' operator for string-like behavior."""
        return item in self.formatted_context
    
    def __len__(self):
        """Support len() for string-like behavior."""
        return len(self.formatted_context)

class SearchTool:
    """
    Tool that uses the Leanworks API to search for information when other tools
    cannot provide sufficient context.
    """
    def __init__(self, firestore_client, org_slug, secret_manager_client, read_document_ids: set | None = None, credential_path: str = "gcp_credential.json"):
        # Read project_id from credential file
        import json
        with open(credential_path, "r") as f:
            credential_data = json.load(f)
        project_id = credential_data.get("project_id")
        
        # Helper function to get secret
        def get_secret(name):
            full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
            response = secret_manager_client.access_secret_version(name=full_name)
            return response.payload.data.decode("UTF-8")
        
        model_client = Anthropic(api_key=get_secret("claude-api-key"))
        
        # Use the module-level imports directly
        embedding_model_client = GoogleEmbedding(get_secret("gemini-api-key"))
        
        # Initialize vector database client
        vectordb_client = PineconeHybridIndex(
            pinecone_key=get_secret("pinecone-api-key"),
            embedding_model_client=embedding_model_client
        )
        
        # Use shared indexes with namespaces instead of per-org indexes
        # This matches the data-pipeline pattern to avoid hitting Pinecone index limits
        vectordb_client.load_hybrid_index(
            dense_index_name="leanworks-dense",
            sparse_index_name="leanworks-sparse"
        )
        
        self.chat = AsyncChat(
            vectordb_client=vectordb_client,
            firestore_client=firestore_client,
            org_slug=org_slug,
            model_client=model_client
        )
        # Store org_slug for namespace calculations
        self.org_slug = org_slug
        self.tool_responses_namespace = f"{org_slug}_tool_responses"
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
    
    def _convert_date_to_timestamp(self, date_str: str) -> int:
        """
        Convert date string to Unix timestamp.
        
        Args:
            date_str: Date string in format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
            
        Returns:
            Unix timestamp as integer
        """
        if not date_str:
            return None
            
        try:
            # Handle different date formats
            if 'T' in date_str:
                # ISO format with time
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(date_str)
            else:
                # Date only format
                dt = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Convert to UTC timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return int(dt.timestamp())
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
        
    @property
    def search_documents_property(self):
        description = """
        Search for relevant information using the team's knowledge base and stored tool responses.

        When to use this tool:
        1. As a fallback when domain-specific tools (project_management, doc_management, etc.) return insufficient/empty results, errors, or are otherwise not suitable to answer the question
        2. When you are unsure which specific tool to use, or need to identify relevant resources to guide further actions
        3. When you need to find information from previous tool responses, need more detailed information, or have any uncertainty about the quality or completeness of your answer
        4. When you anticipate needing to call a specific tool multiple times with different queries or scopes, use this tool first to minimize tool iterations.
        5. Always use together with execute_sql_query when searching for progress updates: progress update tables may lack sufficient context. Call both tools in the same turn (see system prompt for details).

        This tool searches across two types of content:
        1. Knowledge base documents (Confluence, Jira, GitHub, Slack, etc.)
        2. Stored tool responses (large outputs from previous tool executions)

        The response will be a list of results ordered by relevance, most relevant first.

        You might need to use this tool multiple times with different queries or scopes.
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
                        "description": "Query to search for relevant information"
                    },
                    "search_scope": {
                        "type": "string",
                        "enum": ["all", "knowledge_base", "tool_responses"],
                        "description": "Scope of search: 'all' (default) searches both knowledge base and tool responses, 'knowledge_base' searches only documents, 'tool_responses' searches only stored tool outputs",
                        "default": "all"
                    },
                    "data_source": {
                        "type": "string",
                        "description": "Optional data source filter (knowledge_base only). One of: confluence, jira, gitlab_issue, gitlab_commits, github_commits, slack, teams, notion, google_doc, google_sheet, servicenow"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Optional tool name filter (tool_responses only). Filter results to specific tool outputs"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date for filtering by timestamp (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date for filtering by timestamp (YYYY-MM-DD)"
                    }
                },
                "required": ["query"]
            }
        }

    async def async_search_documents(self, query: str, data_source: str = None, start_date: str = None, end_date: str = None, search_scope: str = "all", tool_name: str = None):
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

            # Prepare search tasks based on scope
            search_tasks = []

            if search_scope in ["all", "knowledge_base"]:
                # Build filters for knowledge base
                kb_filters = {}
                if data_source:
                    kb_filters["data_source"] = {"$eq": data_source}
                if start_date or end_date:
                    kb_filters["timestamp"] = self._build_timestamp_filter(start_date, end_date)

                # Search knowledge base namespace
                kb_task = self._search_namespace(
                    all_queries,
                    self.org_slug,  # Main namespace
                    top_k=RETRIEVE_TOP_K,
                    filters=kb_filters
                )
                search_tasks.append(("knowledge_base", kb_task))

            if search_scope in ["all", "tool_responses"]:
                # Build filters for tool responses
                tr_filters = {"type": {"$eq": "tool_response"}}
                if tool_name:
                    tr_filters["tool_name"] = {"$eq": tool_name}
                if start_date or end_date:
                    tr_filters["timestamp"] = self._build_timestamp_filter(start_date, end_date)

                # Search tool responses namespace
                tr_task = self._search_namespace(
                    all_queries,
                    self.tool_responses_namespace,
                    top_k=RETRIEVE_TOP_K,
                    filters=tr_filters
                )
                search_tasks.append(("tool_responses", tr_task))

            # Execute searches in parallel
            results_by_source = {}
            for source_type, task in search_tasks:
                try:
                    results = await task
                    results_by_source[source_type] = results
                except Exception as e:
                    logger.error(f"Error searching {source_type}: {e}")
                    results_by_source[source_type] = []

            # Merge results if searching both
            if len(results_by_source) > 1:
                nodes = self._merge_search_results(results_by_source)
            else:
                nodes = list(results_by_source.values())[0] if results_by_source else type('obj', (object,), {'matches': []})()
            logger.info(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes for query: '{query}'")
            
            # Use async postprocessing with non-blocking reranking and deduplication
            context, data_sources = await self.chat.async_postprocess_nodes(
                nodes, 
                query,
                use_span_selection=True,
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
            
            # Also check metadata for timestamp if context extraction fails
            if not timestamp_str and ctx.get("metadata", {}).get("timestamp"):
                try:
                    # Convert Unix timestamp to readable format
                    import time
                    timestamp = ctx["metadata"]["timestamp"]
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        timestamp_str = f" (from {dt.isoformat()})"
                except (ValueError, TypeError):
                    pass
            
            # Add source information if available
            source_str = ""
            if ctx.get("data_source"):
                source_str = ctx['data_source']
            
            # Convert Unix timestamps in the context text to ISO format for better readability
            context_text = self._convert_unix_timestamps_in_text(ctx.get("context", ""))
            
            # If context is empty, try to get information from metadata as fallback
            if not context_text.strip():
                metadata = ctx.get("metadata", {})
                if metadata:
                    # Try to construct context from available metadata
                    context_parts = []
                    for key, value in metadata.items():
                        if value and key not in ['data_source', 'doc_id']:
                            context_parts.append(f"{key}: {value}")
                    
                    if context_parts:
                        context_text = "\n".join(context_parts)
                    else:
                        context_text = "Content not available in this context."
            
            title = f"DOCUMENT - Date: {timestamp_str}, Source: {source_str}, Doc ID: {ctx['doc_id']}"
            formatted_context += f"{title}\n{context_text}\n\n"

        logger.info(f"Formatted context: {formatted_context}")
        # Return both formatted context and data sources
        return {
            "formatted_context": formatted_context,
            "data_sources": data_sources
        }

    async def _search_namespace(
        self,
        queries: List[str],
        namespace: str,
        top_k: int,
        filters: dict = None
    ) -> List[Dict[str, Any]]:
        """
        Search a specific namespace with given queries.

        Args:
            queries: List of query strings (original + rewrites)
            namespace: Pinecone namespace to search
            top_k: Number of results to retrieve
            filters: Optional metadata filters

        Returns:
            List of search results with metadata
        """
        # Run the synchronous retrieve_nodes in an executor to make it non-blocking
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            lambda: self.chat.retrieve_nodes(
                queries,
                top_k=top_k,
                filters=filters,
                namespace=namespace
            )
        )
        return nodes

    def _merge_search_results(
        self,
        results_by_source: Dict[str, List]
    ) -> List:
        """
        Merge results from multiple namespaces using Reciprocal Rank Fusion.

        Args:
            results_by_source: Dict mapping source_type to list of results

        Returns:
            Merged and ranked list of results
        """
        from collections import defaultdict

        # RRF scoring: score = sum(1 / (k + rank)) for each source
        k = 60  # RRF constant
        rrf_scores = defaultdict(float)
        all_results = {}

        for source_type, results in results_by_source.items():
            for rank, result in enumerate(results.matches if hasattr(results, 'matches') else results, start=1):
                result_id = result.id
                rrf_scores[result_id] += 1.0 / (k + rank)

                # Store result with source type metadata
                if result_id not in all_results:
                    # Add source_type to metadata
                    if hasattr(result, 'metadata'):
                        result.metadata['source_type'] = source_type
                    all_results[result_id] = result

        # Sort by RRF score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: rrf_scores[x.id],
            reverse=True
        )

        # Return as SimpleNamespace to match expected format
        from types import SimpleNamespace
        return SimpleNamespace(matches=sorted_results)

    def _build_timestamp_filter(self, start_date: str = None, end_date: str = None) -> dict:
        """
        Build timestamp filter for search queries.

        Args:
            start_date: Optional start date string
            end_date: Optional end date string

        Returns:
            Timestamp filter dict or empty dict if no valid dates
        """
        timestamp_filter = {}
        if start_date:
            start_timestamp = self._convert_date_to_timestamp(start_date)
            if start_timestamp is not None:
                timestamp_filter["$gte"] = start_timestamp
                logger.info(f"Applied start timestamp filter: {start_timestamp}")

        if end_date:
            end_timestamp = self._convert_date_to_timestamp(end_date)
            if end_timestamp is not None:
                timestamp_filter["$lte"] = end_timestamp
                logger.info(f"Applied end timestamp filter: {end_timestamp}")

        if timestamp_filter:
            logger.info(f"Applied timestamp filtering with {len(timestamp_filter)} conditions")

        return timestamp_filter

    def search_documents(self, query: str, data_source: str = None, start_date: str = None, end_date: str = None, search_scope: str = "all", tool_name: str = None):
        """
        Synchronous wrapper for the async search_documents method.
        This allows the method to be called from synchronous code.

        Args:
            query: The search query
            data_source: Optional data source name to filter
            start_date: Optional start date for filtering documents by timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
            end_date: Optional end date for filtering documents by timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
            search_scope: Scope of search ("all", "knowledge_base", or "tool_responses")
            tool_name: Optional tool name filter for tool responses
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
                            data_source=data_source,
                            start_date=start_date,
                            end_date=end_date,
                            search_scope=search_scope,
                            tool_name=tool_name
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
                    data_source=data_source,
                    start_date=start_date,
                    end_date=end_date,
                    search_scope=search_scope,
                    tool_name=tool_name
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