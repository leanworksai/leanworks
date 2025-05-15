from pinecone import Pinecone
from typing import List, Dict, Tuple, Any
from leanworks.rag.filters import FilterExtractor
from leanworks.rag.memory import MemoryManager
from leanworks.rag.reranker import CrossEncoderReranker
from leanworks.rag.setting import *
from leanworks.rag.embedding import GoogleEmbedding
from leanworks.rag.query import QueryRewriter
import datetime
import logging
import asyncio
import json
from types import SimpleNamespace
# Set up logging
logger = logging.getLogger(__name__)

class Chat(FilterExtractor, MemoryManager, QueryRewriter, CrossEncoderReranker):
    """
    Chat class for retrieving context from Pinecone and generating responses using OpenAI.
    Provides synchronous functionality for RAG operations.
    """
    def __init__(
        self,
        pinecone_api_key: str,
        index_host: str,
        storage_client,
        embedding_model_api_key: str,
        model_client,
        user_id: str | None = None,
        session_id: str | None = None
    ):
        """
        Initialize Chat with Pinecone vector store and memory management.
        
        Args:
            pinecone_api_key: API key for Pinecone
            index_host: Host URL for the Pinecone index
            storage_client: Initialized CloudStorage client for memory persistence
            embedding_model_api_key: API key for the embedding model
            model_client: Initialized OpenAI client for LLM generation
            user_id: ID of the user
            session_id: ID of the current conversation session
        """
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(host=index_host)
        self.model_client = model_client
        # Initialize embedding model
        self.embedding_model = GoogleEmbedding(embedding_model_api_key)
        # Initialize memory manager if user_id and session_id are provided
        self.memory_enabled = user_id is not None and session_id is not None
        if self.memory_enabled:
            MemoryManager.__init__(self, model_client, storage_client, user_id, session_id)
            logger.info(f"Memory enabled for user_id: {user_id}, session_id: {session_id}")
        else:
            logger.info("Memory disabled - no user_id or session_id provided")
            
        # Initialize the FilterExtractor part
        FilterExtractor.__init__(self)
        
        # Initialize QueryRewriter
        QueryRewriter.__init__(self, model_client)

        # Initialize CrossEncoderReranker
        CrossEncoderReranker.__init__(self, model_client)
            
        logger.info("RAG system initialized successfully")

    def retrieve_nodes(self, query: str | List[str], top_k: int, filters: dict = None) -> Tuple[List[dict], List[str]]:
        """
        Retrieve relevant context from Pinecone for one or multiple queries.
        
        Args:
            query: The user query or list of queries. If a list is provided, time filters are 
                  extracted from the first query and applied to all retrievals.
            top_k: Number of context chunks to retrieve (default from settings)
            filters: Dictionary of filters to apply to the query (default None)
            
        Returns:
            Tuple containing (list of relevant context dicts with context and timestamp, list of unique source links)
        """
        # Handle single query case by converting to list
        queries = [query] if isinstance(query, str) else query
        if not queries:
            logger.warning("No queries provided to retrieve_nodes")
            return SimpleNamespace(matches=[])
            
        logger.info(f"Retrieving nodes for {len(queries)} queries with top_k={top_k}")
        
        # Results container
        all_matches = []
        
        # Query Pinecone for each query
        try:
            for q in queries:
                query_embedding = self.embedding_model.get_embedding(q)
                nodes = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                    filter=filters
                )
                
                # Add matches to combined results
                if hasattr(nodes, 'matches'):
                    all_matches.extend(nodes.matches)
                    logger.info(f"Retrieved {len(nodes.matches)} nodes for query: '{q}'")
                
            # Deduplicate matches by ID
            seen_ids = set()
            unique_matches = []
            for match in all_matches:
                if match.id not in seen_ids:
                    seen_ids.add(match.id)
                    unique_matches.append(match)
            
            # Create a new response object with deduplicated matches
            combined_response = SimpleNamespace(
                matches=unique_matches[:top_k]  # Limit to top_k after deduplication
            )
            
            logger.info(f"Combined and deduplicated to {len(combined_response.matches)} nodes")
            return combined_response
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            # Return an empty result structure with similar interface as Pinecone response
            empty_response = SimpleNamespace(matches=[])
            return empty_response
    
    def postprocess_nodes(
            self, nodes: List[dict], 
            query: str, 
            apply_filters: bool = False, 
            use_reranker: bool = False, 
            **kwargs
            ) -> Tuple[List[dict], List[str]]:
        """
        Process retrieved nodes from Pinecone and extract context information.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            apply_filters: Whether to apply user filters extracted from the query (default True)
            use_reranker: Whether to apply reranking (default None, which falls back to instance setting)
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        logger.debug("Initial documents", nodes.matches)
        # Filter results by relevance score
        filtered_results = [match for match in nodes.matches if match.score >= SIMILARITY_CUTOFF]
        logger.debug("Score filtered documents", filtered_results)

        # Extract user filters from query only if apply_filters is True
        if apply_filters:
            logger.info("Applying user filters")
            user_filters = self.extract_user_filters(query)
            if user_filters:
                # Filter results by user access
                user_filtered_results = []
                for match in filtered_results:
                    # Check if the node has user access information
                    if "users" in match.metadata:
                        # Get the users who can access this node
                        node_users = match.metadata["users"]
                        # If node_users is a string, check if all user_filters are in it
                        if isinstance(node_users, str):
                            if any(user in node_users for user in user_filters) or "everyone" in node_users:
                                user_filtered_results.append(match)
                    else:
                        # If no user access information, assume public access
                        user_filtered_results.append(match)
            else:
                user_filtered_results = filtered_results
        else:
            user_filtered_results = filtered_results
        # Apply reranking if enabled and needed (directly on the filtered results)
        rerank_top_k = kwargs.get("rerank_top_k")
        if use_reranker and user_filtered_results:
            if rerank_top_k is None:
                raise ValueError("rerank_top_k is required when use_reranker is True")
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(user_filtered_results):
                logger.info(f"Applying reranking to get top {rerank_top_k} documents...")
                try:
                    # Use the reranker to improve precision
                    logger.info("Initializing CrossEncoderReranker")
                    reranked_results = self.rerank(query, user_filtered_results, top_k=rerank_top_k)
                    logger.info(f"Successfully reranked results to {len(reranked_results)} documents")
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    reranked_results = user_filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                reranked_results = sorted(
                    user_filtered_results[:rerank_top_k],
                    key=lambda x: x.metadata.get("timestamp", 0)
                )
        else:
            reranked_results = sorted(
                user_filtered_results[:rerank_top_k],
                key=lambda x: x.metadata.get("timestamp", 0)
            )
        logger.debug("Reranked documents", reranked_results)

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in reranked_results:
            # Extract source information
            data_source = match.metadata["data_source"]
            links.add(match.metadata["link"])
            
            # Get timestamp if available
            timestamp = match.metadata.get("timestamp")
            
            # Extract context text using various fallback methods
            context_text = json.loads(match.metadata.get("_node_content", "")).get("text", "")
            
            # Skip duplicates
            if not context_text or context_text in seen_contexts:
                continue
                
            seen_contexts.add(context_text)
            
            # Add to contexts list
            contexts.append({
                "context": context_text,
                "timestamp": timestamp,
                "data_source": data_source
            })
        logger.debug(f"Filtered contexts: {contexts}")
        return contexts, list(links)
    
    def _retrieve_memory(self):
        """Helper method to retrieve memory in parallel"""
        recent_memories = self.get_recent_memories()
        memory_context = self.format_memory_context(recent_memories)
        return memory_context

    def _should_apply_reranking(self, filtered_results: List) -> bool:
        """Determine if reranking is needed based on result quality"""
        if len(filtered_results) <= 3:
            return False  # Skip reranking for very small result sets
        
        # Skip if similarity scores are already very high
        high_confidence = all(match.score > 0.8 for match in filtered_results[:3])
        return not high_confidence
    
    def get_response(
            self, query: str, 
            model: str = GENERATION_MODEL,
            top_k: int = RETRIEVE_TOP_K,
            include_memory: bool = INCLUDE_MEMORY,
            use_reranker: bool = USE_RERANKER,
            apply_filters: bool = APPLY_FILTERS,
            query_rewrites: bool = QUERY_REWRITES,
            cited_context: dict = None,
            **kwargs
            ) -> Dict[str, any]:
        """
        Generate a response using RAG approach.
        
        Args:
            query: The user query
            model: The model to use for generation
            include_memory: Whether to include recent conversation history
            cited_context: Specific context cited by the user, a dictionary
            
        Returns:
            Dictionary with 'content' (the answer) and 'data_sources' (list of unique links)
        """
        logger.info(f"Generating response for query: '{query}' using model: {model}")

        # Get today's date in ISO UTC format
        today_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Add cited context to the query if provided
        if cited_context:
            full_query = f"(Current date: {today_date}, User cited context: {cited_context}) {query}"
        else:
            full_query = f"(Current date: {today_date}) {query}"

        # Retrieve context
        context, data_sources = [], []
        rerank_top_k = kwargs.get("rerank_top_k", RERANK_TOP_K)
        try:
            if query_rewrites:
                query_rewrites = self.rewrite_query(query)
                all_queries = [full_query] + query_rewrites
                print(all_queries)
            else:
                all_queries = [full_query]
            if apply_filters:
                filters = self.extract_time_filters(query, self.model_client)
            else:
                filters = None
            nodes = self.retrieve_nodes(all_queries, top_k=top_k, filters=filters)
            context, data_sources = self.postprocess_nodes(
                nodes, 
                full_query, 
                apply_filters=apply_filters, 
                use_reranker=use_reranker, 
                rerank_top_k=rerank_top_k
                )
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}")
        
        # Get memory if needed
        memory_context = []
        if include_memory and self.memory_enabled:
            memory_context = self._retrieve_memory()
        
        # Format context with recency information
        formatted_context = ""
        
        # Add memory context first if available
        if memory_context:
            formatted_context += "Recent conversations:\n"
            formatted_context += "\n".join(memory_context)
            formatted_context += "\n\nCurrent relevant documents:\n"
        
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
        
        prompt = f"Context information (ordered by relevance, most relevant first):\n{formatted_context}\n\nUser query: {full_query}\n\nResponse:"
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {GENERATION_MODEL_SYSTEM_PROMPT}")
        logger.info(f"User prompt: {prompt}")

        try:
            response = self.model_client.chat.completions.create(
                model=model,
                max_tokens=256,  # Allow for longer responses
                messages=[
                    {"role": "system", "content": GENERATION_MODEL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content
            
            # Log a preview of the model's response
            logger.info(f"Model {model} response: {answer}")
        except Exception as e:
            logger.error(f"Error generating model {model} response: {str(e)}")
            answer = "I apologize, but I'm currently experiencing technical difficulties and couldn't generate a response. Please try again later."
        
        # Store in memory if enabled
        if self.memory_enabled:
            self.add_memory(query, answer)
        
        # Return dictionary with content and data_sources
        return {
            "content": answer,
            "data_sources": data_sources
        }

class AsyncChat(Chat):
    """
    Asynchronous version of the Chat class that provides non-blocking operations
    for better performance in async environments.
    """
    def __init__(
        self,
        pinecone_api_key: str,
        index_host: str,
        storage_client,
        embedding_model_api_key: str,
        model_client,
        user_id: str | None = None,
        session_id: str | None = None
    ):
        """Initialize AsyncChat with the same parameters as Chat."""
        super().__init__(
            pinecone_api_key=pinecone_api_key,
            index_host=index_host,
            storage_client=storage_client,
            embedding_model_api_key=embedding_model_api_key,
            model_client=model_client,
            user_id=user_id,
            session_id=session_id
        )
        logger.info("AsyncChat initialized successfully")
    
    async def async_postprocess_nodes(
            self, nodes: List[dict], 
            query: str, 
            apply_filters: bool = False, 
            use_reranker: bool = False, 
            **kwargs
        ) -> Tuple[List[dict], List[str]]:
        """
        Asynchronous version of postprocess_nodes that uses async reranking for better performance.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            apply_filters: Whether to apply user filters extracted from the query
            use_reranker: Whether to apply reranking
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Async postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        logger.debug("Initial documents", nodes.matches)
        # Filter results by relevance score
        filtered_results = [match for match in nodes.matches if match.score >= SIMILARITY_CUTOFF]
        logger.debug("Score filtered documents", filtered_results)

        # Extract user filters from query only if apply_filters is True
        if apply_filters:
            user_filters = self.extract_user_filters(query)
            if user_filters:
                # Filter results by user access
                user_filtered_results = []
                for match in filtered_results:
                    # Check if the node has user access information
                    if "users" in match.metadata:
                        # Get the users who can access this node
                        node_users = match.metadata["users"]
                        # If node_users is a string, check if all user_filters are in it
                        if isinstance(node_users, str):
                            if any(user in node_users for user in user_filters) or "everyone" in node_users:
                                user_filtered_results.append(match)
                    else:
                        # If no user access information, assume public access
                        user_filtered_results.append(match)
            else:
                user_filtered_results = filtered_results
        else:
            user_filtered_results = filtered_results
            
        # Apply reranking if enabled and needed (directly on the filtered results)
        rerank_top_k = kwargs.get("rerank_top_k")
        if use_reranker and user_filtered_results:
            if rerank_top_k is None:
                raise ValueError("rerank_top_k is required when use_reranker is True")
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(user_filtered_results):
                logger.info(f"Applying async reranking to get top {rerank_top_k} documents...")
                try:
                    # Use async reranker to improve precision without blocking
                    reranked_results = await self.async_rerank(
                        query, 
                        user_filtered_results,
                        top_k=rerank_top_k
                    )
                    logger.info(f"Successfully reranked results to {len(reranked_results)} documents")
                except Exception as e:
                    logger.error(f"Error during async reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    reranked_results = user_filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                reranked_results = sorted(
                    user_filtered_results[:rerank_top_k],
                    key=lambda x: x.metadata.get("timestamp", 0)
                )
        else:
            reranked_results = sorted(
                user_filtered_results[:rerank_top_k],
                key=lambda x: x.metadata.get("timestamp", 0)
            )
        logger.debug("Reranked documents", reranked_results)

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in reranked_results:
            # Extract source information
            data_source = match.metadata["data_source"]
            links.add(match.metadata["link"])
            
            # Get timestamp if available
            timestamp = match.metadata.get("timestamp")
            
            # Extract context text using the same method as in Chat class
            context_text = json.loads(match.metadata.get("_node_content", "")).get("text", "")
            
            # Skip duplicates
            if not context_text or context_text in seen_contexts:
                continue
                
            seen_contexts.add(context_text)
            
            # Add to contexts list
            contexts.append({
                "context": context_text,
                "timestamp": timestamp,
                "data_source": data_source
            })
        logger.debug(f"Filtered contexts: {contexts}")
        return contexts, list(links)
        
    async def async_get_response(
            self, query: str, 
            model: str = GENERATION_MODEL,
            top_k: int = RETRIEVE_TOP_K,
            include_memory: bool = INCLUDE_MEMORY,
            use_reranker: bool = USE_RERANKER,
            apply_filters: bool = APPLY_FILTERS,
            query_rewrites: bool = QUERY_REWRITES,
            cited_context: dict = None,
            **kwargs
            ) -> Dict[str, any]:
        """
        Asynchronously generate a response using RAG approach with improved performance.
        
        Args:
            query: The user query
            model: The model to use for generation
            top_k: Number of context chunks to retrieve
            include_memory: Whether to include recent conversation history
            use_reranker: Whether to apply reranking
            apply_filters: Whether to apply time filters extracted from the query
            query_rewrites: Whether to use query rewriting for better recall
            cited_context: Specific context cited by the user, a dictionary
            
        Returns:
            Dictionary with 'content' (the answer) and 'data_sources' (list of unique links)
        """
        logger.info(f"Asynchronously generating response for query: '{query}' using model: {model}")

        # Get today's date in ISO UTC format
        today_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Add cited context to the query if provided
        if cited_context:
            full_query = f"(Current date: {today_date}, User cited context: {cited_context}) {query}"
        else:
            full_query = f"(Current date: {today_date}) {query}"

        # Retrieve context asynchronously
        rerank_top_k = kwargs.get("rerank_top_k", RERANK_TOP_K)
        logger.info(f"Starting async context retrieval with top_k={top_k}, rerank_top_k={rerank_top_k}, query_rewrites={query_rewrites}")
        
        # Create tasks for parallel execution
        tasks = []
        
        # Task 1: Generate query rewrites if needed (runs in parallel)
        rewrites_task = None
        if query_rewrites:
            rewrites_task = asyncio.create_task(self.async_rewrite_query(query))
            tasks.append(rewrites_task)
        
        # Task 2: Extract time filters if apply_filters is True (runs in parallel)
        filters_task = None
        if apply_filters:
            filters_task = asyncio.create_task(self.async_extract_time_filters(query))
            tasks.append(filters_task)
            
        # Task 3: Memory retrieval (if needed) - runs in parallel
        memory_task = None
        if include_memory and self.memory_enabled:
            loop = asyncio.get_event_loop()
            # This already returns a Future, no need to wrap in create_task
            memory_task = loop.run_in_executor(None, self._retrieve_memory)
            tasks.append(memory_task)
            
        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks)
            
        # Prepare all queries for retrieval
        all_queries = [full_query]
        if query_rewrites and rewrites_task:
            try:
                rewrites = rewrites_task.result()
                all_queries.extend(rewrites)
                logger.info(f"Generated {len(rewrites)} query rewrites: {rewrites}")
            except Exception as e:
                logger.error(f"Error getting query rewrites: {str(e)}")
                
        # Get time filters for retrieval
        filters = None
        if apply_filters and filters_task:
            try:
                filters = filters_task.result()
                logger.info(f"Applied time filters: {filters}")
            except Exception as e:
                logger.error(f"Error getting time filters: {str(e)}")
                
        # Log to make it clear what queries are being used
        logger.info(f"Using queries for retrieval: {all_queries} with filters: {filters}")
        
        # Retrieve nodes (blocking operation, run in executor)
        loop = asyncio.get_event_loop()
        try:
            nodes = await loop.run_in_executor(
                None, 
                lambda: self.retrieve_nodes(all_queries, top_k, filters)
            )
            
            # Use async postprocessing with non-blocking reranking
            context, data_sources = await self.async_postprocess_nodes(
                nodes, 
                full_query, 
                apply_filters=apply_filters, 
                use_reranker=use_reranker, 
                rerank_top_k=rerank_top_k
            )
            logger.info(f"Retrieved and processed {len(context)} contexts")
        except Exception as e:
            logger.error(f"Error in async context retrieval: {str(e)}")
            context, data_sources = [], []
        
        # Get memory context if the task was started
        memory_context = []
        if memory_task:
            try:
                # Just await the Future directly
                memory_context = await memory_task
                logger.info(f"Retrieved memory context with {len(memory_context)} entries")
            except Exception as e:
                logger.error(f"Error retrieving memory: {str(e)}")
            
        # Format context with recency information
        formatted_context = ""
        
        # Add memory context first if available
        if memory_context:
            formatted_context += "Recent conversations:\n"
            formatted_context += "\n".join(memory_context)
            formatted_context += "\n\nCurrent relevant documents:\n"
        
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
        
        prompt = f"Context information (ordered by relevance, most relevant first):\n{formatted_context}\n\nUser query: {full_query}\n\nResponse:"
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {GENERATION_MODEL_SYSTEM_PROMPT}")
        logger.info(f"User prompt: {prompt}")

        try:
            # Call the model asynchronously
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": GENERATION_MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                        ],
                    temperature=0.0
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=90)
            answer = response.choices[0].message.content
            
            # Log a preview of the model's response
            logger.info(f"Model {model} response: {answer}")
        except asyncio.TimeoutError:
            logger.error(f"Model {model} response generation timed out after 90 seconds")
            answer = "I apologize, but I'm currently experiencing technical difficulties and couldn't generate a response. Please try again later."
        except Exception as e:
            logger.error(f"Error generating model {model} response: {str(e)}")
            answer = "I apologize, but I'm currently experiencing technical difficulties and couldn't generate a response. Please try again later."
        
        # Store in memory if enabled
        if self.memory_enabled:
            # Run memory storage in background
            loop.run_in_executor(None, self.add_memory, query, answer)
        
        # Return dictionary with content and data_sources
        return {
            "content": answer,
            "data_sources": data_sources
        }
        
    async def async_rerank(self, query: str, documents: List[Any], **kwargs) -> List[Any]:
        """Asynchronous reranking to improve response time"""
        loop = asyncio.get_event_loop()
        # Pass through all parameters to ensure identical results with sync version
        return await loop.run_in_executor(
            None, 
            lambda: self.rerank(query, documents, **kwargs)
        )