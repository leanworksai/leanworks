from pinecone import Pinecone
from typing import List, Dict, Tuple, Any
import numpy as np
from leanworks.rag.filters import FilterExtractor
from leanworks.rag.memory import MemoryManager
from leanworks.rag.reranker import CrossEncoderReranker
from leanworks.rag.setting import GENERATION_MODEL, RETRIEVE_TOP_K, RERANK_TOP_K, SIMILARITY_CUTOFF
from leanworks.rag.embedding import GoogleEmbedding
import datetime
import logging
from functools import lru_cache
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
from google.genai import types
import json

# Set up logging
logger = logging.getLogger(__name__)

class Chat(FilterExtractor, MemoryManager):
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
        session_id: str | None = None,
        use_reranker: bool = True,
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
            use_reranker: Whether to use the reranker for improved precision
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
        
        # Initialize the Reranker if enabled
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(model_client)
            logger.info("Cross-encoder reranker initialized")
            
        logger.info("RAG system initialized successfully")

    def retrieve_nodes(self, query: str, top_k: int = RETRIEVE_TOP_K) -> Tuple[List[dict], List[str]]:
        """
        Retrieve relevant context from Pinecone.
        
        Args:
            query: The user query
            top_k: Number of context chunks to retrieve (default from settings)
            
        Returns:
            Tuple containing (list of relevant context dicts with context and timestamp, list of unique source links)
        """
        logger.info(f"Retrieving nodes for query: '{query}' with top_k={top_k}")
        query_embedding = self.embedding_model.get_embedding(query)
        
        # Extract filters from query
        time_filters = self.extract_time_filters(query, self.model_client)

        
        # Prepare filter dict for Pinecone
        filter_dict = {}
        if time_filters["start_timestamp"]:
            filter_dict["timestamp"] = {"$gte": time_filters["start_timestamp"]}
            logger.debug(f"Applied start timestamp filter: {time_filters['start_timestamp']}")
        if time_filters["end_timestamp"]:
            if "timestamp" in filter_dict:
                filter_dict["timestamp"]["$lte"] = time_filters["end_timestamp"]
            else:
                filter_dict["timestamp"] = {"$lte": time_filters["end_timestamp"]}
            logger.debug(f"Applied end timestamp filter: {time_filters['end_timestamp']}")
        
        # Query Pinecone with time filter if available
        try:
            nodes = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            logger.info(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes from Pinecone")
            return nodes
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            # Return an empty result structure with similar interface as Pinecone response
            from types import SimpleNamespace
            empty_response = SimpleNamespace(matches=[])
            return empty_response
    
    def postprocess_nodes(self, nodes: List[dict], query: str, rerank_top_k: int = RERANK_TOP_K) -> Tuple[List[dict], List[str]]:
        """
        Process retrieved nodes from Pinecone and extract context information.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            rerank_top_k: Number of top documents to keep after reranking
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        logger.debug("Initial documents", nodes.matches)
        # Filter results by relevance score
        filtered_results = [match for match in nodes.matches if match.score >= SIMILARITY_CUTOFF]
        logger.debug("Score filtered documents", filtered_results)
        # Apply reranking if enabled and needed (directly on the filtered results)
        if self.use_reranker and filtered_results:
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(filtered_results):
                logger.info(f"Applying reranking to get top {rerank_top_k} documents...")
                try:
                    # Use the reranker to improve precision
                    reranked_results = self.reranker.rerank(query, filtered_results, top_k=rerank_top_k)
                    filtered_results = reranked_results
                    logger.info(f"Successfully reranked results to {len(filtered_results)} documents")
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    filtered_results = filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                filtered_results = filtered_results[:rerank_top_k]
        logger.debug("Reranked documents", filtered_results)

        # Extract user filters from query
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
        
        # Sort results by timestamp (most recent first)
        # If reranked, we'll keep the reranking order for better precision
        if self.use_reranker:
            matches = user_filtered_results
        else:
            matches = sorted(
                user_filtered_results,
                key=lambda x: x.metadata.get("timestamp", 0)
            )

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in matches:
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
    
    def _retrieve_and_process_context(self, query_with_date, top_k=RETRIEVE_TOP_K, rerank_top_k=RERANK_TOP_K):
        """Helper method to retrieve and process context in parallel"""
        try:
            nodes = self.retrieve_nodes(query_with_date, top_k=top_k)
            context, data_sources = self.postprocess_nodes(nodes, query_with_date, rerank_top_k=rerank_top_k)
            return context, data_sources
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}")
            # Return empty results in case of any error
            return [], []
    
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
            include_memory: bool = True, 
            cited_context: dict = None
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

        # Fallback models if the specified model is unavailable
        fallback_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "gpt-4o", "claude-3-haiku-20240307"]

        # Get today's date in ISO UTC format
        today_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Add cited context to the query if provided
        if cited_context:
            full_query = f"(Current date: {today_date}, User cited context: {cited_context}) {query}"
        else:
            full_query = f"(Current date: {today_date}) {query}"

        # Use ThreadPoolExecutor to parallelize memory and context retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start context retrieval task
            context_future = executor.submit(self._retrieve_and_process_context, full_query)
            
            # Start memory retrieval task if needed
            memory_future = None
            if include_memory and self.memory_enabled:
                memory_future = executor.submit(self._retrieve_memory)
            
            # Wait for both tasks to complete with timeouts
            try:
                # Set timeout for context retrieval (60 seconds)
                context, data_sources = context_future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                logger.error("Context retrieval timed out after 60 seconds, using empty context")
                context, data_sources = [], []
            
            # Get memory context if requested (with timeout)
            memory_context = []
            if memory_future:
                try:
                    memory_context = memory_future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    logger.error("Memory retrieval timed out after 30 seconds, using empty memory context")
        
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
        system_prompt = f'''You are a helpful technical project manager that answers your teammates' questions based on the provided context. 
        When recent conversations are provided, use them to maintain consistency with previous responses. 
        User cited context serves as reference for the user query if it is provided.
        '''
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {prompt}")

        # Try primary model first, then fallback models if there's an error
        models_to_try = [model] + fallback_models
        answer = None

        for current_model in models_to_try:
            try:
                # Add timeout handling for model response generation
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.model_client.chat.completions.create,
                        model=current_model,
                        max_tokens=1024,  # Allow for longer responses
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    # Wait for result with 90 second timeout
                    response = future.result(timeout=90)
                    answer = response.choices[0].message.content
                
                # Log a preview of the model's response
                logger.info(f"Model {current_model} response: {answer}")
                # Break the loop if response was successful
                break
            except concurrent.futures.TimeoutError:
                logger.error(f"Model {current_model} response generation timed out after 90 seconds")
                continue
            except Exception as e:
                logger.error(f"Error generating model {current_model} response: {str(e)}")
                continue
        
        # If all models failed, return a generic error message
        if answer is None:
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
        session_id: str | None = None,
        use_reranker: bool = True,
    ):
        """Initialize AsyncChat with the same parameters as Chat."""
        super().__init__(
            pinecone_api_key=pinecone_api_key,
            index_host=index_host,
            storage_client=storage_client,
            embedding_model_api_key=embedding_model_api_key,
            model_client=model_client,
            user_id=user_id,
            session_id=session_id,
            use_reranker=use_reranker
        )
        logger.info("AsyncChat initialized successfully")
    
    async def async_postprocess_nodes(self, nodes: List[dict], query: str, rerank_top_k: int = RERANK_TOP_K) -> Tuple[List[dict], List[str]]:
        """
        Asynchronous version of postprocess_nodes that uses async reranking for better performance.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            rerank_top_k: Number of top documents to keep after reranking
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Async postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        # Filter results by relevance score
        filtered_results = [match for match in nodes.matches if match.score >= SIMILARITY_CUTOFF]
        
        # Apply reranking if enabled and needed (directly on the filtered results)
        if self.use_reranker and filtered_results:
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(filtered_results):
                logger.info(f"Applying async reranking to get top {rerank_top_k} documents...")
                try:
                    # Use async reranker to improve precision without blocking
                    reranked_results = await self.async_rerank(query, filtered_results)
                    filtered_results = reranked_results
                    logger.info(f"Successfully reranked results to {len(filtered_results)} documents")
                except Exception as e:
                    logger.error(f"Error during async reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    filtered_results = filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                filtered_results = filtered_results[:rerank_top_k]
        
        # Extract user filters from query
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
        
        # Sort results by timestamp (most recent first)
        # If reranked, we'll keep the reranking order for better precision
        if self.use_reranker:
            matches = user_filtered_results
        else:
            matches = sorted(
                user_filtered_results,
                key=lambda x: x.metadata.get("timestamp", 0)
            )

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in matches:
            # Extract source information
            data_source = match.metadata["data_source"]
            links.add(match.metadata["link"])
            
            # Get timestamp if available
            timestamp = match.metadata.get("timestamp")
            
            # Extract context text using various fallback methods
            context_text = ""
            if "text" in match.metadata:
                context_text = match.metadata["text"]
            elif "_node_content" in match.metadata:
                try:
                    node_content = json.loads(match.metadata.get("_node_content", ""))
                    context_text = node_content.get("text", "") or match.metadata.get("context", "")
                except:
                    logger.warning("Failed to parse _node_content JSON, falling back to context field")
                    context_text = match.metadata.get("context", "")
            
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
        
    async def _async_retrieve_and_process_context(self, query_with_date, top_k=RETRIEVE_TOP_K, rerank_top_k=RERANK_TOP_K):
        """Asynchronous helper method to retrieve and process context"""
        # Retrieve nodes (this is I/O bound, so we'll run it in the executor)
        loop = asyncio.get_event_loop()
        try:
            nodes = await loop.run_in_executor(None, self.retrieve_nodes, query_with_date, top_k)
            
            # Use async postprocessing with non-blocking reranking
            context, data_sources = await self.async_postprocess_nodes(nodes, query_with_date, rerank_top_k=rerank_top_k)
            return context, data_sources
        except Exception as e:
            logger.error(f"Error in async context retrieval: {str(e)}")
            # Return empty results in case of any error
            return [], []

    async def async_rerank(self, query: str, documents: List[Any]) -> List[Any]:
        """Asynchronous reranking to improve response time"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.reranker.rerank, query, documents)
    
    async def async_get_response(
            self, query: str, 
            model: str = GENERATION_MODEL, 
            include_memory: bool = True, 
            cited_context: dict = None
            ) -> Dict[str, any]:
        """
        Asynchronously generate a response using RAG approach with improved performance.
        
        Args:
            query: The user query
            model: The model to use for generation
            include_memory: Whether to include recent conversation history
            cited_context: Specific context cited by the user, a dictionary
            
        Returns:
            Dictionary with 'content' (the answer) and 'data_sources' (list of unique links)
        """
        logger.info(f"Asynchronously generating response for query: '{query}' using model: {model}")

        # Fallback models if the specified model is unavailable
        fallback_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "gpt-4o", "claude-3-haiku-20240307"]

        # Get today's date in ISO UTC format
        today_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Add cited context to the query if provided
        if cited_context:
            full_query = f"(Current date: {today_date}, User cited context: {cited_context}) {query}"
        else:
            full_query = f"(Current date: {today_date}) {query}"

        # Create tasks for async processing
        context_task = asyncio.create_task(self._async_retrieve_and_process_context(full_query))
        
        # Get memory in parallel if needed
        memory_context = []
        if include_memory and self.memory_enabled:
            loop = asyncio.get_event_loop()
            try:
                memory_context = await asyncio.wait_for(
                    loop.run_in_executor(None, self._retrieve_memory),
                    timeout=30
                )
            except asyncio.TimeoutError:
                logger.error("Memory retrieval timed out after 30 seconds, using empty memory context")
        
        # Wait for context with timeout
        try:
            context, data_sources = await asyncio.wait_for(context_task, timeout=60)
        except asyncio.TimeoutError:
            logger.error("Context retrieval timed out after 60 seconds, using empty context")
            context, data_sources = [], []
            
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
        system_prompt = f'''You are a helpful technical project manager that answers your teammates' questions based on the provided context. 
        When recent conversations are provided, use them to maintain consistency with previous responses. 
        User cited context serves as reference for the user query if it is provided.
        '''
        
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {prompt}")

        # Try primary model first, then fallback models if there's an error
        models_to_try = [model] + fallback_models
        answer = None

        for current_model in models_to_try:
            try:
                # Call the model asynchronously
                loop = asyncio.get_event_loop()
                response_future = loop.run_in_executor(
                    None,
                    lambda: self.model_client.chat.completions.create(
                        model=current_model,
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                            ]
                    )
                )
                
                # Wait for result with timeout
                response = await asyncio.wait_for(response_future, timeout=90)
                answer = response.choices[0].message.content
                
                # Log a preview of the model's response
                logger.info(f"Model {current_model} response: {answer}")
                # Break the loop if response was successful
                break
            except asyncio.TimeoutError:
                logger.error(f"Model {current_model} response generation timed out after 90 seconds")
                continue
            except Exception as e:
                logger.error(f"Error generating model {current_model} response: {str(e)}")
                continue
        
        # If all models failed, return a generic error message
        if answer is None:
            answer = "I apologize, but I'm currently experiencing technical difficulties and couldn't generate a response. Please try again later."
        
        # Store in memory if enabled
        if self.memory_enabled:
            # Run memory storage in background
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.add_memory, query, answer)
        
        # Return dictionary with content and data_sources
        return {
            "content": answer,
            "data_sources": data_sources
        }