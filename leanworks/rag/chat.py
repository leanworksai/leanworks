from pinecone import Pinecone
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import json
from leanworks.rag.filters import FilterExtractor
from leanworks.rag.memory import MemoryManager
from leanworks.rag.query import QueryParser
from leanworks.rag.reranker import CrossEncoderReranker, HybridReranker
import datetime
import logging
from functools import lru_cache
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Set up logging
logger = logging.getLogger(__name__)

class Chat(FilterExtractor, MemoryManager, QueryParser):
    """
    Chat class for retrieving context from Pinecone and generating responses using OpenAI.
    """
    def __init__(
        self,
        pinecone_api_key: str,
        index_host: str,
        storage_client,
        embedding_model_client,
        model_client,
        user_id: str | None = None,
        session_id: str | None = None,
        use_reranker: bool = True,
        reranker_type: str = "cross_encoder"
    ):
        """
        Initialize Chat with Pinecone vector store and memory management.
        
        Args:
            pinecone_api_key: API key for Pinecone
            index_host: Host URL for the Pinecone index
            storage_client: Initialized CloudStorage client for memory persistence
            embedding_model_client: Embedding model client
            model_client: Initialized OpenAI client for LLM generation
            user_id: ID of the user
            session_id: ID of the current conversation session
            use_reranker: Whether to use the reranker for improved precision
            reranker_type: Type of reranker to use ("cross_encoder" or "hybrid")
        """
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(host=index_host)
        self.model_client = model_client
        self.embedding_model_client = embedding_model_client
        # Initialize memory manager if user_id and session_id are provided
        self.memory_enabled = user_id is not None and session_id is not None
        if self.memory_enabled:
            MemoryManager.__init__(self, model_client, storage_client, user_id, session_id)
            logger.info(f"Memory enabled for user_id: {user_id}, session_id: {session_id}")
        else:
            logger.info("Memory disabled - no user_id or session_id provided")
            
        # Initialize the FilterExtractor part
        FilterExtractor.__init__(self)
        
        # Initialize the QueryParser part
        QueryParser.__init__(self, model_client)
        
        # Initialize the Reranker if enabled
        self.use_reranker = use_reranker
        if self.use_reranker:
            if reranker_type == "hybrid":
                self.reranker = HybridReranker(model_client)
                logger.info("Hybrid reranker initialized")
            else:
                self.reranker = CrossEncoderReranker(model_client)
                logger.info("Cross-encoder reranker initialized")
            
        logger.info("RAG system initialized successfully")

    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text using OpenAI with caching."""
        logger.debug(f"Generating embedding for text of length: {len(text)}")
        try:
            # Add timeout handling for embedding generation
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.embedding_model_client.models.embed_content,
                    model="text-embedding-004",
                    contents=text
                )
                # Wait for result with 30 second timeout
                result = future.result(timeout=30)
                return np.array(result.embeddings[0].values)
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero embedding as fallback
            return np.zeros(1536)  # Standard embedding dimension

    def retrieve_nodes(self, query: str, top_k: int = 20) -> Tuple[List[dict], List[str]]:
        """
        Retrieve relevant context from Pinecone.
        
        Args:
            query: The user query
            top_k: Number of context chunks to retrieve (default 20 for reranking)
            
        Returns:
            Tuple containing (list of relevant context dicts with context and timestamp, list of unique source links)
        """
        logger.info(f"Retrieving nodes for query: '{query}' with top_k={top_k}")
        query_embedding = self._get_embedding(query)
        
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
        nodes = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        logger.info(f"Retrieved {len(nodes.matches) if hasattr(nodes, 'matches') else 0} nodes from Pinecone")
        return nodes
    
    def postprocess_nodes(self, nodes: List[dict], query: str, rerank_top_k: int = 5) -> Tuple[List[dict], List[str]]:
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
        # Filter results by relevance score
        filtered_results = [match for match in nodes.matches if match.score >= 0.4]
        
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
        logger.info(f"Filtered contexts: {contexts}")
        return contexts, list(links)
    
    async def async_postprocess_nodes(self, nodes: List[dict], query: str, rerank_top_k: int = 5) -> Tuple[List[dict], List[str]]:
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
        filtered_results = [match for match in nodes.matches if match.score >= 0.4]
        
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
        logger.info(f"Filtered contexts: {contexts}")
        return contexts, list(links)
    
    async def async_get_response(
            self, query: str, 
            model: str = "claude-3-5-haiku-20241022", 
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
            recency_indicator = f"[RECENT DOCUMENT - Date: {timestamp_str}{source_str}]: "
            formatted_context += recency_indicator + ctx["context"] + "\n\n"
        
        prompt = f"Context information (ordered by recency, most recent first):\n{formatted_context}\n\nUser query: {full_query}\n\nResponse:"
        system_prompt = f'''You are a helpful technical project manager that answers your teammates' questions based on the provided context. 
        Pay special attention to more recent information as it's more likely to be relevant and up-to-date. When recent conversations are provided, 
        use them to maintain consistency with previous responses. When the user has cited specific context, prioritize that information in your response.
        
        If the user's question is vague or lacks sufficient context, you may ask at most one clarifying question to better understand their needs.
        '''
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {prompt}")

        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    max_tokens=4096,
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
            logger.info(f"Model response: {answer}")
        except asyncio.TimeoutError:
            logger.error("Model response generation timed out after 90 seconds")
            answer = "I apologize, but I'm currently experiencing high load and couldn't generate a response in time. Please try again with a simpler query or try again later."
        except Exception as e:
            logger.error(f"Error generating model response: {str(e)}")
            answer = f"I encountered an error processing your request: {str(e)[:100]}... Please try again."
        
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
    
    def _retrieve_and_process_context(self, query_with_date, top_k=20, rerank_top_k=5):
        """Helper method to retrieve and process context in parallel"""
        nodes = self.retrieve_nodes(query_with_date, top_k=top_k)
        context, data_sources = self.postprocess_nodes(nodes, query_with_date, rerank_top_k=rerank_top_k)
        return context, data_sources
    
    async def _async_retrieve_and_process_context(self, query_with_date, top_k=20, rerank_top_k=5):
        """Asynchronous helper method to retrieve and process context"""
        # Retrieve nodes (this is I/O bound, so we'll run it in the executor)
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(None, self.retrieve_nodes, query_with_date, top_k)
        
        # Use async postprocessing with non-blocking reranking
        context, data_sources = await self.async_postprocess_nodes(nodes, query_with_date, rerank_top_k=rerank_top_k)
        return context, data_sources
    
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

    async def async_rerank(self, query: str, documents: List[Any]) -> List[Any]:
        """Asynchronous reranking to improve response time"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.reranker.rerank, query, documents)

    def get_response(
            self, query: str, 
            model: str = "claude-3-5-haiku-20241022", 
            include_memory: bool = True, 
            cited_context: dict = None
            ) -> Dict[str, any]:
        """
        Generate a response using RAG approach with OpenAI.
        This is a synchronous wrapper around the async version.
        
        Args:
            query: The user query
            model: The model to use for generation
            include_memory: Whether to include recent conversation history
            cited_context: Specific context cited by the user, a dictionary
            
        Returns:
            Dictionary with 'content' (the answer) and 'data_sources' (list of unique links)
        """
        logger.info(f"Generating response for query: '{query}' using model: {model}")
        
        # Use asyncio.run to run the async version from synchronous code
        try:
            import nest_asyncio
            # Apply nest_asyncio to allow running asyncio inside Jupyter/IPython
            nest_asyncio.apply()
        except ImportError:
            logger.info("nest_asyncio not available, skipping")
        
        # Run the async version in the default event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop if the current one is running
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.async_get_response(query, model, include_memory, cited_context)
            )
        except Exception as e:
            logger.error(f"Error executing async response: {str(e)}")
            # Fallback to the original implementation in case of issues with async
            return self._sync_get_response(query, model, include_memory, cited_context)
    
    def _sync_get_response(
            self, query: str, 
            model: str = "claude-3-5-haiku-20241022", 
            include_memory: bool = True, 
            cited_context: dict = None
            ) -> Dict[str, any]:
        """
        Original synchronous implementation of get_response.
        Used as a fallback if async version fails.
        """
        logger.info(f"Using synchronous fallback for query: '{query}'")

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
            recency_indicator = f"[RECENT DOCUMENT - Date: {timestamp_str}{source_str}]: "
            formatted_context += recency_indicator + ctx["context"] + "\n\n"
        
        prompt = f"Context information (ordered by recency, most recent first):\n{formatted_context}\n\nUser query: {full_query}\n\nResponse:"
        system_prompt = f'''You are a helpful technical project manager that answers your teammates' questions based on the provided context. 
        Pay special attention to more recent information as it's more likely to be relevant and up-to-date. When recent conversations are provided, 
        use them to maintain consistency with previous responses. When the user has cited specific context, prioritize that information in your response.
        
        If the user's question is vague or lacks sufficient context, you may ask at most one clarifying question to better understand their needs.
        '''
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {prompt}")

        try:
            # Add timeout handling for model response generation
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.model_client.chat.completions.create,
                    model=model,
                    max_tokens=4096,  # Allow for longer responses
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                # Wait for result with 90 second timeout
                response = future.result(timeout=90)
                answer = response.choices[0].message.content
            
            # Log a preview of the model's response
            logger.info(f"Model response: {answer}")
        except concurrent.futures.TimeoutError:
            logger.error("Model response generation timed out after 90 seconds")
            answer = "I apologize, but I'm currently experiencing high load and couldn't generate a response in time. Please try again with a simpler query or try again later."
        except Exception as e:
            logger.error(f"Error generating model response: {str(e)}")
            answer = f"I encountered an error processing your request: {str(e)[:100]}... Please try again."
        
        # Store in memory if enabled
        if self.memory_enabled:
            self.add_memory(query, answer)
        
        # Return dictionary with content and data_sources
        return {
            "content": answer,
            "data_sources": data_sources
        }