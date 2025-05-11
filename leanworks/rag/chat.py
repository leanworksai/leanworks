from pinecone import Pinecone
from typing import List, Dict, Tuple, Any
from leanworks.rag.filters import FilterExtractor
from leanworks.rag.memory import MemoryManager
from leanworks.rag.reranker import CrossEncoderReranker
from leanworks.rag.setting import *
from leanworks.rag.embedding import GoogleEmbedding
import datetime
import logging
import asyncio
import json
from types import SimpleNamespace
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
            
        logger.info("RAG system initialized successfully")

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
        
        system_prompt = '''You are **SearchQueryRewriter‑MQR**, a large‑language‑model agent that creates
        *diverse, high‑recall* rewrites of a user's information‑seeking query.

        ## Instructions
        1. Read the **Original Query**.
        2. Produce **{{N}}** DISTINCT rewrites (do **NOT** answer the question).
        3. Follow these rewriting strategies *at least once each*  
        a. **Equality** – preserve all meaning; just de‑chatify the wording.  
        b. **Expansion** – add missing context a domain expert would expect  
            (e.g., synonyms, acronyms, date ranges, entity types).  
        c. **Reduction** – strip to the absolute core keywords.  
        d. *(Optional if N > 2)* Other creative perspectives that could surface
            different documents (e.g., broader background, comparison terms).
        4. **Constraints**  
        • ≤ 20 tokens per rewrite.  
        • Remove pronouns/ellipsis; name all entities explicitly.  
        • Avoid stop‑words unless essential (e.g., "of", "in").  
        • No duplicate semantic meaning across rewrites.
        5. Return a **valid JSON** object ONLY without any other text:

        ```json
        { "rewrites": [" ... ", " ... ", ...] }
        ```'''
        
        user_prompt = f"Original Query: {query}\nNumber of rewrites: {num_rewrites}"
        
        try:
            response = self.model_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                
                return rewrites
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return [query]
                
        except Exception as e:
            logger.error(f"Error generating query rewrites: {str(e)}")
            # Return original query if rewriting fails
            return [query]

    def retrieve_nodes(self, query: str | List[str], top_k: int, apply_filters: bool = False) -> Tuple[List[dict], List[str]]:
        """
        Retrieve relevant context from Pinecone for one or multiple queries.
        
        Args:
            query: The user query or list of queries. If a list is provided, time filters are 
                  extracted from the first query and applied to all retrievals.
            top_k: Number of context chunks to retrieve (default from settings)
            apply_filters: Whether to apply time filters extracted from the query (default False)
            
        Returns:
            Tuple containing (list of relevant context dicts with context and timestamp, list of unique source links)
        """
        # Handle single query case by converting to list
        queries = [query] if isinstance(query, str) else query
        if not queries:
            logger.warning("No queries provided to retrieve_nodes")
            return SimpleNamespace(matches=[])
            
        logger.info(f"Retrieving nodes for {len(queries)} queries with top_k={top_k}")
        
        filter_dict = {}
        # Extract filters from the first query only if apply_filters is True
        if apply_filters:
            logger.info("Applying time filters from first query")
            # Extract filters from the first query
            time_filters = self.extract_time_filters(queries[0], self.model_client)
            
            # Prepare filter dict for Pinecone
            if time_filters["start_timestamp"]:
                filter_dict["timestamp"] = {"$gte": time_filters["start_timestamp"]}
                logger.debug(f"Applied start timestamp filter: {time_filters['start_timestamp']}")
            if time_filters["end_timestamp"]:
                if "timestamp" in filter_dict:
                    filter_dict["timestamp"]["$lte"] = time_filters["end_timestamp"]
                else:
                    filter_dict["timestamp"] = {"$lte": time_filters["end_timestamp"]}
                logger.debug(f"Applied end timestamp filter: {time_filters['end_timestamp']}")
        
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
                    filter=filter_dict if filter_dict else None
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
                    self.reranker = CrossEncoderReranker(self.model_client)
                    reranked_results = self.reranker.rerank(query, user_filtered_results, top_k=rerank_top_k)
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
            nodes = self.retrieve_nodes(all_queries, top_k=top_k, apply_filters=apply_filters)
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
        system_prompt = f'''You are a helpful technical project manager that answers your teammates' questions based on the provided context. 
        When recent conversations are provided, use them to maintain consistency with previous responses. 
        User cited context serves as reference for the user query if it is provided.
        '''
        
        # Log the prompt being sent to the model
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {prompt}")

        try:
            response = self.model_client.chat.completions.create(
                model=model,
                max_tokens=1024,  # Allow for longer responses
                messages=[
                    {"role": "system", "content": system_prompt},
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
        
        system_prompt = '''You are **SearchQueryRewriter‑MQR**, a large‑language‑model agent that creates
        *diverse, high‑recall* rewrites of a user's information‑seeking query.

        ## Instructions
        1. Read the **Original Query**.
        2. Produce **{{N}}** DISTINCT rewrites (do **NOT** answer the question).
        3. Follow these rewriting strategies *at least once each*  
        a. **Equality** – preserve all meaning; just de‑chatify the wording.  
        b. **Expansion** – add missing context a domain expert would expect  
            (e.g., synonyms, acronyms, date ranges, entity types).  
        c. **Reduction** – strip to the absolute core keywords.  
        d. *(Optional if N > 2)* Other creative perspectives that could surface
            different documents (e.g., broader background, comparison terms).
        4. **Constraints**  
        • ≤ 20 tokens per rewrite.  
        • Remove pronouns/ellipsis; name all entities explicitly.  
        • Avoid stop‑words unless essential (e.g., "of", "in").  
        • No duplicate semantic meaning across rewrites.
        5. Return a **valid JSON** object ONLY without any other text:

        ```json
        { "rewrites": [" ... ", " ... ", ...] }
        ```'''
        
        user_prompt = f"Original Query: {query}\nNumber of rewrites: {num_rewrites}"
        
        try:
            # Run the model call in executor to make it non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,  # Use some temperature for diversity
                    response_format={"type": "json_object"}
                )
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
                
                return rewrites
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return [query]
                
        except Exception as e:
            logger.error(f"Error generating query rewrites: {str(e)}")
            # Return original query if rewriting fails
            return [query]
    
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
                    self.reranker = CrossEncoderReranker(self.model_client)
                    reranked_results = await self.async_rerank(query, user_filtered_results)
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
        
    async def _async_retrieve_and_process_context(self, query_with_date, top_k=RETRIEVE_TOP_K, rerank_top_k=None, apply_filters=APPLY_FILTERS, use_reranker=USE_RERANKER, query_rewrites=QUERY_REWRITES):
        """Asynchronous helper method to retrieve and process context"""
        # Retrieve nodes (this is I/O bound, so we'll run it in the executor)
        loop = asyncio.get_event_loop()
        try:
            # Generate query rewrites if needed
            all_queries = [query_with_date]
            if query_rewrites:
                rewrites = await self.async_rewrite_query(query_with_date)
                all_queries.extend(rewrites)
                logger.info(f"Generated {len(rewrites)} query rewrites: {rewrites}")
            
            # Use run_in_executor for the blocking retrieve_nodes operation
            nodes = await loop.run_in_executor(
                None, 
                lambda: self.retrieve_nodes(all_queries, top_k, apply_filters)
            )
            
            # Use async postprocessing with non-blocking reranking
            context, data_sources = await self.async_postprocess_nodes(
                nodes, 
                query_with_date, 
                apply_filters=apply_filters, 
                use_reranker=use_reranker, 
                rerank_top_k=rerank_top_k
            )
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

        # Create tasks for async processing
        rerank_top_k = kwargs.get("rerank_top_k", RERANK_TOP_K)
        context_task = asyncio.create_task(self._async_retrieve_and_process_context(
            full_query, 
            top_k=top_k, 
            rerank_top_k=rerank_top_k,
            apply_filters=apply_filters,
            use_reranker=use_reranker,
            query_rewrites=query_rewrites
        ))
        
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

        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": system_prompt},
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
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.add_memory, query, answer)
        
        # Return dictionary with content and data_sources
        return {
            "content": answer,
            "data_sources": data_sources
        }