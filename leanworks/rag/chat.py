from pinecone import Pinecone
from typing import List, Dict, Tuple, Any
from leanworks.rag.filters import FilterExtractor
from leanworks.agent.memory import MemoryManager
from leanworks.rag.reranker.reranker_factory import RerankerFactory
from leanworks.rag.span_selection import SpanSelector
from leanworks.rag.data_source_formatter import DataSourceFormatter
from leanworks.setting import *
from leanworks.rag.query import QueryRewriter
import datetime
import logging
import asyncio
import json
from types import SimpleNamespace
# Set up logging
logger = logging.getLogger(__name__)

class Chat(FilterExtractor, MemoryManager, QueryRewriter):
    """
    Chat class for retrieving context from Pinecone and generating responses using OpenAI.
    Provides synchronous functionality for RAG operations.
    """
    def __init__(
        self,
        vectordb_client,
        firestore_client,
        org_slug,
        model_client,
        user_id: str | None = None,
        session_id: str | None = None
    ):
        """
        Initialize Chat with vector database client and memory management.
        
        Args:
            vectordb_client: Initialized PineconeHybridIndex client for hybrid search
            firestore_client: Firestore client for memory persistence
            org_slug: Organization name for Firestore path
            model_client: Initialized OpenAI client for LLM generation
            user_id: ID of the user
            session_id: ID of the current conversation session
        """
        
        # Use the provided vector database client
        self.vectordb_client = vectordb_client
        
        self.model_client = model_client
        # Store org_slug for namespace usage in Pinecone queries
        self.org_slug = org_slug
        # Initialize memory manager if user_id and session_id are provided
        self.memory_enabled = user_id is not None and session_id is not None
        if self.memory_enabled:
            MemoryManager.__init__(self, model_client, firestore_client, org_slug, user_id, session_id)
            logger.info(f"Memory enabled for user_id: {user_id}, session_id: {session_id}")
        else:
            logger.info("Memory disabled - no user_id or session_id provided")
            
        # Initialize the FilterExtractor part
        FilterExtractor.__init__(self)
        
        # Initialize QueryRewriter
        QueryRewriter.__init__(self, model_client)

        # Initialize the reranker using factory
        self.reranker = RerankerFactory.create_reranker(
            reranker_type=RERANKER_TYPE,
            model_client=model_client
        )
        
        # Initialize span selector with hybrid scoring
        from leanworks.setting import (
            USE_HYBRID_SPAN_SELECTION, 
            SPAN_SELECTION_RRF_K, 
            SPAN_SELECTION_TOP_SENTENCES, 
            SPAN_SELECTION_CONTEXT_WINDOW
        )
        # Reuse the existing reranker for span selection to avoid loading models twice
        self.span_selector = SpanSelector(
            top_spans_per_doc=SPAN_SELECTION_TOP_SENTENCES,
            context_window=SPAN_SELECTION_CONTEXT_WINDOW,
            min_span_length=20,
            max_span_length=400,
            reranker=self.reranker,  # Reuse existing reranker (LLM or BGE)
            span_selection_type=SPAN_SELECTION_TYPE,
            use_sliding_windows=True,  # Use sliding windows for better semantic recall
            window_size=96,  # 96 tokens per window
            window_stride=48,  # 48 token stride
            max_span_candidates=60,  # Cap total candidates to 60
            max_final_spans=18,  # Return top 18 spans globally
            use_bm25_prefilter=True,  # Use BM25 pre-filtering
            bm25_k1=1.2,
            bm25_b=0.75
        )
        
        
        # Initialize data source formatter
        self.data_source_formatter = DataSourceFormatter()
            
        logger.info("RAG system initialized successfully")

    def _extract_timestamp_from_context(self, context_text: str) -> str:
        """Extract timestamp from context text."""
        import re
        from datetime import datetime
        
        # Common timestamp patterns
        patterns = [
            r'date[:\s]+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:[+-]\d{2}:\d{2}|Z)?)',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:[+-]\d{2}:\d{2}|Z)?)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context_text, re.IGNORECASE)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Try to parse the timestamp
                    if 'T' in timestamp_str:
                        # ISO format
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str[:-1] + '+00:00'
                        dt = datetime.fromisoformat(timestamp_str)
                    else:
                        # Date only or simple format
                        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S' if ' ' in timestamp_str else '%Y-%m-%d')
                    
                    return dt.isoformat()
                except ValueError:
                    continue
        
        # Check for Unix timestamps (both integer and float formats)
        unix_patterns = [
            r'timestamp[:\s]+(\d{10,13}(?:\.\d+)?)',  # timestamp: 1756087205.146079
            r'(\d{10,13}(?:\.\d+)?)',  # standalone Unix timestamp
        ]
        
        for pattern in unix_patterns:
            match = re.search(pattern, context_text)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Convert Unix timestamp to datetime
                    unix_timestamp = float(timestamp_str)
                    # Handle both seconds and milliseconds
                    if unix_timestamp > 1e10:  # Likely milliseconds
                        unix_timestamp = unix_timestamp / 1000
                    dt = datetime.fromtimestamp(unix_timestamp)
                    return dt.isoformat()
                except (ValueError, OSError):
                    continue
        
        return None

    def retrieve_nodes(self, query: str | List[str], top_k: int, filters: dict = None, alpha: float = ALPHA) -> SimpleNamespace:
        """
        Retrieve relevant context using hybrid search for one or multiple queries.
        
        Args:
            query: The user query or list of queries. If a list is provided, hybrid search is 
                  performed for each query and results are combined.
            top_k: Number of context chunks to retrieve (default from settings)
            filters: Dictionary of filters to apply to the query (default None)
            
        Returns:
            SimpleNamespace with 'matches' attribute containing hybrid search results
        """
        # Handle single query case by converting to list
        queries = [query] if isinstance(query, str) else query
        if not queries:
            logger.warning("No queries provided to retrieve_nodes")
            return SimpleNamespace(matches=[])
            
        logger.info(f"Retrieving nodes using hybrid search for {len(queries)} queries with top_k={top_k}")
        
        # Results container
        all_matches = []
        
        # Perform hybrid search for each query
        try:
            # Use org_slug directly as namespace (can contain underscores)
            namespace = self.org_slug if self.org_slug else ""
            
            for q in queries:
                # Use hybrid search from PineconeHybridIndex
                hybrid_results = self.vectordb_client.hybrid_search(
                    query=q,
                    top_k=top_k,
                    alpha=alpha,
                    namespace=namespace,
                    filter=filters
                )
                
                # Convert hybrid search results to match Pinecone response format
                for result in hybrid_results:
                    # Create match object with same structure as Pinecone response
                    match = SimpleNamespace(
                        id=result['id'],
                        score=result['combined_score'],  # Use combined hybrid score
                        metadata=result['metadata']
                    )
                    all_matches.append(match)
                
                logger.info(f"Retrieved {len(hybrid_results)} hybrid search results")
                
            # Deduplicate matches by ID
            seen_ids = set()
            unique_matches = []
            for match in all_matches:
                if match.id not in seen_ids:
                    seen_ids.add(match.id)
                    unique_matches.append(match)
            
            # Sort by combined score (highest first) and limit to top_k
            unique_matches.sort(key=lambda x: x.score, reverse=True)
            
            # Create a new response object with deduplicated matches
            combined_response = SimpleNamespace(
                matches=unique_matches[:top_k]  # Limit to top_k after deduplication
            )
            logger.info(f"Combined and deduplicated to {len(combined_response.matches)} hybrid search results")
            return combined_response
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            # Return an empty result structure with similar interface as Pinecone response
            empty_response = SimpleNamespace(matches=[])
            return empty_response
    
    def postprocess_nodes(
            self, nodes: List[dict], 
            query: str, 
            use_reranker: bool = USE_RERANKER, 
            use_span_selection: bool = USE_SPAN_SELECTION,
            min_score_threshold: float = MIN_SCORE_THRESHOLD,
            recency_weight: float = RECENCY_WEIGHT,
            recency_coefficient: float = RECENCY_COEFFICIENT,
            **kwargs
            ) -> Tuple[List[dict], List[str]]:
        """
        Process retrieved nodes from Pinecone and extract context information.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            use_reranker: Whether to apply reranking (default None, which falls back to instance setting)
            use_span_selection: Whether to apply span selection
            min_score_threshold: Minimum score threshold for filtering results
            recency_weight: Weight for recency scoring
            recency_coefficient: Coefficient for recency calculation
            read_document_ids: Set of document IDs already read to skip duplicates
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        logger.debug("Initial documents", nodes.matches)
        
        # Get read document IDs for deduplication
        read_document_ids = kwargs.get("read_document_ids")
        if read_document_ids is None:
            logger.warning("No read_document_ids provided for deduplication - creating temporary set")
            read_document_ids = set()
        
        logger.info(f"Starting sync deduplication with {len(read_document_ids)} already read documents")
        
        # Filter results by relevance score AND deduplication
        filtered_results = []
        skipped_count = 0
        for match in nodes.matches:
            if match.id not in read_document_ids:
                filtered_results.append(match)
                # Add this document ID to the read set
                read_document_ids.add(match.id)
                logger.debug(f"Added new document ID: {match.id}")
            else:
                skipped_count += 1
                logger.info(f"Skipped duplicate document ID: {match.id}")
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} duplicate documents that were already read")
        
        logger.info(f"After sync processing: {len(read_document_ids)} total documents in read set")
            
        logger.debug("Score filtered and deduplicated documents", filtered_results)

        # Apply reranking if enabled and needed (directly on the filtered results)
        rerank_top_k = kwargs.get("rerank_top_k")
        if use_reranker and filtered_results:
            if rerank_top_k is None:
                raise ValueError("rerank_top_k is required when use_reranker is True")
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(filtered_results):
                logger.info(f"Applying reranking to get top {rerank_top_k} documents...")
                try:
                    # Use the reranker to improve precision
                    logger.info(f"Using {RERANKER_TYPE} reranker")
                    reranked_results = self.reranker.rerank(query, filtered_results, top_k=rerank_top_k, min_score_threshold=min_score_threshold, recency_weight=recency_weight, recency_coefficient=recency_coefficient)
                    logger.info(f"Successfully reranked results to {len(reranked_results)} documents")
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    reranked_results = filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                # Note: Timestamp-based sorting removed as timestamp fields are no longer used
                reranked_results = filtered_results[:rerank_top_k]
        else:
            # Note: Timestamp-based sorting removed as timestamp fields are no longer used
            reranked_results = filtered_results[:rerank_top_k]
        logger.debug("Reranked documents", reranked_results)

        # Apply span selection if enabled
        if use_span_selection and reranked_results:
            try:
                logger.info("Applying span selection to select relevant sentences...")
                reranked_results = self.span_selector.select_spans(query, reranked_results)
                
                # Log span selection statistics
                stats = self.span_selector.get_selection_stats(reranked_results)
                logger.info(f"Span selection stats: {stats}")
            except Exception as e:
                logger.error(f"Error during span selection: {str(e)}, proceeding without span selection")

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in reranked_results:
            # Extract source information
            data_source = match.metadata["data_source"]
            links.add(match.metadata["link"])
            
            # Extract context text using various fallback methods
            context_text = match.metadata.get("chunk_text", "")
            
            # Skip duplicates
            if not context_text or context_text in seen_contexts:
                continue
                    
            seen_contexts.add(context_text)
            
            # Add to contexts list
            contexts.append({
                "context": context_text,
                "data_source": data_source,
                "doc_id": match.id
            })
        
        
        logger.debug(f"Final contexts: {contexts}")
        
        # Use the scalable data source formatter
        formatted_data_sources = self.data_source_formatter.format_data_sources(links, contexts, simple_mode=True, show_all_links=False, raw_links_only=False)
        
        return contexts, formatted_data_sources
    
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
            query_rewrites: bool = QUERY_REWRITES,
            use_span_selection: bool = USE_SPAN_SELECTION,
            min_score_threshold: float = MIN_SCORE_THRESHOLD,
            recency_weight: float = RECENCY_WEIGHT,
            recency_coefficient: float = RECENCY_COEFFICIENT,
            cited_context: dict = None,
            alpha: float = ALPHA,
            **kwargs
            ) -> Dict[str, any]:
        """
        Generate a response using RAG approach.
        
        Args:
            query: The user query
            model: The model to use for generation
            include_memory: Whether to include recent conversation history
            use_span_selection: Whether to apply span selection to extract relevant sentences
            min_score_threshold: Minimum score threshold for filtering results
            recency_weight: Weight for recency scoring
            recency_coefficient: Coefficient for recency calculation
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
            else:
                all_queries = [full_query]
            # Note: Time filters disabled as timestamp fields are no longer used in context structure
            # Timestamp information is now extracted from context text when needed for display
            filters = None
            nodes = self.retrieve_nodes(all_queries, top_k=top_k, filters=filters, alpha=alpha)
            context, data_sources = self.postprocess_nodes(
                nodes, 
                full_query, 
                use_reranker=use_reranker, 
                use_span_selection=use_span_selection,
                min_score_threshold=min_score_threshold,
                recency_weight=recency_weight,
                recency_coefficient=recency_coefficient,
                rerank_top_k=rerank_top_k,
                **kwargs
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
            # Extract timestamp from context text if available
            timestamp_str = ""
            extracted_timestamp = self._extract_timestamp_from_context(ctx.get("context", ""))
            if extracted_timestamp:
                timestamp_str = f" (from {extracted_timestamp})"
            
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
        vectordb_client,
        firestore_client,
        org_slug,
        model_client,
        user_id: str | None = None,
        session_id: str | None = None
    ):
        """Initialize AsyncChat with the same parameters as Chat."""
        super().__init__(
            vectordb_client=vectordb_client,
            firestore_client=firestore_client,
            org_slug=org_slug,
            model_client=model_client,
            user_id=user_id,
            session_id=session_id
        )
        logger.info("AsyncChat initialized successfully")
    
    async def async_postprocess_nodes(
            self, nodes: List[dict], 
            query: str, 
            use_reranker: bool = USE_RERANKER, 
            use_span_selection: bool = USE_SPAN_SELECTION,
            min_score_threshold: float = MIN_SCORE_THRESHOLD,
            recency_weight: float = RECENCY_WEIGHT,
            recency_coefficient: float = RECENCY_COEFFICIENT,
            **kwargs
        ) -> Tuple[List[dict], List[str]]:
        """
        Asynchronous version of postprocess_nodes that uses async reranking for better performance.
        
        Args:
            nodes: The query results from Pinecone
            query: The user query
            use_reranker: Whether to apply reranking
            use_span_selection: Whether to apply span selection
            min_score_threshold: Minimum score threshold for filtering results
            recency_weight: Weight for recency scoring
            recency_coefficient: Coefficient for recency calculation
            read_document_ids: Set of document IDs already read to skip duplicates
            
        Returns:
            Tuple containing (list of context dicts with text/timestamp/source, list of unique data sources)
        """
        logger.info(f"Async postprocessing {len(nodes.matches) if hasattr(nodes, 'matches') else 0} retrieved nodes")
        logger.debug("Initial documents", nodes.matches)
        
        # Get read document IDs for deduplication
        read_document_ids = kwargs.get("read_document_ids")
        if read_document_ids is None:
            logger.warning("No read_document_ids provided for deduplication - creating temporary set")
            read_document_ids = set()
        
        logger.info(f"Starting async deduplication with {len(read_document_ids)} already read documents")
        
        # Filter results by relevance score AND deduplication
        filtered_results = []
        skipped_count = 0
        for match in nodes.matches:
            if match.id not in read_document_ids:
                filtered_results.append(match)
                # Add this document ID to the read set
                read_document_ids.add(match.id)
                logger.debug(f"Added new document ID: {match.id}")
            else:
                skipped_count += 1
                logger.info(f"Skipped duplicate document ID: {match.id}")
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} duplicate documents that were already read")
        
        logger.info(f"After async processing: {len(read_document_ids)} total documents in read set")
        
        logger.debug("Score filtered and deduplicated documents", filtered_results)
            
        # Apply reranking if enabled and needed (directly on the filtered results)
        rerank_top_k = kwargs.get("rerank_top_k")
        if use_reranker and filtered_results:
            if rerank_top_k is None:
                raise ValueError("rerank_top_k is required when use_reranker is True")
            # Check if reranking should be applied based on result quality
            if self._should_apply_reranking(filtered_results):
                logger.info(f"Applying async reranking to get top {rerank_top_k} documents...")
                try:
                    # Use async reranker to improve precision without blocking
                    reranked_results = await self.reranker.rerank_async(
                        query, 
                        filtered_results,
                        top_k=rerank_top_k,
                        min_score_threshold=min_score_threshold,
                        recency_weight=recency_weight,
                        recency_coefficient=recency_coefficient
                    )
                    logger.info(f"Successfully reranked results to {len(reranked_results)} documents")
                except Exception as e:
                    logger.error(f"Error during async reranking: {str(e)}, falling back to vector rankings")
                    # In case of error, limit to top_k based on vector similarity
                    reranked_results = filtered_results[:rerank_top_k]
            else:
                logger.info("Skipping reranking due to high quality initial results")
                # Just take the top results without reranking
                # Note: Timestamp-based sorting removed as timestamp fields are no longer used
                reranked_results = filtered_results[:rerank_top_k]
        else:
            # Note: Timestamp-based sorting removed as timestamp fields are no longer used
            reranked_results = filtered_results[:rerank_top_k]
        logger.debug("Reranked documents", reranked_results)

        # Apply span selection if enabled
        if use_span_selection and reranked_results:
            try:
                logger.info("Applying span selection to select relevant sentences...")
                reranked_results = self.span_selector.select_spans(query, reranked_results)
                
                # Log span selection statistics
                stats = self.span_selector.get_selection_stats(reranked_results)
                logger.info(f"Span selection stats: {stats}")
            except Exception as e:
                logger.error(f"Error during span selection: {str(e)}, proceeding without span selection")

        # Extract text from metadata
        contexts = []
        links = set()
        seen_contexts = set()
        
        for match in reranked_results:
            # Extract source information
            data_source = match.metadata["data_source"]
            links.add(match.metadata["link"])
            
            # Extract context text directly from metadata
            context_text = match.metadata.get("chunk_text", "")
            
            # Skip duplicates
            if not context_text or context_text in seen_contexts:
                continue
                
            seen_contexts.add(context_text)
            
            # Add to contexts list
            contexts.append({
                "context": context_text,
                "data_source": data_source,
                "doc_id": match.id
            })
        
        
        logger.debug(f"Final contexts: {contexts}")
        
        # Use the scalable data source formatter
        formatted_data_sources = self.data_source_formatter.format_data_sources(links, contexts, simple_mode=True, show_all_links=False, raw_links_only=False)
        
        return contexts, formatted_data_sources
        
    async def async_get_response(
            self, query: str, 
            model: str = GENERATION_MODEL,
            top_k: int = RETRIEVE_TOP_K,
            include_memory: bool = INCLUDE_MEMORY,
            use_reranker: bool = USE_RERANKER,
            query_rewrites: bool = QUERY_REWRITES,
            use_span_selection: bool = USE_SPAN_SELECTION,
            min_score_threshold: float = MIN_SCORE_THRESHOLD,
            recency_weight: float = RECENCY_WEIGHT,
            recency_coefficient: float = RECENCY_COEFFICIENT,
            cited_context: dict = None,
            alpha: float = ALPHA,
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
            query_rewrites: Whether to use query rewriting for better recall
            use_span_selection: Whether to apply span selection to extract relevant sentences
            cited_context: Specific context cited by the user, a dictionary
            min_score_threshold: Minimum score threshold for filtering results
            recency_weight: Weight for recency scoring
            recency_coefficient: Coefficient for recency calculation
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
        
        # Note: Time filters disabled as timestamp fields are no longer used in context structure
        # Timestamp information is now extracted from context text when needed for display
        filters_task = None
            
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
                
        # Note: Time filters disabled as timestamp fields are no longer used in context structure
        filters = None
                
        # Log to make it clear what queries are being used
        logger.info(f"Using queries for retrieval: {all_queries} with filters: {filters}")
        
        # Retrieve nodes (blocking operation, run in executor)
        loop = asyncio.get_event_loop()
        try:
            nodes = await loop.run_in_executor(
                None, 
                lambda: self.retrieve_nodes(all_queries, top_k, filters, alpha)
            )
            
            # Use async postprocessing with non-blocking reranking
            context, data_sources = await self.async_postprocess_nodes(
                nodes, 
                full_query, 
                use_reranker=use_reranker,
                use_span_selection=use_span_selection,
                min_score_threshold=min_score_threshold,
                recency_weight=recency_weight,
                recency_coefficient=recency_coefficient,
                rerank_top_k=rerank_top_k,
                **kwargs
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
            # Extract timestamp from context text if available
            timestamp_str = ""
            extracted_timestamp = self._extract_timestamp_from_context(ctx.get("context", ""))
            if extracted_timestamp:
                timestamp_str = f" (from {extracted_timestamp})"
            
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