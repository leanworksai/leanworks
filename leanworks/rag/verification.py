import logging
import json
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import datetime
import numpy as np
from openai import types
from functools import lru_cache
from pinecone import Pinecone
from leanworks.rag.setting import RETRIEVE_TOP_K, SIMILARITY_CUTOFF

# Set up logging
logger = logging.getLogger(__name__)

class AnswerVerifier:
    """
    A verification module that uses a mini RAG system to verify and correct answers 
    from a generation model using the same context sources.
    """
    
    def __init__(self, model_client, embedding_model_client=None, pinecone_api_key=None, index_host=None):
        """
        Initialize the answer verifier with model clients and optionally Pinecone for standalone mini RAG.
        
        Args:
            model_client: The model client for verification (e.g., OpenAI client)
            embedding_model_client: Optional embedding model client for semantic analysis
            pinecone_api_key: Optional Pinecone API key for standalone operation
            index_host: Optional Pinecone index host for standalone operation
        """
        self.model_client = model_client
        self.embedding_model_client = embedding_model_client
        
        # Initialize Pinecone if credentials are provided (for standalone operation)
        self.standalone_mode = pinecone_api_key is not None and index_host is not None
        if self.standalone_mode:
            pc = Pinecone(api_key=pinecone_api_key)
            self.index = pc.Index(host=index_host)
            logger.info("Answer verifier initialized in standalone mode with Pinecone")
        else:
            self.index = None
            logger.info("Answer verifier initialized in integrated mode (using external context)")
    
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
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                # Wait for result with 30 second timeout
                result = future.result(timeout=30)
                return np.array(result.embeddings[0].values)
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero embedding as fallback
            return np.zeros(768)  # Standard embedding dimension
        
    def retrieve_nodes(self, query: str, top_k: int = RETRIEVE_TOP_K) -> List[dict]:
        """
        Retrieve relevant context from Pinecone for verification.
        Requires the verifier to be initialized with Pinecone credentials.
        
        Args:
            query: The user query
            top_k: Number of context chunks to retrieve
            
        Returns:
            List of relevant context dictionaries with metadata
        """
        if not self.standalone_mode or self.index is None:
            logger.error("Cannot retrieve nodes - verifier not initialized with Pinecone credentials")
            return []
            
        logger.info(f"Retrieving nodes for verification: '{query}' with top_k={top_k}")
        query_embedding = self._get_embedding(query)
        
        # Query Pinecone
        try:
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            matches = getattr(response, 'matches', [])
            logger.info(f"Retrieved {len(matches)} nodes from Pinecone for verification")
            
            # Convert Pinecone response to context format
            contexts = []
            for match in matches:
                if match.score < SIMILARITY_CUTOFF:
                    continue
                    
                metadata = match.metadata
                context_text = metadata.get("text", "")
                if not context_text and "_node_content" in metadata:
                    try:
                        node_content = json.loads(metadata.get("_node_content", ""))
                        context_text = node_content.get("text", "") or metadata.get("context", "")
                    except:
                        context_text = metadata.get("context", "")
                
                if not context_text:
                    continue
                    
                contexts.append({
                    "context": context_text,
                    "timestamp": metadata.get("timestamp"),
                    "data_source": metadata.get("data_source", "unknown"),
                    "score": match.score
                })
                
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying Pinecone for verification: {str(e)}")
            return []
    
    async def async_retrieve_nodes(self, query: str, top_k: int = RETRIEVE_TOP_K) -> List[dict]:
        """
        Asynchronously retrieve relevant context from Pinecone for verification.
        
        Args:
            query: The user query
            top_k: Number of context chunks to retrieve
            
        Returns:
            List of relevant context dictionaries with metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve_nodes, query, top_k)
    
    def verify_with_mini_rag(
        self,
        answer: str,
        query: str,
        verification_model: str,
        top_k: int = RETRIEVE_TOP_K,
        max_iterations: int = 2,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verify an answer using a mini RAG system by retrieving context and verifying against it.
        This method retrieves its own context from the vector store using both the original query 
        and the generated answer to form a more targeted retrieval query.
        
        Args:
            answer: The answer to verify
            query: The original user query
            verification_model: The model to use for verification
            top_k: Number of contexts to retrieve
            max_iterations: Maximum number of verification iterations
            confidence_threshold: Threshold for correction confidence
            
        Returns:
            Dictionary with verified answer and metadata
        """
        if not self.standalone_mode:
            logger.error("Cannot use mini RAG verification - verifier not in standalone mode")
            return {
                "original_answer": answer,
                "verified_answer": answer,
                "was_corrected": False,
                "error": "Verifier not initialized with Pinecone credentials"
            }
            
        # Create an enhanced query combining the original query and answer
        # to target retrieval for verification purposes
        enhanced_query = f"{query} {answer}"
        logger.info(f"Creating enhanced verification query combining original query and answer")
        
        # Retrieve relevant context for verification using the enhanced query
        contexts = self.retrieve_nodes(enhanced_query, top_k=top_k)
        
        if not contexts:
            logger.warning("No contexts retrieved for verification, returning original answer")
            return {
                "original_answer": answer,
                "verified_answer": answer,
                "was_corrected": False
            }
            
        # Verify the answer against retrieved contexts
        return self.verify_answer(
            answer=answer,
            context=contexts,
            query=query,
            verification_model=verification_model,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold
        )
    
    async def async_verify_with_mini_rag(
        self,
        answer: str,
        query: str,
        verification_model: str,
        top_k: int = RETRIEVE_TOP_K,
        max_iterations: int = 2,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Asynchronously verify an answer using a mini RAG system by retrieving context and verifying against it.
        Uses both the original query and the generated answer to form a more targeted retrieval query.
        
        Args:
            answer: The answer to verify
            query: The original user query
            verification_model: The model to use for verification
            top_k: Number of contexts to retrieve
            max_iterations: Maximum number of verification iterations
            confidence_threshold: Threshold for correction confidence
            
        Returns:
            Dictionary with verified answer and metadata
        """
        if not self.standalone_mode:
            logger.error("Cannot use mini RAG verification - verifier not in standalone mode")
            return {
                "original_answer": answer,
                "verified_answer": answer,
                "was_corrected": False,
                "error": "Verifier not initialized with Pinecone credentials"
            }
            
        # Create an enhanced query combining the original query and answer
        # to target retrieval for verification purposes
        enhanced_query = f"{query} {answer}"
        logger.info(f"Creating enhanced verification query combining original query and answer")
        
        # Retrieve relevant context for verification using the enhanced query
        contexts = await self.async_retrieve_nodes(enhanced_query, top_k=top_k)
        
        if not contexts:
            logger.warning("No contexts retrieved for verification, returning original answer")
            return {
                "original_answer": answer,
                "verified_answer": answer,
                "was_corrected": False
            }
            
        # Verify the answer against retrieved contexts
        return await self.async_verify_answer(
            answer=answer,
            context=contexts,
            query=query,
            verification_model=verification_model,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold
        )
    
    def verify_answer(
        self, 
        answer: str, 
        context: List[dict], 
        query: str, 
        verification_model: str,
        max_iterations: int = 2,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verify an answer against the context and correct any factual inaccuracies.
        
        Args:
            answer: The answer to verify
            context: List of context dictionaries with retrieved information
            query: The original user query
            verification_model: The model to use for verification
            max_iterations: Maximum number of verification iterations
            confidence_threshold: Threshold for correction confidence
            
        Returns:
            Dictionary with verified answer and metadata
        """
        logger.info(f"Starting verification process with {len(context)} context chunks")
        
        # Track all verification iterations
        verification_history = []
        current_answer = answer
        
        for iteration in range(max_iterations):
            logger.info(f"Verification iteration {iteration+1}/{max_iterations}")
            
            # Run verification step
            verification_result = self._run_verification_step(
                current_answer, context, query, verification_model
            )
            
            verification_history.append(verification_result)
            
            # If no issues found, break the loop
            if not verification_result.get("issues_found", False):
                logger.info("No issues found in this iteration")
                break
                
            # Get corrected answer and confidence
            corrected_answer = verification_result.get("corrected_answer", current_answer)
            overall_confidence = verification_result.get("overall_confidence", 0.0)
            
            # Log corrections
            corrections = verification_result.get("corrections", [])
            for correction in corrections:
                logger.info(f"Correction (confidence: {correction.get('confidence', 0.0)}): "
                            f"'{correction.get('original', '')}' -> '{correction.get('corrected', '')}'")
            
            # Update current answer if confidence is sufficient
            if overall_confidence >= confidence_threshold:
                logger.info(f"Applied corrections with confidence {overall_confidence}")
                current_answer = corrected_answer
            else:
                logger.info(f"Skipping corrections due to low confidence: {overall_confidence}")
                break
                
        # Determine final answer
        was_corrected = current_answer != answer
        
        # Prepare result
        result = {
            "original_answer": answer,
            "verified_answer": current_answer,
            "was_corrected": was_corrected,
            "verification_history": verification_history,
            "correction_count": len(verification_history)
        }
        
        # Add note if corrections were made
        if was_corrected:
            result["verified_answer"] += "\n\n(Note: This response has been verified and corrected for factual accuracy.)"
            
        return result
    
    def _run_verification_step(
        self, 
        answer: str, 
        context: List[dict], 
        query: str, 
        model: str
    ) -> Dict[str, Any]:
        """
        Run a single verification step to identify and correct issues.
        
        Args:
            answer: The answer to verify
            context: List of context dictionaries
            query: The original user query
            model: The model to use for verification
            
        Returns:
            Verification result dictionary
        """
        # Prepare context string for verification
        context_for_verification = self._prepare_context_for_verification(context)
        
        # Craft verification prompt
        verification_prompt = self._create_verification_prompt(
            context_for_verification, query, answer
        )
        
        # Try to get verification and correction
        try:
            # Add timeout handling for verification
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.model_client.chat.completions.create,
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an expert fact-checker that identifies and corrects factual errors based strictly on the provided context."},
                        {"role": "user", "content": verification_prompt}
                    ]
                )
                # Wait for result with 60 second timeout
                response = future.result(timeout=60)
                verification_result = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                verification_data = json.loads(verification_result)
                return verification_data
            except json.JSONDecodeError:
                logger.error("Failed to parse verification result as JSON")
                return {"issues_found": False, "error": "JSON parsing error"}
                
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.error(f"Error during verification step: {str(e)}")
            return {"issues_found": False, "error": str(e)}
    
    def _prepare_context_for_verification(self, context: List[dict]) -> str:
        """
        Prepare context for verification by combining context items with metadata.
        
        Args:
            context: List of context dictionaries
            
        Returns:
            String containing formatted context for verification
        """
        context_parts = []
        
        for i, ctx in enumerate(context):
            context_text = ctx.get("context", "")
            if not context_text:
                continue
                
            # Add metadata if available
            metadata = []
            if ctx.get("timestamp"):
                try:
                    timestamp = datetime.datetime.fromtimestamp(ctx["timestamp"], tz=datetime.timezone.utc)
                    metadata.append(f"Date: {timestamp.isoformat()}")
                except (TypeError, ValueError):
                    pass
                    
            if ctx.get("data_source"):
                metadata.append(f"Source: {ctx.get('data_source')}")
                
            metadata_str = f" [{', '.join(metadata)}]" if metadata else ""
            
            # Add formatted context
            context_parts.append(f"Context {i+1}{metadata_str}:\n{context_text}\n")
            
        return "\n".join(context_parts)
    
    def _create_verification_prompt(
        self, 
        context: str, 
        query: str, 
        answer: str
    ) -> str:
        """
        Create a verification prompt for the fact-checking model.
        
        Args:
            context: Formatted context string
            query: Original user query
            answer: Answer to verify
            
        Returns:
            Verification prompt string
        """
        return f"""
Your task is to verify if the following AI-generated answer is factually accurate based on the provided context.

CONTEXT INFORMATION:
{context}

USER QUERY:
{query}

AI-GENERATED ANSWER:
{answer}

Please analyze the AI-generated answer and:
1. Identify any statements that contradict the context information
2. Identify any claims that are not supported by the context
3. For each issue, explain why it's incorrect based on the context
4. Provide a corrected version of each problematic statement
5. Provide a confidence score (0.0-1.0) for each correction

Format your response as a JSON object with these fields:
- "issues_found": boolean (true if any issues were found, false if not)
- "analysis": list of identified issues, each with "statement", "explanation", and "severity" fields
- "corrections": list of objects with "original" text, "corrected" text, and "confidence" score
- "corrected_answer": the original answer with all corrections applied
- "overall_confidence": average confidence score across all corrections (0.0-1.0)

ONLY return the JSON object and nothing else.
"""

    async def async_verify_answer(
        self, 
        answer: str, 
        context: List[dict], 
        query: str, 
        verification_model: str,
        max_iterations: int = 2,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Asynchronously verify an answer against the context.
        
        Args:
            answer: The answer to verify
            context: List of context dictionaries with retrieved information
            query: The original user query
            verification_model: The model to use for verification
            max_iterations: Maximum number of verification iterations
            confidence_threshold: Threshold for correction confidence
            
        Returns:
            Dictionary with verified answer and metadata
        """
        logger.info(f"Starting async verification process with {len(context)} context chunks")
        
        # Track all verification iterations
        verification_history = []
        current_answer = answer
        
        for iteration in range(max_iterations):
            logger.info(f"Async verification iteration {iteration+1}/{max_iterations}")
            
            # Run verification step asynchronously
            verification_result = await self._async_run_verification_step(
                current_answer, context, query, verification_model
            )
            
            verification_history.append(verification_result)
            
            # If no issues found, break the loop
            if not verification_result.get("issues_found", False):
                logger.info("No issues found in this iteration")
                break
                
            # Get corrected answer and confidence
            corrected_answer = verification_result.get("corrected_answer", current_answer)
            overall_confidence = verification_result.get("overall_confidence", 0.0)
            
            # Log corrections
            corrections = verification_result.get("corrections", [])
            for correction in corrections:
                logger.info(f"Correction (confidence: {correction.get('confidence', 0.0)}): "
                            f"'{correction.get('original', '')}' -> '{correction.get('corrected', '')}'")
            
            # Update current answer if confidence is sufficient
            if overall_confidence >= confidence_threshold:
                logger.info(f"Applied corrections with confidence {overall_confidence}")
                current_answer = corrected_answer
            else:
                logger.info(f"Skipping corrections due to low confidence: {overall_confidence}")
                break
                
        # Determine final answer
        was_corrected = current_answer != answer
        
        # Prepare result
        result = {
            "original_answer": answer,
            "verified_answer": current_answer,
            "was_corrected": was_corrected,
            "verification_history": verification_history,
            "correction_count": len(verification_history)
        }
        
        # Add note if corrections were made
        if was_corrected:
            result["verified_answer"] += "\n\n(Note: This response has been verified and corrected for factual accuracy.)"
            
        return result
    
    async def _async_run_verification_step(
        self, 
        answer: str, 
        context: List[dict], 
        query: str, 
        model: str
    ) -> Dict[str, Any]:
        """
        Asynchronously run a single verification step.
        
        Args:
            answer: The answer to verify
            context: List of context dictionaries
            query: The original user query
            model: The model to use for verification
            
        Returns:
            Verification result dictionary
        """
        # Prepare context string for verification
        context_for_verification = self._prepare_context_for_verification(context)
        
        # Craft verification prompt
        verification_prompt = self._create_verification_prompt(
            context_for_verification, query, answer
        )
        
        # Try to get verification and correction
        try:
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an expert fact-checker that identifies and corrects factual errors based strictly on the provided context."},
                        {"role": "user", "content": verification_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=60)
            verification_result = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                verification_data = json.loads(verification_result)
                return verification_data
            except json.JSONDecodeError:
                logger.error("Failed to parse verification result as JSON")
                return {"issues_found": False, "error": "JSON parsing error"}
                
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"Error during async verification step: {str(e)}")
            return {"issues_found": False, "error": str(e)} 