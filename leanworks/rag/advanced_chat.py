import logging
import json
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from leanworks.rag.verification import AnswerVerifier
from leanworks.rag.chat import Chat
from leanworks.rag.setting import OTHER_MODEL, RERANK_TOP_K

# Set up logging
logger = logging.getLogger(__name__)

class AdvancedChat:
    """
    Advanced Chat for RAG systems.
    
    Implements a verification chain with 7 steps:
    1. Initial retrieval
    2. Draft answer
    3. Claim decomposition
    4. Plan verification queries
    5. Retrieve & rank evidence
    6. LLM verification
    7. Repair & finalize
    """
    
    def __init__(
        self,
        chat: Chat,
        model_client,
        embedding_model_client=None,
        verification_model: str = OTHER_MODEL,
        max_claims: int = 5,
        verification_confidence_threshold: float = 0.7,
        max_verification_iterations: int = 2
    ):
        """
        Initialize the Chain of Verification system.
        
        Args:
            chat: Initialized Chat instance for RAG
            model_client: The model client for language model operations
            embedding_model_client: Optional embedding model client
            verification_model: The model to use for verification
            max_claims: Maximum number of claims to verify
            verification_confidence_threshold: Confidence threshold for applying corrections
            max_verification_iterations: Maximum number of verification iterations
        """
        self.chat = chat
        self.model_client = model_client
        self.embedding_model_client = embedding_model_client
        self.verification_model = verification_model
        self.max_claims = max_claims
        self.confidence_threshold = verification_confidence_threshold
        self.max_verification_iterations = max_verification_iterations
        
        # Initialize verifier for claim verification
        self.verifier = AnswerVerifier(
            model_client=model_client,
            embedding_model_client=embedding_model_client,
            pinecone_api_key=chat.pinecone_api_key,
            index_host=chat.index_host
        )
        
        logger.info("Chain of Verification initialized")
    
    async def process_query(
        self,
        query: str,
        generation_model: str,
        include_memory: bool = True,
        cited_context: dict = None,
        top_k: int = 10,
        rerank_top_k: int = RERANK_TOP_K
    ) -> Dict[str, Any]:
        """
        Process a query through the Chain of Verification.
        
        Args:
            query: The user query
            generation_model: The model to use for generation
            include_memory: Whether to include memory
            cited_context: Specific context cited by the user
            top_k: Number of initial documents to retrieve
            rerank_top_k: Number of documents to keep after reranking
            
        Returns:
            Dictionary with verified answer and metadata
        """
        logger.info(f"Processing query through Chain of Verification: '{query}'")
        
        # Step 1: Initial Retrieval (using the chat module)
        logger.info("Step 1: Initial retrieval")
        
        # We'll use the chat's internal functions to get context
        context_task = asyncio.create_task(
            self.chat._async_retrieve_and_process_context(query, top_k=top_k, rerank_top_k=rerank_top_k)
        )
        
        # Get memory in parallel if needed
        memory_context = []
        if include_memory and self.chat.memory_enabled:
            loop = asyncio.get_event_loop()
            try:
                memory_context = await asyncio.wait_for(
                    loop.run_in_executor(None, self.chat._retrieve_memory),
                    timeout=30
                )
            except asyncio.TimeoutError:
                logger.error("Memory retrieval timed out after 30 seconds")
        
        # Wait for context with timeout
        try:
            context, data_sources = await asyncio.wait_for(context_task, timeout=60)
        except asyncio.TimeoutError:
            logger.error("Context retrieval timed out after 60 seconds")
            context, data_sources = [], []
        
        # Step 2: Generate Draft Answer
        logger.info("Step 2: Generate draft answer")
        draft_answer = await self._generate_draft_answer(
            query, context, memory_context, generation_model, cited_context
        )
        
        # Early return if we couldn't generate a draft answer
        if not draft_answer:
            logger.warning("Failed to generate draft answer, returning error message")
            return {
                "content": "I apologize, but I couldn't generate a response at this time.",
                "data_sources": data_sources,
                "verification_meta": {
                    "was_corrected": False,
                    "error": "Failed to generate draft answer"
                }
            }
        
        # Step 3: Claim Decomposition
        logger.info("Step 3: Decompose claims from draft answer")
        claims = await self._extract_claims(draft_answer, query)
        
        # If no claims were extracted, return the draft answer
        if not claims:
            logger.info("No verifiable claims extracted, returning draft answer")
            return {
                "content": draft_answer,
                "data_sources": data_sources,
                "verification_meta": {
                    "was_corrected": False,
                    "claims_found": 0
                }
            }
        
        # Limit the number of claims to verify
        if len(claims) > self.max_claims:
            logger.info(f"Limiting claims from {len(claims)} to {self.max_claims}")
            claims = claims[:self.max_claims]
        
        # Step 4: Plan Verification Queries
        logger.info(f"Step 4: Plan verification queries for {len(claims)} claims")
        verification_queries = await self._plan_verification_queries(claims, query)
        
        # Step 5: Retrieve & Rank Evidence (in parallel for all claims)
        logger.info("Step 5: Retrieve and rank evidence for verification")
        verification_evidence = await self._retrieve_evidence_for_claims(
            verification_queries, claims, top_k=5
        )
        
        # Step 6: LLM Verification
        logger.info("Step 6: Verify claims with LLM")
        verification_results = await self._verify_claims(
            claims, verification_evidence, query
        )
        
        # Step 7: Repair & Finalize
        logger.info("Step 7: Repair and finalize answer")
        final_answer, was_corrected = await self._repair_and_finalize(
            draft_answer, verification_results, query, context
        )
        
        # Construct result
        result = {
            "content": final_answer,
            "data_sources": data_sources,
            "verification_meta": {
                "was_corrected": was_corrected,
                "claims_found": len(claims),
                "verification_results": verification_results
            }
        }
        
        # Store in memory if enabled
        if self.chat.memory_enabled:
            # Run memory storage in background
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.chat.add_memory, query, final_answer)
        
        return result
    
    async def _generate_draft_answer(
        self,
        query: str,
        context: List[dict],
        memory_context: List[str],
        model: str,
        cited_context: dict = None
    ) -> str:
        """
        Generate a draft answer using the retrieved context.
        
        Args:
            query: The user query
            context: Retrieved context
            memory_context: Memory context
            model: Generation model
            cited_context: User cited context
            
        Returns:
            Draft answer
        """
        # Format context for the model
        formatted_context = self._format_context(context, memory_context)
        
        # Create a prompt with citation instructions
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        Your answer should be based ONLY on the provided context. If the context doesn't contain 
        the necessary information, acknowledge what you don't know.
        
        When referencing information, cite the document number in [Doc X] format 
        where X is the document number from the context.
        """
        
        user_prompt = f"""Context information:\n{formatted_context}\n\nUser query: {query}"""
        if cited_context:
            user_prompt += f"\nUser cited context: {cited_context}"
        
        user_prompt += "\n\nProvide a clear, concise answer with appropriate citations to the documents."
        
        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=60)
            draft_answer = response.choices[0].message.content
            logger.info(f"Generated draft answer: {draft_answer[:100]}...")
            return draft_answer
            
        except Exception as e:
            logger.error(f"Error generating draft answer: {str(e)}")
            return None
    
    def _format_context(self, context: List[dict], memory_context: List[str]) -> str:
        """Format context for the model with document numbers for citation."""
        formatted_parts = []
        
        # Add memory context if available
        if memory_context:
            formatted_parts.append("Recent conversations:")
            formatted_parts.append("\n".join(memory_context))
            formatted_parts.append("\nRelevant documents:")
        
        # Add document context with numbers for citation
        for i, ctx in enumerate(context):
            doc_num = i + 1
            context_text = ctx.get("context", "")
            source = ctx.get("data_source", "Unknown source")
            formatted_parts.append(f"[Doc {doc_num}] {source}: {context_text}")
        
        return "\n\n".join(formatted_parts)
    
    async def _extract_claims(self, answer: str, query: str) -> List[Dict[str, str]]:
        """
        Extract verifiable claims from the draft answer.
        
        Args:
            answer: Draft answer
            query: Original query
            
        Returns:
            List of claims with metadata
        """
        system_prompt = """You are an expert at identifying factual claims in text.
        Your task is to extract specific, verifiable factual claims from the provided answer.
        Focus on claims about dates, numbers, names, events, and other factual statements."""
        
        user_prompt = f"""Extract all verifiable factual claims from this answer to the query: "{query}"
        
        Answer to analyze:
        {answer}
        
        For each claim:
        1. Extract the exact claim text
        2. Assign a unique ID (C1, C2, etc.)
        3. Rate how important the claim is to the answer (1-5, where 5 is critical)
        
        Format your response as a JSON array with objects containing "id", "claim", and "importance" fields.
        Only include clear factual statements that could be verified with evidence."""
        
        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=self.verification_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=30)
            claims_text = response.choices[0].message.content
            
            # Parse claims from JSON
            try:
                claims_data = json.loads(claims_text)
                claims = claims_data.get("claims", [])
                if not claims and isinstance(claims_data, list):
                    claims = claims_data
                
                logger.info(f"Extracted {len(claims)} claims from draft answer")
                return claims
            except json.JSONDecodeError:
                logger.error(f"Failed to parse claims JSON: {claims_text}")
                # Try to extract claims with regex as fallback
                return self._extract_claims_with_regex(claims_text)
                
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return []
    
    def _extract_claims_with_regex(self, text: str) -> List[Dict[str, str]]:
        """Extract claims using regex as a fallback method."""
        # Look for patterns like "C1: <claim text>" or similar
        claim_pattern = re.compile(r'[C](\d+)[:\)]\s*([^"]+?)(?=\n|$|\d+[:\)])')
        matches = claim_pattern.findall(text)
        
        claims = []
        for i, (claim_id, claim_text) in enumerate(matches):
            claims.append({
                "id": f"C{claim_id}",
                "claim": claim_text.strip(),
                "importance": 3  # Default importance
            })
        
        logger.info(f"Extracted {len(claims)} claims using regex fallback")
        return claims
    
    async def _plan_verification_queries(
        self, 
        claims: List[Dict[str, str]], 
        original_query: str
    ) -> List[Dict[str, str]]:
        """
        Generate focused queries for each claim to verify.
        
        Args:
            claims: List of extracted claims
            original_query: Original user query
            
        Returns:
            List of claims with verification queries
        """
        system_prompt = """You are an expert at formulating precise search queries to verify factual claims.
        Your task is to create focused queries that will retrieve evidence to verify or refute each claim."""
        
        user_prompt = f"""For each of the following claims extracted from an answer to the query "{original_query}",
        create 1-2 focused search queries that would help verify or refute the claim.
        
        The queries should:
        1. Be specific and targeted to verify the exact claim
        2. Focus on retrieving factual evidence (dates, numbers, names, etc.)
        3. Be different from the original query to get diverse evidence
        
        Claims:
        {json.dumps(claims, indent=2)}
        
        Format your response as a JSON object where the keys are claim IDs and the values are arrays of search queries."""
        
        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=self.verification_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=30)
            queries_text = response.choices[0].message.content
            
            # Parse queries from JSON
            try:
                queries_data = json.loads(queries_text)
                
                # Add queries back to the claims
                for claim in claims:
                    claim_id = claim["id"]
                    if claim_id in queries_data:
                        claim["verification_queries"] = queries_data[claim_id]
                    else:
                        # Fallback: create a simple query from the claim
                        claim["verification_queries"] = [claim["claim"]]
                
                logger.info(f"Generated verification queries for {len(claims)} claims")
                return claims
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse verification queries JSON: {queries_text}")
                # Fallback: create simple queries from the claims
                for claim in claims:
                    claim["verification_queries"] = [claim["claim"]]
                return claims
                
        except Exception as e:
            logger.error(f"Error planning verification queries: {str(e)}")
            # Fallback: create simple queries from the claims
            for claim in claims:
                claim["verification_queries"] = [claim["claim"]]
            return claims
    
    async def _retrieve_evidence_for_claims(
        self, 
        claims: List[Dict[str, str]], 
        original_claims: List[Dict[str, str]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for each claim using their verification queries.
        
        Args:
            claims: List of claims with verification queries
            original_claims: Original claims list (may be the same as claims)
            top_k: Number of documents to retrieve per query
            
        Returns:
            Claims with retrieved evidence
        """
        # Create tasks for each claim's queries
        tasks = []
        
        for claim in claims:
            claim_id = claim["id"]
            queries = claim.get("verification_queries", [claim["claim"]])
            
            # For each query, create a retrieval task
            for query in queries:
                task = asyncio.create_task(self._retrieve_evidence_for_query(
                    query, claim_id, top_k
                ))
                tasks.append((claim_id, query, task))
        
        # Wait for all tasks to complete
        evidence_by_claim = {}
        
        for claim_id, query, task in tasks:
            try:
                evidence = await task
                
                # Initialize evidence list for this claim if needed
                if claim_id not in evidence_by_claim:
                    evidence_by_claim[claim_id] = []
                
                # Add evidence from this query
                if evidence:
                    evidence_by_claim[claim_id].extend(evidence)
                    
            except Exception as e:
                logger.error(f"Error retrieving evidence for claim {claim_id}, query '{query}': {str(e)}")
        
        # Add evidence back to the claims
        for claim in claims:
            claim_id = claim["id"]
            claim["evidence"] = evidence_by_claim.get(claim_id, [])
            logger.info(f"Retrieved {len(claim.get('evidence', []))} pieces of evidence for claim {claim_id}")
        
        return claims
    
    async def _retrieve_evidence_for_query(
        self, 
        query: str, 
        claim_id: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for a verification query.
        
        Args:
            query: Verification query
            claim_id: ID of the claim being verified
            top_k: Number of documents to retrieve
            
        Returns:
            List of evidence documents
        """
        try:
            # Use the verifier's mini RAG capability
            # We'll construct an enhanced query combining the claim verification query
            nodes = self.verifier.retrieve_nodes(query, top_k=top_k)
            
            # Extract and format evidence
            evidence = []
            for i, match in enumerate(nodes):
                metadata = match.metadata
                
                # Extract text using fallback methods
                context_text = ""
                if "text" in metadata:
                    context_text = metadata["text"]
                elif "_node_content" in metadata:
                    try:
                        node_content = json.loads(metadata.get("_node_content", ""))
                        context_text = node_content.get("text", "") or metadata.get("context", "")
                    except:
                        context_text = metadata.get("context", "")
                
                if not context_text:
                    continue
                
                # Add evidence with metadata
                evidence.append({
                    "id": f"E{claim_id}-{i+1}",
                    "text": context_text,
                    "source": metadata.get("data_source", "Unknown"),
                    "similarity": match.score,
                    "timestamp": metadata.get("timestamp")
                })
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error retrieving evidence for query '{query}': {str(e)}")
            return []
    
    async def _verify_claims(
        self,
        claims: List[Dict[str, str]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Verify each claim against its retrieved evidence.
        
        Args:
            claims: List of claims with evidence
            query: Original user query
            
        Returns:
            Claims with verification results
        """
        # Create verification tasks for each claim
        tasks = []
        
        for claim in claims:
            claim_id = claim["id"]
            claim_text = claim["claim"]
            evidence = claim.get("evidence", [])
            
            # Skip verification if no evidence
            if not evidence:
                claim["verification"] = {
                    "status": "not_sure",
                    "explanation": "No evidence available to verify this claim.",
                    "confidence": 0.0
                }
                continue
            
            # Create verification task
            task = asyncio.create_task(self._verify_claim(
                claim_id, claim_text, evidence, query
            ))
            tasks.append((claim_id, task))
        
        # Wait for all verification tasks to complete
        for claim_id, task in tasks:
            try:
                verification_result = await task
                
                # Find the claim and add verification result
                for claim in claims:
                    if claim["id"] == claim_id:
                        claim["verification"] = verification_result
                        break
                        
            except Exception as e:
                logger.error(f"Error verifying claim {claim_id}: {str(e)}")
                
                # Set default verification result
                for claim in claims:
                    if claim["id"] == claim_id:
                        claim["verification"] = {
                            "status": "not_sure",
                            "explanation": f"Error during verification: {str(e)}",
                            "confidence": 0.0
                        }
                        break
        
        return claims
    
    async def _verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        evidence: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Verify a claim against its evidence.
        
        Args:
            claim_id: ID of the claim
            claim_text: Text of the claim
            evidence: List of evidence documents
            query: Original user query
            
        Returns:
            Verification result
        """
        # Format evidence for the verification
        formatted_evidence = ""
        for i, doc in enumerate(evidence):
            formatted_evidence += f"Evidence {i+1} [{doc.get('source', 'Unknown source')}]:\n{doc.get('text', '')}\n\n"
        
        system_prompt = """You are an expert fact-checker who carefully evaluates claims based on provided evidence.
        Your task is to determine if a claim is supported, refuted, or uncertain based ONLY on the evidence provided.
        Be objective, thorough, and precise in your analysis."""
        
        user_prompt = f"""Evaluate the following claim based SOLELY on the provided evidence:
        
        ORIGINAL QUERY: {query}
        
        CLAIM: {claim_text}
        
        EVIDENCE:
        {formatted_evidence}
        
        Provide your verdict in JSON format with the following fields:
        - "status": Either "supported", "refuted", or "not_sure"
        - "explanation": Your reasoning for this verdict based on the evidence
        - "confidence": A number between 0.0 and 1.0 representing your confidence
        - "corrected_claim": If refuted, provide a corrected version of the claim (if possible)
        
        Only return the JSON object without any other text."""
        
        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=self.verification_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=30)
            result_text = response.choices[0].message.content
            
            # Parse verification result
            try:
                verification_result = json.loads(result_text)
                logger.info(f"Claim {claim_id} verification result: {verification_result.get('status')} with confidence {verification_result.get('confidence')}")
                return verification_result
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse verification result JSON: {result_text}")
                # Return default result
                return {
                    "status": "not_sure",
                    "explanation": "Error parsing verification result",
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"Error during claim verification: {str(e)}")
            # Return default result
            return {
                "status": "not_sure",
                "explanation": f"Error during verification: {str(e)}",
                "confidence": 0.0
            }
    
    async def _repair_and_finalize(
        self,
        draft_answer: str,
        verification_results: List[Dict[str, Any]],
        query: str,
        context: List[dict]
    ) -> Tuple[str, bool]:
        """
        Repair the draft answer based on verification results and finalize it.
        
        Args:
            draft_answer: Original draft answer
            verification_results: Verification results for claims
            query: Original user query
            context: Original retrieved context
            
        Returns:
            Tuple of (final answer, was_corrected flag)
        """
        # Count verification statuses
        supported = 0
        refuted = 0
        uncertain = 0
        
        for claim in verification_results:
            verification = claim.get("verification", {})
            status = verification.get("status", "not_sure")
            
            if status == "supported":
                supported += 1
            elif status == "refuted":
                refuted += 1
            else:  # not_sure
                uncertain += 1
        
        # If no claims were refuted or uncertain, return the draft answer
        if refuted == 0 and uncertain == 0:
            logger.info("All claims supported, returning draft answer without corrections")
            return draft_answer, False
        
        # Format verification results for the model
        formatted_verification = ""
        for claim in verification_results:
            claim_id = claim.get("id", "")
            claim_text = claim.get("claim", "")
            verification = claim.get("verification", {})
            
            status = verification.get("status", "not_sure")
            explanation = verification.get("explanation", "")
            confidence = verification.get("confidence", 0.0)
            corrected_claim = verification.get("corrected_claim", "")
            
            formatted_verification += f"Claim {claim_id}: {claim_text}\n"
            formatted_verification += f"Status: {status}\n"
            formatted_verification += f"Confidence: {confidence}\n"
            formatted_verification += f"Explanation: {explanation}\n"
            
            if status == "refuted" and corrected_claim:
                formatted_verification += f"Corrected version: {corrected_claim}\n"
            
            formatted_verification += "\n"
        
        system_prompt = """You are a careful AI assistant that ensures factual accuracy in responses.
        Your task is to revise a draft answer based on claim verification results.
        For claims that were refuted or uncertain, you should correct, remove, or add caution notes.
        Preserve the overall structure and style of the original answer while ensuring factual accuracy."""
        
        user_prompt = f"""Revise the following draft answer to the query: "{query}"
        
        Some factual claims in the answer have been verified, and the verification results are provided below.
        
        DRAFT ANSWER:
        {draft_answer}
        
        VERIFICATION RESULTS:
        {formatted_verification}
        
        Please create a revised answer that:
        1. Corrects any refuted claims with their verified versions
        2. Adds caution notes for uncertain claims
        3. Maintains the overall structure and style of the original answer
        4. Keeps all supported claims unchanged
        
        Your revised answer should be factually accurate based on the verification results while being helpful to the user."""
        
        try:
            # Call the model asynchronously
            loop = asyncio.get_event_loop()
            response_future = loop.run_in_executor(
                None,
                lambda: self.model_client.chat.completions.create(
                    model=self.verification_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            )
            
            # Wait for result with timeout
            response = await asyncio.wait_for(response_future, timeout=60)
            final_answer = response.choices[0].message.content
            
            logger.info(f"Generated corrected answer based on verification results")
            return final_answer, True
            
        except Exception as e:
            logger.error(f"Error repairing answer: {str(e)}")
            # Return the draft answer with a note about verification issues
            if refuted > 0:
                note = f"\n\n(Note: Some information in this response may not be accurate.)"
                return draft_answer + note, True
            else:
                return draft_answer, False 