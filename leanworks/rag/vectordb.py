from pinecone import Pinecone, ServerlessSpec
from .embedding import GoogleEmbedding
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
import tiktoken

class PineconeHybridIndex:
    def __init__(self, pinecone_key, embedding_model_client, chunk_size=512, chunk_overlap=128):
        self.pc = Pinecone(api_key=pinecone_key)
        self.dense_index = None
        self.sparse_index = None
        self.chunk_size = chunk_size  # tokens
        self.chunk_overlap = chunk_overlap  # tokens
        self.tokenizer = tiktoken.get_encoding("o200k_base")  # GPT-4o tokenizer
        self.embedding_model_client = embedding_model_client
    def create_hybrid_index(
            self, 
            dense_index_name: str,
            sparse_index_name: str
            ):
        """Load both dense and sparse indexes for hybrid search."""
        
        # Create dense index
        self._create_dense_index(dense_index_name)
        
        # Create sparse index
        self._create_sparse_index(sparse_index_name)
        
        # Return the indexes
        return self.dense_index, self.sparse_index

    def load_hybrid_index(
            self, 
            dense_index_name: str,
            sparse_index_name: str
            ):
        """Load both dense and sparse indexes for hybrid search."""
        
        # Connect to the dense index
        self.dense_index = self.pc.Index(dense_index_name)
        
        # Connect to the sparse index
        self.sparse_index = self.pc.Index(sparse_index_name)
        
        # Return the indexes
        return self.dense_index, self.sparse_index
    
    def _create_dense_index(self, index_name: str):
        """Create and connect to dense index."""
        try:
            self.pc.delete_index(index_name)
            logging.info(f"Dense index {index_name} deleted.")
        except Exception:
            logging.info(f"Dense index {index_name} doesn't exist.")
        
        # Create new dense index
        self.pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            vector_type="dense",
            spec=ServerlessSpec(cloud="gcp", region="us-central1"),
        )
        logging.info(f"Dense index {index_name} created.")
        
        # Connect to the dense index
        self.dense_index = self.pc.Index(index_name)
    
    def _create_sparse_index(self, index_name: str):
        """Create and connect to sparse index."""
        try:
            self.pc.delete_index(index_name)
            logging.info(f"Sparse index {index_name} deleted.")
        except Exception:
            logging.info(f"Sparse index {index_name} doesn't exist.")
        
        # Create new sparse index (dimension not specified for sparse indexes)
        self.pc.create_index(
            name=index_name,
            metric="dotproduct",
            vector_type="sparse",
            spec=ServerlessSpec(cloud="gcp", region="us-central1"),
        )
        logging.info(f"Sparse index {index_name} created.")
        
        # Connect to the sparse index
        self.sparse_index = self.pc.Index(index_name)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap based on token count."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            if end > len(tokens):
                end = len(tokens)
            
            # Extract token chunk and decode back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
                
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_sparse_embedding(self, text: str, vocab_size: int = 30000) -> Dict[str, Any]:
        """Generate enhanced sparse embedding with proper handling of user names and emails."""
        # Enhanced tokenization that preserves important patterns
        tokens = self._enhanced_tokenize(text.lower())
        
        # Count term frequencies
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        
        # Create sparse vector with enhanced weighting
        indices = []
        values = []
        total_tokens = len(tokens)
        
        for term, count in term_counts.items():
            # Use hash to create consistent index for each term
            index = hash(term) % vocab_size
            if index not in indices:  # Avoid duplicate indices
                indices.append(index)
                
                # Enhanced weighting based on term importance
                weight = self._calculate_term_weight(term, count, total_tokens)
                values.append(weight)
        
        return {
            'indices': indices,
            'values': values
        }
    
    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization that preserves important patterns like emails and user IDs."""
        tokens = []
        
        # First, extract and preserve important patterns
        important_patterns = []
        
        # Email addresses (preserve as single tokens)
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            important_patterns.append(email)
            # Also add email parts for partial matching
            local_part = email.split('@')[0]
            domain_part = email.split('@')[1]
            important_patterns.extend([local_part, domain_part])
            # Extract name parts from email local part
            name_parts = re.split(r'[._-]', local_part)
            important_patterns.extend([part for part in name_parts if len(part) > 1])
        
        # User mentions (@username)
        mention_pattern = r'@[a-zA-Z0-9._-]+'
        mentions = re.findall(mention_pattern, text)
        for mention in mentions:
            important_patterns.append(mention)
            # Also add without @ symbol
            important_patterns.append(mention[1:])
        
        # User IDs and structured identifiers (word.word, word_word, word-word)
        structured_id_pattern = r'\b[a-zA-Z0-9]+[._-][a-zA-Z0-9._-]+\b'
        structured_ids = re.findall(structured_id_pattern, text)
        for sid in structured_ids:
            important_patterns.append(sid)
            # Also add individual parts
            parts = re.split(r'[._-]', sid)
            important_patterns.extend([part for part in parts if len(part) > 1])
        
        # Names with capital letters (likely proper nouns/names)
        name_pattern = r'\b[A-Z][a-z]+\b'
        names = re.findall(name_pattern, text)
        important_patterns.extend(names)
        
        # Add all important patterns to tokens
        tokens.extend(important_patterns)
        
        # Regular tokenization for remaining text
        # Remove already processed patterns to avoid duplication
        remaining_text = text
        for pattern in important_patterns:
            remaining_text = remaining_text.replace(pattern.lower(), ' ')
        
        # Extract regular words
        regular_words = re.findall(r'\b\w+\b', remaining_text)
        tokens.extend([word for word in regular_words if len(word) > 1])
        
        return tokens
    
    def _calculate_term_weight(self, term: str, count: int, total_tokens: int) -> float:
        """Calculate enhanced weight for terms based on their importance."""
        base_tf = count / total_tokens if total_tokens > 0 else 0
        
        # Identity term boosting
        identity_boost = 1.0
        
        # Email addresses get highest boost
        if '@' in term and '.' in term:
            identity_boost = 5.0
        # User mentions
        elif term.startswith('@'):
            identity_boost = 4.0
        # Structured identifiers (likely user IDs)
        elif any(char in term for char in ['.', '_', '-']) and len(term) > 3:
            identity_boost = 3.0
        # Capitalized words (likely names)
        elif term[0].isupper() if term else False:
            identity_boost = 2.5
        # Common name patterns
        elif any(name_indicator in term.lower() for name_indicator in ['name', 'user', 'id', 'email']):
            identity_boost = 2.0
        
        # Apply length-based adjustment (longer terms are typically more specific)
        length_boost = min(2.0, 1.0 + (len(term) - 3) * 0.1) if len(term) > 3 else 1.0
        
        # Final weight calculation
        final_weight = base_tf * identity_boost * length_boost
        
        # Cap the maximum weight to prevent any single term from dominating
        return min(final_weight, 1.0)
    
    def upsert_documents_hybrid(self, documents, retries=3, delay=2):
        """Upsert documents to both dense and sparse indexes."""
        documents = [doc for doc in documents if doc is not None]
        
        if not documents:
            logging.warning("No documents to upsert")
            return None
        
        attempt = 0
        while attempt < retries:
            try:
                dense_vectors = []
                sparse_vectors = []
                
                for doc in documents:
                    # Get document content
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    else:
                        content = str(doc)
                        metadata = {}
                    document_id = metadata.get('id', str(uuid.uuid4()))
                    # Split document into chunks
                    chunks = self._chunk_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        # Create unique ID for this chunk (same for both indexes)
                        chunk_id = f"{document_id}_chunk_{i}"
                        
                        # Prepare metadata for this chunk
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            'chunk_number': i,
                            'chunk_text': chunk,
                            'document_id': document_id
                        })
                        
                        # Generate dense embedding
                        dense_embedding = self.embedding_model_client.get_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
                        # Convert numpy array to list for Pinecone serialization
                        if hasattr(dense_embedding, 'tolist'):
                            dense_embedding = dense_embedding.tolist()
                        dense_vectors.append({
                            'id': chunk_id,
                            'values': dense_embedding,
                            'metadata': chunk_metadata
                        })
                        
                        # Generate sparse embedding
                        sparse_embedding = self._get_sparse_embedding(chunk)
                        sparse_vectors.append({
                            'id': chunk_id,
                            'values': sparse_embedding['values'],
                            'sparse_values': {
                                'indices': sparse_embedding['indices'],
                                'values': sparse_embedding['values']
                            },
                            'metadata': chunk_metadata
                        })
                
                # Upsert to dense index
                self._upsert_to_index(self.dense_index, dense_vectors, "dense")
                
                # Upsert to sparse index
                self._upsert_to_sparse_index(self.sparse_index, sparse_vectors, "sparse")
                
                logging.info(f"Successfully upserted {len(dense_vectors)} document chunks to both indexes")
                return self.dense_index, self.sparse_index
                
            except Exception as e:
                attempt += 1
                if attempt < retries:
                    logging.warning(f"Attempt {attempt} failed, retrying in {delay} seconds...")
                    logging.error(e)
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached, operation failed.")
                    raise e
    
    def _upsert_to_index(self, index, vectors, index_type):
        """Upsert vectors to a specific index."""
        batch_size = 100
        total_vectors = len(vectors)
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            logging.info(f"Upserted {index_type} batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
    
    def _upsert_to_sparse_index(self, index, vectors, index_type):
        """Upsert sparse vectors to sparse index."""
        batch_size = 100
        total_vectors = len(vectors)
        
        for i in range(0, total_vectors, batch_size):
            batch = []
            for vector in vectors[i:i + batch_size]:
                sparse_vector = {
                    'id': vector['id'],
                    'sparse_values': vector['sparse_values'],
                    'metadata': vector['metadata']
                }
                batch.append(sparse_vector)
            
            index.upsert(vectors=batch)
            logging.info(f"Upserted {index_type} batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10, 
        alpha: float = 0.7,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse results.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0.0 = full sparse, 1.0 = full dense)
            namespace: Namespace to search in
            filter: Pinecone filter dictionary for metadata-based filtering
            
        Returns:
            List of search results with combined scores
        """
        # Get embeddings for query
        dense_query_embedding = self.embedding_model_client.get_embedding(query, task_type="RETRIEVAL_QUERY")
        # Convert numpy array to list for Pinecone serialization
        if hasattr(dense_query_embedding, 'tolist'):
            dense_query_embedding = dense_query_embedding.tolist()
        sparse_query_embedding = self._get_sparse_embedding(query)
        
        # Prepare query parameters
        query_params = {
            'top_k': top_k * 2,  # Get more results to merge
            'include_metadata': True,
            'namespace': namespace
        }
        
        # Add filter if provided
        if filter is not None:
            query_params['filter'] = filter
        
        # Search dense index
        dense_results = self.dense_index.query(
            vector=dense_query_embedding,
            **query_params
        )
        
        # Search sparse index
        sparse_results = self.sparse_index.query(
            sparse_vector=sparse_query_embedding,
            **query_params
        )
        
        # Merge and deduplicate results
        merged_results = self._merge_results(
            dense_results['matches'], 
            sparse_results['matches'], 
            alpha=alpha
        )
        
        # Return top results
        return merged_results[:top_k]
    
    def _merge_results(
        self, 
        dense_results: List[Dict], 
        sparse_results: List[Dict], 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from dense and sparse searches.
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search  
            alpha: Weight for combining scores
            
        Returns:
            Merged and sorted results list
        """
        # Normalize scores to 0-1 range
        dense_scores = [match['score'] for match in dense_results]
        sparse_scores = [match['score'] for match in sparse_results]
        
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_min = min(dense_scores) if dense_scores else 0.0
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        sparse_min = min(sparse_scores) if sparse_scores else 0.0
        
        # Create combined results dictionary
        combined_results = {}
        
        # Add dense results
        for match in dense_results:
            normalized_score = (match['score'] - dense_min) / (dense_max - dense_min) if dense_max != dense_min else 0.5
            combined_results[match['id']] = {
                'id': match['id'],
                'metadata': match['metadata'],
                'dense_score': normalized_score,
                'sparse_score': 0.0,
                'combined_score': alpha * normalized_score
            }
        
        # Add sparse results and combine scores
        for match in sparse_results:
            normalized_score = (match['score'] - sparse_min) / (sparse_max - sparse_min) if sparse_max != sparse_min else 0.5
            
            if match['id'] in combined_results:
                # Update existing result
                combined_results[match['id']]['sparse_score'] = normalized_score
                combined_results[match['id']]['combined_score'] = (
                    alpha * combined_results[match['id']]['dense_score'] + 
                    (1 - alpha) * normalized_score
                )
            else:
                # Add new result
                combined_results[match['id']] = {
                    'id': match['id'],
                    'metadata': match['metadata'],
                    'dense_score': 0.0,
                    'sparse_score': normalized_score,
                    'combined_score': (1 - alpha) * normalized_score
                }
        
        # Sort by combined score and return as list
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return sorted_results