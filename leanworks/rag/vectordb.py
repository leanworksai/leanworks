from pinecone import Pinecone, ServerlessSpec
from leanworks.rag.embedding import GoogleEmbedding
from leanworks.setting import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
import tiktoken

# Constants
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_VOCAB_SIZE = 30000
UPSERT_BATCH_SIZE = 100
SERVERLESS_SPEC = ServerlessSpec(cloud="gcp", region="us-central1")

class PineconeHybridIndex:
    def __init__(self, pinecone_key: str, embedding_model_client: GoogleEmbedding, 
                 chunk_size: int = 512, chunk_overlap: int = 128):
        self.pc = Pinecone(api_key=pinecone_key)
        self.dense_index = None
        self.sparse_index = None
        self.chunk_size = chunk_size  # tokens
        self.chunk_overlap = chunk_overlap  # tokens
        self.tokenizer = tiktoken.get_encoding("o200k_base")  # GPT-4o tokenizer
        self.embedding_model_client = embedding_model_client

    def create_hybrid_index(self, dense_index_name: str, sparse_index_name: str):
        """Create both dense and sparse indexes for hybrid search."""
        self._create_dense_index(dense_index_name)
        self._create_sparse_index(sparse_index_name)
        return self.dense_index, self.sparse_index

    def load_hybrid_index(self, dense_index_name: str, sparse_index_name: str):
        """Load both dense and sparse indexes for hybrid search."""
        self.dense_index = self.pc.Index(dense_index_name)
        self.sparse_index = self.pc.Index(sparse_index_name)
        return self.dense_index, self.sparse_index
    
    def _create_dense_index(self, index_name: str):
        """Create and connect to dense index."""
        self._delete_index_if_exists(index_name)
        
        self.pc.create_index(
            name=index_name,
            dimension=DEFAULT_EMBEDDING_DIMENSION,
            metric="cosine",
            vector_type="dense",
            spec=SERVERLESS_SPEC,
        )
        logging.info(f"Dense index {index_name} created.")
        
        self.dense_index = self.pc.Index(index_name)
    
    def _create_sparse_index(self, index_name: str):
        """Create and connect to sparse index."""
        self._delete_index_if_exists(index_name)
        
        self.pc.create_index(
            name=index_name,
            metric="dotproduct",
            vector_type="sparse",
            spec=SERVERLESS_SPEC,
        )
        logging.info(f"Sparse index {index_name} created.")
        
        self.sparse_index = self.pc.Index(index_name)
    
    def _delete_index_if_exists(self, index_name: str):
        """Delete index if it exists."""
        try:
            self.pc.delete_index(index_name)
            logging.info(f"Index {index_name} deleted.")
        except Exception:
            logging.info(f"Index {index_name} doesn't exist.")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap based on token count."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
                
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_sparse_embedding(self, text: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Dict[str, Any]:
        """Generate enhanced sparse embedding with proper handling of user names and emails."""
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
            index = hash(term) % vocab_size
            if index not in indices:  # Avoid duplicate indices
                indices.append(index)
                weight = self._calculate_term_weight(term, count, total_tokens)
                values.append(weight)
        
        return {'indices': indices, 'values': values}
    
    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization that preserves important patterns like emails and user IDs."""
        tokens = []
        important_patterns = []
        
        # Define patterns to preserve
        patterns = {
            'email': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            'mention': r'@[a-zA-Z0-9._-]+',
            'structured_id': r'\b[a-zA-Z0-9]+[._-][a-zA-Z0-9._-]+\b',
            'name': r'\b[A-Z][a-z]+\b'
        }
        
        # Extract important patterns
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            
            for match in matches:
                important_patterns.append(match)
                
                if pattern_type == 'email':
                    # Add email parts for partial matching
                    parts = match.split('@')
                    important_patterns.extend(parts)
                    # Extract name parts from email local part
                    name_parts = re.split(r'[._-]', parts[0])
                    important_patterns.extend([part for part in name_parts if len(part) > 1])
                elif pattern_type == 'mention':
                    # Add without @ symbol
                    important_patterns.append(match[1:])
                elif pattern_type == 'structured_id':
                    # Add individual parts
                    parts = re.split(r'[._-]', match)
                    important_patterns.extend([part for part in parts if len(part) > 1])
        
        tokens.extend(important_patterns)
        
        # Regular tokenization for remaining text
        remaining_text = text
        for pattern in important_patterns:
            remaining_text = remaining_text.replace(pattern.lower(), ' ')
        
        regular_words = re.findall(r'\b\w+\b', remaining_text)
        tokens.extend([word for word in regular_words if len(word) > 1])
        
        return tokens
    
    def _calculate_term_weight(self, term: str, count: int, total_tokens: int) -> float:
        """Calculate enhanced weight for terms based on their importance."""
        base_tf = count / total_tokens if total_tokens > 0 else 0
        
        # Identity term boosting based on patterns
        if '@' in term and '.' in term:
            identity_boost = 5.0  # Email addresses
        elif term.startswith('@'):
            identity_boost = 4.0  # User mentions
        elif any(char in term for char in ['.', '_', '-']) and len(term) > 3:
            identity_boost = 3.0  # Structured identifiers
        elif term and term[0].isupper():
            identity_boost = 2.5  # Capitalized words (likely names)
        elif any(indicator in term.lower() for indicator in ['name', 'user', 'id', 'email']):
            identity_boost = 2.0  # Common name patterns
        else:
            identity_boost = 1.0
        
        # Length-based adjustment (longer terms are typically more specific)
        length_boost = min(2.0, 1.0 + (len(term) - 3) * 0.1) if len(term) > 3 else 1.0
        
        # Final weight calculation with cap
        final_weight = base_tf * identity_boost * length_boost
        return min(final_weight, 1.0)
    
    def upsert_documents_hybrid(self, documents, retries: int = 3, delay: int = 2):
        """Upsert documents to both dense and sparse indexes with improved batching."""
        documents = [doc for doc in documents if doc is not None]
        
        if not documents:
            logging.warning("No documents to upsert")
            return None
        
        for attempt in range(retries):
            try:
                # Prepare chunks and metadata
                all_chunks, chunk_metadata_list = self._prepare_chunks(documents)
                
                # Generate embeddings and create vectors
                dense_vectors, sparse_vectors = self._create_vectors(all_chunks, chunk_metadata_list)
                
                # Upsert to both indexes
                self._upsert_vectors(self.dense_index, dense_vectors, "dense")
                self._upsert_vectors(self.sparse_index, sparse_vectors, "sparse", is_sparse=True)
                
                logging.info(f"Successfully upserted {len(dense_vectors)} document chunks to both indexes")
                return self.dense_index, self.sparse_index
                
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                    logging.error(e)
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached, operation failed.")
                    raise e
    
    def _prepare_chunks(self, documents) -> tuple[List[str], List[Dict]]:
        """Prepare chunks and metadata from documents."""
        all_chunks = []
        chunk_metadata_list = []
        
        for doc in documents:
            # Get document content and metadata
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            else:
                content = str(doc)
                metadata = {}
            
            document_id = metadata.get('id', str(uuid.uuid4()))
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_number': i,
                    'chunk_text': chunk,
                    'document_id': document_id,
                    'chunk_id': chunk_id
                })
                
                all_chunks.append(chunk)
                chunk_metadata_list.append(chunk_metadata)
        
        logging.info(f"Processing {len(all_chunks)} chunks for embedding generation")
        return all_chunks, chunk_metadata_list
    
    def _create_vectors(self, all_chunks: List[str], chunk_metadata_list: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Create dense and sparse vectors from chunks."""
        logging.info("Generating dense embeddings for all chunks...")
        dense_embeddings = self.embedding_model_client.get_embeddings_batch(
            all_chunks, 
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        dense_vectors = []
        sparse_vectors = []
        
        for chunk, chunk_metadata, dense_embedding in zip(all_chunks, chunk_metadata_list, dense_embeddings):
            chunk_id = chunk_metadata['chunk_id']
            final_metadata = {k: v for k, v in chunk_metadata.items() if k != 'chunk_id'}
            
            # Convert numpy array to list for Pinecone serialization
            if hasattr(dense_embedding, 'tolist'):
                dense_embedding = dense_embedding.tolist()
            
            # Create dense vector
            dense_vectors.append({
                'id': chunk_id,
                'values': dense_embedding,
                'metadata': final_metadata
            })
            
            # Create sparse vector
            sparse_embedding = self._get_sparse_embedding(chunk)
            sparse_vectors.append({
                'id': chunk_id,
                'values': sparse_embedding['values'],
                'sparse_values': {
                    'indices': sparse_embedding['indices'],
                    'values': sparse_embedding['values']
                },
                'metadata': final_metadata
            })
        
        return dense_vectors, sparse_vectors
    
    def _upsert_vectors(self, index, vectors: List[Dict], index_type: str, is_sparse: bool = False):
        """Upsert vectors to a specific index."""
        total_vectors = len(vectors)
        
        for i in range(0, total_vectors, UPSERT_BATCH_SIZE):
            batch = vectors[i:i + UPSERT_BATCH_SIZE]
            
            if is_sparse:
                # Transform sparse vectors for upsert
                sparse_batch = []
                for vector in batch:
                    sparse_vector = {
                        'id': vector['id'],
                        'sparse_values': vector['sparse_values'],
                        'metadata': vector['metadata']
                    }
                    sparse_batch.append(sparse_vector)
                index.upsert(vectors=sparse_batch)
            else:
                index.upsert(vectors=batch)
            
            batch_num = i // UPSERT_BATCH_SIZE + 1
            total_batches = (total_vectors + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
            logging.info(f"Upserted {index_type} batch {batch_num}/{total_batches}")
    
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
        if hasattr(dense_query_embedding, 'tolist'):
            dense_query_embedding = dense_query_embedding.tolist()
        
        sparse_query_embedding = self._get_sparse_embedding(query)
        
        # Prepare query parameters
        query_params = {
            'top_k': top_k * 2,  # Get more results to merge
            'include_metadata': True,
            'namespace': namespace
        }
        
        if filter is not None:
            query_params['filter'] = filter
        
        # Search both indexes
        dense_results = self.dense_index.query(vector=dense_query_embedding, **query_params)
        sparse_results = self.sparse_index.query(sparse_vector=sparse_query_embedding, **query_params)
        
        # Merge and return top results
        merged_results = self._merge_results(dense_results['matches'], sparse_results['matches'], alpha)
        return merged_results[:top_k]
    
    def _merge_results(
        self, 
        dense_results: List[Dict], 
        sparse_results: List[Dict], 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from dense and sparse searches."""
        # Normalize scores to 0-1 range
        dense_scores = [match['score'] for match in dense_results]
        sparse_scores = [match['score'] for match in sparse_results]
        
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_min = min(dense_scores) if dense_scores else 0.0
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        sparse_min = min(sparse_scores) if sparse_scores else 0.0
        
        combined_results = {}
        
        # Process dense results
        for match in dense_results:
            normalized_score = self._normalize_score(match['score'], dense_min, dense_max)
            combined_results[match['id']] = {
                'id': match['id'],
                'metadata': match['metadata'],
                'dense_score': normalized_score,
                'sparse_score': 0.0,
                'combined_score': alpha * normalized_score
            }
        
        # Process sparse results and combine scores
        for match in sparse_results:
            normalized_score = self._normalize_score(match['score'], sparse_min, sparse_max)
            
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
        
        # Sort by combined score and return
        return sorted(combined_results.values(), key=lambda x: x['combined_score'], reverse=True)
    
    def _normalize_score(self, score: float, min_score: float, max_score: float) -> float:
        """Normalize a score to 0-1 range."""
        if max_score == min_score:
            return 0.5
        return (score - min_score) / (max_score - min_score)