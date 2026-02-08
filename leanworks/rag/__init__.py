"""RAG (Retrieval Augmented Generation) functionality for LeanWorks."""

from leanworks.rag.filters import FilterExtractor
from leanworks.rag.reranker import BaseReranker, CrossEncoderReranker, RerankerFactory, get_bge_reranker
from leanworks.rag.embedding import GoogleEmbedding
from leanworks.rag.vectordb_gcp import GCPVectorSearchIndex
from leanworks.rag.vectordb_client import create_vectordb_client
from leanworks.rag.query import QueryRewriter
from leanworks.setting import *

# Make BGEReranker available through a function to avoid onnxruntime dependency
def get_BGEReranker():
    """Get BGEReranker class, importing it only when needed."""
    return get_bge_reranker()
