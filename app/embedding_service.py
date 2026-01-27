import numpy as np
from typing import List, Dict, Optional
import torch
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import embedding_model_settings

logger = logging.getLogger(__name__)

embedder: Optional[HuggingFaceEmbeddings] = None

def initialize_embedder():
    global embedder
    if embedder is not None:
        logger.info("Embedder already initialized.")
        return
    
    logger.info("Initializing embedder...")
    try:
        embedder = HuggingFaceEmbeddings(
            model_name = embedding_model_settings.model_name,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': embedding_model_settings.batch_size
            }
        )
        logger.info("Embedder initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        raise RuntimeError("Embedder initialization failed") from e
    

def embed_query(query: str) -> np.ndarray:
    if embedder is None:
        raise RuntimeError("Embedder not initialized. Call initialize_embedder() first.")
    logger.info(f"Embedding query: {query}")
    try:
        vector = embedder.embed_query(query)
        embedding = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        logger.info(f"Query embedded successfully. Shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise RuntimeError("Query embedding failed") from e


def embed_documents(documents: List[str]) -> np.ndarray:
    if embedder is None:
        raise RuntimeError("Embedder not initialized. Call initialize_embedder() first.")
    logger.info(f"Embedding {len(documents)} documents.")
    try:
        vectors = embedder.embed_documents(documents)
        embeddings = np.asarray(vectors, dtype=np.float32)
        logger.info(f"Documents embedded successfully. Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to embed documents: {e}")
        raise RuntimeError("Document embedding failed") from e
    

def get_embedding_info() -> Dict:
    if embedder is None:
        raise RuntimeError("Embedder not initialized. Call initialize_embedder() first.")
    info = {
        "model_name": embedding_model_settings.model_name,
        "embedding_dimension": embedding_model_settings.embedding_dimension,
    }
    logger.info(f"Embedding model info: {info}")
    return info




