import faiss
import numpy as np
import json
from typing import List, Dict, Optional, Set, Tuple
from app.config import vector_db_settings, reranker_settings
import logging
from sentence_transformers import CrossEncoder
import os


logger = logging.getLogger(__name__)
logger.info("This log message is stored in the file.")

faiss_index: Optional[faiss.IndexIVF] = None
embeddings: Optional[np.ndarray] = None
chunks_metadata: Optional[List[Dict]] = None
reranker_model: Optional[CrossEncoder] = None


def load_vector_database():
    global faiss_index, embeddings, chunks_metadata

    faiss_index_path = vector_db_settings.faiss_index_path
    embeddings_path = vector_db_settings.embeddings_path
    chunks_path = vector_db_settings.chunks_path

    if faiss_index is not None and embeddings is not None and chunks_metadata is not None:
        logger.info("Vector database already loaded.")
        return
    
    try:
        faiss_index = faiss.read_index(faiss_index_path)
        logger.info(f"Loaded FAISS index from {faiss_index_path}.")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise FileNotFoundError(f"Could not load FAISS index from {faiss_index_path}")

    try:
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings from {embeddings_path}.")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise FileNotFoundError(f"Could not load embeddings from {embeddings_path}")
    
    try:
        with open(chunks_path, 'r', encoding="utf-8") as f:
            chunks_metadata = json.load(f)
        logger.info(f"Loaded chunks metadata from {chunks_path}.")
    except Exception as e:
        logger.error(f"Error loading chunks metadata: {e}")
        raise FileNotFoundError(f"Could not load chunks metadata from {chunks_path}")
    
    logger.info("Vector database loaded successfully.")


def initialize_reranker() -> None:

    global reranker_model 

    if reranker_model is not None:
        logger.info("ReRanker already initialized")
        return

    if not reranker_settings.enable_reranker:
        logger.info("Reranking disabled")
        return
    

    try:
        reranker_model = CrossEncoder(
          reranker_settings.reranker_model,
          max_length = 512  
        )
        logger.info("reranker model loaded")
    except Exception as e:
        logger.error(f"Failed to load reranker {e}")
        raise RuntimeError(f"Reranker loading failed: {e}")
    

def search_similar_vectors(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
    if faiss_index is None or embeddings is None or chunks_metadata is None:
        raise ValueError("Vector database is not loaded. Call load_vector_database() first.")
    
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    if(query_embedding.shape[1] != embeddings.shape[1]):
        raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {embeddings.shape[1]}.")
    
    if query_embedding.dtype != np.float32:
        query_embedding = np.array([query_embedding]).astype('float32')

    top_k = min(top_k, faiss_index.ntotal)

    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks_metadata):
            chunk_info = chunks_metadata[idx]
            chunk_info['distance'] = float(dist)
            results.append(chunk_info)
    
    logger.info(f"Found {len(results)} similar vectors.")
    return results
   


def multi_query_retrieve(
        query_embeddings: List[np.ndarray],
        k_per_query: int = reranker_settings.default_k_per_query,
        max_total: int = reranker_settings.max_candidates
) -> List[Dict]:
    
    if faiss_index is None or chunks_metadata is None:
        raise RuntimeError("Vector store not initialized")
    
    seen: Set[int] = set()
    merged: List[Tuple[int, float]] = []

    logger.debug(f"Multi-Query: {len(query_embeddings)} queries, k={k_per_query}")

    for q_emb in query_embeddings:

        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype(np.float32)

        k = min(k_per_query, faiss_index.ntotal)
        D, I = faiss_index.search(q_emb, k)

        for dist, idx in zip(D[0], I[0]):
            idx = int(idx)

            if 0 <= idx < len(chunks_metadata) and idx not in seen:
                seen.add(idx)
                merged.append((idx, float(dist)))

    merged.sort(key=lambda x: x[1])

    selected = []

    for idx, score in merged[: max_total]:
        item = dict(chunks_metadata[idx])
        item['score'] = score
        item['idx'] = idx
        selected.append(item)

    logger.debug(f"Multi-query retrived {len(selected)} unique chunks")
    return selected


def rerank_docs(question: str, docs: List[Dict], top_k: int = reranker_settings.default_k_per_query) -> List[Dict]:

    global reranker_model

    if reranker_model is None:
        logger.warning("Reranker is not yet initialized")
        return docs[:top_k]
    
    if not docs:
        return []
    
    chunk_text = []
    for d in docs:
        text = d.get('text', '')
        if not text:
            text = d.get('page_content', '')
        if not text:
            text = d.get('metadata', {}).get('heading', '')

        chunk_text.append(text)

    pairs = [(question, text) for text in chunk_text]

    try:
        scores = reranker_model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return docs[:top_k]
    
   
    scored_docs = []
    for doc, score in zip(docs, scores):
        doc_copy = dict(doc)
        doc_copy['rerank_score'] = float(score)
        scored_docs.append(doc_copy)
    
  
    scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
    
  
    logger.debug(
        f"Re-ranked {len(docs)} docs → top {min(top_k, len(scored_docs))}\n"
        f"1. {scored_docs[0]['metadata']['heading'][:50]}... score {scored_docs[0]['rerank_score']:.3f}"
    )
    
    return scored_docs[:top_k]

def retrieve_with_reranking(
    question: str,
    query_embeddings: List[np.ndarray],
    k_per_query: int = 3,
    max_retrieved: int = 10,
    rerank_top_k: int = 5
) -> Tuple[List[Dict], Dict]:
    
    import time
    
    timings = {}
    
    
    logger.debug("STEP 1: Multi-Query Retrieval...")
    t0 = time.time()
    docs = multi_query_retrieve(
        query_embeddings=query_embeddings,
        k_per_query=k_per_query,
        max_total=max_retrieved
    )

    print("Multiquery worked")

    timings['retrieval'] = time.time() - t0
    
    logger.debug(f"Retrieved {len(docs)} candidates in {timings['retrieval']:.3f}s")
    
    if not docs:
        return [], timings
    
   
    logger.debug("STEP 2: Re-Ranking with Cross-Encoder...")
    t1 = time.time()
    docs_reranked = rerank_docs(
        question=question,
        docs=docs,
        top_k=rerank_top_k
    )
    timings['reranking'] = time.time() - t1
    
    logger.debug(f"Re-ranking completed in {timings['reranking']:.3f}s")
    
    return docs_reranked, timings

def format_chunk_for_response(chunk: Dict) -> Dict:
    heading = chunk.get('heading', 'No Heading')
    source = chunk.get('source', 'No Source')
    score = chunk.get('score', 'N/A')
    distance = chunk.get('distance', 'N/A')
    chunks_metadata = chunk.get('chunks_metadata', 'N/A')
    formatted_chunk = {
        "heading": heading,
        "source": source,
        "score": score,
        "distance": distance,
        "metadata": chunks_metadata
    }
    return formatted_chunk

def render_context_string(chunks: List[Dict], max_chars: int = 1000) -> str:
    context_strings = []
    for chunk in chunks:
        heading = chunk.get('heading', 'No Heading')
        source = chunk.get('source', 'No Source')
        score = chunk.get('score', 'N/A')
        distance = chunk.get('distance', 'N/A')
        chunk_str = f"Heading: {heading}\nSource: {source}\nScore: {score}\nDistance: {distance}\n"
        context_strings.append(chunk_str)
    return "\n---\n".join(context_strings)


def get_vector_database_status() -> Dict[str, bool]:
    return {
        "faiss_index_loaded": faiss_index is not None,
        "embeddings_loaded": embeddings is not None,
        "chunks_metadata_loaded": chunks_metadata is not None
    }

