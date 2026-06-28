import chromadb
from sentence_transformers import SentenceTransformer
from .config import settings
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def get_collection():
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        return client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Failed to get collection: {e}")
        raise ValueError(f"Database error: {str(e)}")


def semantic_search(
    query: str, 
    top_k: int = settings.DEFAULT_TOP_K, 
    dataset: Optional[str] = None
) -> List[Dict]:
    """
    Semantic search with flexible metadata support.
    
    Args:
        query: Search query
        top_k: Number of results
        dataset: Optional dataset filter
        
    Returns:
        List of results with all metadata preserved
    """
    top_k = min(max(1, top_k), settings.MAX_TOP_K)
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        model = get_model()
        collection = get_collection()
        query_embedding = model.encode([query.strip()])[0].tolist()
        where_filter = {"dataset": dataset} if dataset else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = round(1 - dist, 4)
            result = {
                "rank": rank,
                "review_text": doc,
                "similarity_score": similarity,
            }
            if meta:
                for key, value in meta.items():
                    if key != "review_text":
                        result[key] = value
            result.setdefault("dataset", "N/A")
            result.setdefault("product_name", "N/A")
            result.setdefault("rating", "N/A")
            result.setdefault("date", "N/A")
            output.append(result)
        
        logger.info(f"Search returned {len(output)} results")
        return output
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise ValueError(f"Search failed: {str(e)}")


def get_all_datasets() -> List[str]:
    """Get all unique datasets in the collection"""
    try:
        collection = get_collection()
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        
        datasets = list(set(
            m.get("dataset", "unknown")
            for m in metadatas
            if isinstance(m, dict)
        ))
        
        return sorted(datasets)
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}")
        return []


def get_metadata_fields(dataset: Optional[str] = None) -> List[str]:
    """Get all available metadata fields in collection"""
    try:
        collection = get_collection()
        
        where_filter = {"dataset": dataset} if dataset else None
        data = collection.get(include=["metadatas"], where=where_filter)
        metadatas = data.get("metadatas") or []
        
        all_fields = set()
        for meta in metadatas:
            if isinstance(meta, dict):
                all_fields.update(meta.keys())
        
        return sorted(list(all_fields))
    except Exception as e:
        logger.error(f"Failed to get metadata fields: {e}")
        return ["dataset", "product_name", "rating", "date", "review_text"]

