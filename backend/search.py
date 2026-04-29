import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, DEFAULT_TOP_K

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ❗ IMPORTANT: NO caching of collection
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, dataset: str = None):
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query])[0].tolist()

    where_filter = {"dataset": dataset} if dataset else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        similarity = round(1 - dist, 4)

        output.append({
            "rank": i + 1,
            "review_text": doc,
            "similarity_score": similarity,
            "dataset": meta.get("dataset", "N/A"),
            "product_name": meta.get("product_name", "N/A"),
            "rating": meta.get("rating", "N/A"),
            "date": meta.get("date", "N/A"),
            "reviewer_name": meta.get("reviewer_name", "N/A"),
            "category": meta.get("category", "N/A"),
        })

    return output