from collections import Counter, defaultdict
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

# Simple sentiment lexicon (lightweight, no extra deps)
POS_WORDS = {
    "good","great","excellent","amazing","love","perfect","awesome","satisfied","happy","fast","best","nice","clear","smooth"
}
NEG_WORDS = {
    "bad","poor","terrible","awful","hate","worst","slow","broken","issue","problem","bug","overheat","overheating","drain","battery","refund","defect"
}

# Optional embedding model for clustering
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def simple_sentiment(text: str) -> str:
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def sentiment_breakdown(docs: List[str]) -> Dict:
    counts = Counter(simple_sentiment(d) for d in docs)
    total = max(1, len(docs))
    return {
        "counts": counts,
        "percent": {k: round(v * 100 / total, 1) for k, v in counts.items()}
    }


# --- Complaint / Praise keyword extraction (cheap but effective) ---
STOP = {"the","and","is","to","of","a","in","it","for","on","with","this","that","was","are","as","my","very"}

def top_keywords(docs: List[str], k: int = 8):
    freq = Counter()
    for d in docs:
        for tok in d.lower().split():
            tok = tok.strip(".,!?()[]\"'")
            if len(tok) < 3 or tok in STOP:
                continue
            freq[tok] += 1
    return [w for w, _ in freq.most_common(k)]


def split_by_sentiment(docs: List[str]):
    pos, neg = [], []
    for d in docs:
        s = simple_sentiment(d)
        if s == "positive":
            pos.append(d)
        elif s == "negative":
            neg.append(d)
    return pos, neg


# --- Very light clustering using embeddings + cosine threshold ---
def cluster_texts(docs: List[str], threshold: float = 0.75):
    if not docs:
        return []

    model = get_model()
    embs = model.encode(docs)

    clusters = []
    used = set()

    def cosine(a, b):
        import numpy as np
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float((a * b).sum())

    for i in range(len(docs)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(docs)):
            if j in used:
                continue
            if cosine(embs[i], embs[j]) >= threshold:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)

    # represent clusters by top keywords
    results = []
    for c in clusters:
        texts = [docs[i] for i in c]
        results.append({
            "size": len(c),
            "keywords": top_keywords(texts, k=5),
            "examples": texts[:2]
        })
    # sort by size
    results.sort(key=lambda x: x["size"], reverse=True)
    return results[:5]