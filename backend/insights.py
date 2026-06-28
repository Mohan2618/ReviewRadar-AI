from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from .config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)

POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "perfect", "awesome", 
    "satisfied", "happy", "fast", "best", "nice", "clear", "smooth", "easy",
    "wonderful", "fantastic", "outstanding", "brilliant", "impressive", "kudos",
    "impressed", "delighted", "superb", "quality", "recommend", "worth"
}

NEG_WORDS = {
    "bad", "poor", "terrible", "awful", "hate", "worst", "slow", "broken", 
    "issue", "problem", "bug", "overheat", "overheating", "drain", "battery", 
    "refund", "defect", "fail", "failure", "crash", "error", "glitch", "complaint",
    "disappointed", "frustrat", "waste", "scam", "fraud", "defective", "useless",
    "horrible", "dreadful", "pathetic", "incompetent", "wrong", "error"
}

STOP = {
    "the", "and", "is", "to", "of", "a", "in", "it", "for", "on", "with", 
    "this", "that", "was", "are", "as", "my", "very", "be", "have", "from",
    "by", "or", "at", "an", "not", "but", "can", "i", "you", "we", "they",
    "he", "she", "what", "which", "who", "when", "where", "why", "how", "me"
}

_model = None
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def simple_sentiment(text: str) -> str:
    """Enhanced sentiment classification"""
    if not text or not isinstance(text, str):
        return "neutral"
    
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def sentiment_breakdown(docs: List[str]) -> Dict:
    """Analyze sentiment distribution"""
    if not docs:
        return {"counts": {}, "percent": {"positive": 0, "negative": 0, "neutral": 0}}
    
    counts = Counter(simple_sentiment(d) for d in docs)
    total = max(1, len(docs))
    
    return {
        "counts": dict(counts),
        "percent": {
            "positive": round(counts.get("positive", 0) * 100 / total, 1),
            "negative": round(counts.get("negative", 0) * 100 / total, 1),
            "neutral": round(counts.get("neutral", 0) * 100 / total, 1),
        }
    }


def top_keywords(docs: List[str], k: int = 8) -> List[str]:
    """Extract top keywords from documents"""
    if not docs:
        return []
    
    freq = Counter()
    for d in docs:
        if not isinstance(d, str):
            continue
        for tok in d.lower().split():
            tok = tok.strip(".,!?()[]\"':-–—")
            if len(tok) >= 3 and tok not in STOP:
                freq[tok] += 1
    
    return [w for w, _ in freq.most_common(k)]


def split_by_sentiment(docs: List[str]) -> Tuple[List[str], List[str]]:
    """Split documents by sentiment"""
    pos, neg = [], []
    for d in docs:
        s = simple_sentiment(d)
        if s == "positive":
            pos.append(d)
        elif s == "negative":
            neg.append(d)
    return pos, neg


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())


def cluster_texts(docs: List[str], threshold: float = settings.CLUSTERING_THRESHOLD) -> List[Dict]:
    if not docs or len(docs) < 2:
        if docs:
            return [{
                "size": len(docs),
                "keywords": top_keywords(docs, k=5),
                "examples": docs[:1]
            }]
        return []

    try:
        model = get_model()
        embs = model.encode(docs)
        clusters = []
        used = set()

        for i in range(len(docs)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(docs)):
                if j in used:
                    continue

                sim = cosine_similarity(embs[i], embs[j])
                if sim >= threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        results = []
        for cluster in clusters:
            texts = [docs[idx] for idx in cluster]
            results.append({
                "size": len(cluster),
                "keywords": top_keywords(texts, k=5),
                "examples": texts[:2]
            })

        results.sort(key=lambda x: x["size"], reverse=True)
        return results[:settings.MAX_CLUSTERS]
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return [{
            "size": len(docs),
            "keywords": top_keywords(docs, k=5),
            "examples": docs[:2]
        }]


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract potential entities (products, features, issues) from text.
    Uses pattern matching for quick extraction.
    """
    if not isinstance(text, str):
        return {"issues": [], "features": [], "products": []}
    
    t = text.lower()
    
    issues = []
    issue_patterns = ["issue", "problem", "bug", "error", "crash", "fail", "break", "slow"]
    for pattern in issue_patterns:
        if pattern in t:
            issues.append(pattern)
    
    features = []
    feature_patterns = ["camera", "battery", "screen", "display", "speed", "performance", "quality"]
    for pattern in feature_patterns:
        if pattern in t:
            features.append(pattern)
    
    return {
        "issues": list(set(issues)),
        "features": list(set(features))
    }


def detailed_analysis(docs: List[str]) -> Dict:
    """
    Perform detailed multi-faceted analysis on documents.
    """
    if not docs:
        return {
            "total_docs": 0,
            "sentiment": {"counts": {}, "percent": {"positive": 0, "negative": 0, "neutral": 0}},
            "keywords": [],
            "entities": {"issues": [], "features": [], "products": []},
            "avg_length": 0
        }
    
    pos_docs, neg_docs = split_by_sentiment(docs)
    
    lengths = [len(d.split()) for d in docs if isinstance(d, str)]
    avg_length = round(sum(lengths) / max(1, len(lengths)), 1)
    all_issues = []
    all_features = []
    for doc in docs:
        entities = extract_entities(doc)
        all_issues.extend(entities["issues"])
        all_features.extend(entities["features"])
    
    return {
        "total_docs": len(docs),
        "sentiment": sentiment_breakdown(docs),
        "keywords": {
            "all": top_keywords(docs, 10),
            "positive": top_keywords(pos_docs, 8) if pos_docs else [],
            "negative": top_keywords(neg_docs, 8) if neg_docs else []
        },
        "entities": {
            "top_issues": list(dict(Counter(all_issues).most_common(5)).keys()),
            "top_features": list(dict(Counter(all_features).most_common(5)).keys())
        },
        "statistics": {
            "avg_review_length": avg_length,
            "positive_count": len(pos_docs),
            "negative_count": len(neg_docs),
            "neutral_count": len(docs) - len(pos_docs) - len(neg_docs)
        }
    }
