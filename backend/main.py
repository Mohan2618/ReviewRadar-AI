from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import tempfile

from .search import semantic_search, get_collection
from .ingest import ingest_reviews
from .config import DEFAULT_TOP_K
from .insights import sentiment_breakdown, top_keywords, split_by_sentiment, cluster_texts


app = FastAPI(
    title="ReviewRadar AI",
    description="Semantic Customer Review Search Engine",
    version="1.0.0"
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Search ───────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K
    dataset: Optional[str] = None


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = semantic_search(
            req.query.strip(),
            req.top_k,
            req.dataset
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/insights")
def insights(req: SearchRequest):
    try:
        results = semantic_search(req.query.strip(), req.top_k or 50, req.dataset)
        docs = [r["review_text"] for r in results]

        sent = sentiment_breakdown(docs)
        pos_docs, neg_docs = split_by_sentiment(docs)

        return {
            "sentiment": sent,
            "top_keywords_all": top_keywords(docs, 10),
            "top_positive_keywords": top_keywords(pos_docs, 8),
            "top_negative_keywords": top_keywords(neg_docs, 8),
            "clusters": cluster_texts(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/compare")
def compare(req: SearchRequest):
    try:
        # get broader pool for stable stats
        results = semantic_search(req.query.strip(), 100, None)

        by_dataset = {}
        for r in results:
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

        summary = {}
        for ds, rows in by_dataset.items():
            docs = [r["review_text"] for r in rows]
            sent = sentiment_breakdown(docs)

            avg_sim = sum(r["similarity_score"] for r in rows) / max(1, len(rows))

            summary[ds] = {
                "count": len(rows),
                "avg_similarity": round(avg_sim, 3),
                "sentiment_percent": sent["percent"],
                "top_issues": top_keywords(
                    [d for d in docs if "negative" in d.lower()], 5
                ),
                "top_praise": top_keywords(
                    [d for d in docs if "positive" in d.lower()], 5
                )
            }

        return {"query": req.query, "datasets": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingest ───────────────────────────────
@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        dataset_name = file.filename
        count = ingest_reviews(tmp_path, dataset_name)
        os.unlink(tmp_path)

        return {"message": f"Indexed {count} reviews"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset List ─────────────────────────
@app.get("/api/datasets")
def list_datasets():
    collection = get_collection()

    data = collection.get(include=["metadatas"])
    datasets = list(set(
        m.get("dataset", "unknown")
        for m in data["metadatas"]
    ))

    return {"datasets": datasets}


# ── Health ───────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}