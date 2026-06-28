import logging
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import traceback

from .config import settings
from .search import semantic_search, get_all_datasets, get_metadata_fields
from .ingest import ingest_reviews
from .insights import sentiment_breakdown, top_keywords, cluster_texts, detailed_analysis

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReviewRadar AI",
    description="Semantic Customer Review Analysis Engine",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_ui():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ReviewRadar AI is running"}


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = settings.DEFAULT_TOP_K
    dataset: Optional[str] = None


class ComparisonRequest(BaseModel):
    query: str
    top_k: Optional[int] = 50
    datasets: Optional[List[str]] = None


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "environment": settings.ENV
    }


@app.get("/api/info")
def get_info():
    """Get system information"""
    try:
        datasets = get_all_datasets()
        return {
            "version": "2.0.0",
            "datasets_count": len(datasets),
            "embedding_model": settings.EMBEDDING_MODEL,
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Info endpoint error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/search")
def search(req: SearchRequest):
    logger.info(f"Search query: {req.query}")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = semantic_search(
            req.query.strip(),
            req.top_k or settings.DEFAULT_TOP_K,
            req.dataset
        )
        logger.info(f"Search returned {len(results)} results")
        return {"results": results, "count": len(results), "query": req.query}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/insights")
def insights(req: SearchRequest):
    logger.info(f"Insights request for query: {req.query}")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = semantic_search(req.query.strip(), req.top_k or 50, req.dataset)
        if not results:
            return {"sentiment": {"counts": {}, "percent": {}}, "keywords": {}, "clusters": [], "entities": {}}

        docs = [r.get("review_text", "") for r in results]
        analysis = detailed_analysis(docs)
        clusters = cluster_texts(docs)
        logger.info(f"Insights generated for {len(docs)} documents")
        return {
            "query": req.query,
            "total_results": len(docs),
            "sentiment": analysis["sentiment"],
            "keywords": analysis["keywords"],
            "clusters": clusters,
            "entities": analysis["entities"],
            "statistics": analysis["statistics"]
        }
    except Exception as e:
        logger.error(f"Insights error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Insights failed: {str(e)}")


@app.post("/api/compare")
def compare(req: ComparisonRequest):
    logger.info(f"Comparison request for query: {req.query}")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = semantic_search(req.query.strip(), 100, None)
        by_dataset = {}
        for r in results:
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

        summary = {}
        for ds, rows in by_dataset.items():
            docs = [r.get("review_text", "") for r in rows]
            sent = sentiment_breakdown(docs)
            avg_sim = sum(r.get("similarity_score", 0) for r in rows) / max(1, len(rows))
            summary[ds] = {
                "count": len(rows),
                "avg_similarity": round(avg_sim, 3),
                "sentiment": sent["percent"],
                "keywords": top_keywords(docs, 5)
            }

        logger.info(f"Comparison completed for {len(summary)} datasets")
        return {"query": req.query, "datasets": summary, "total_results": len(results)}
    except Exception as e:
        logger.error(f"Comparison error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    logger.info(f"Ingestion started for file: {file.filename}")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        dataset_name = file.filename
        count = ingest_reviews(tmp_path, dataset_name)
        logger.info(f"Successfully ingested {count} reviews from {dataset_name}")
        return {"message": f"Indexed {count} reviews successfully", "dataset": dataset_name.replace(".csv", "").strip().lower(), "count": count}
    except ValueError as e:
        logger.error(f"Validation error during ingestion: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/api/datasets")
def list_datasets():
    """
    Get list of all datasets
    """
    logger.info("Fetching datasets list")
    
    try:
        datasets = get_all_datasets()
        return {
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        logger.error(f"Error fetching datasets: {e}")
        return {
            "datasets": [],
            "count": 0,
            "error": str(e)
        }


@app.get("/api/datasets/{dataset_name}/metadata-fields")
def get_dataset_fields(dataset_name: str):
    logger.info(f"Fetching metadata fields for dataset: {dataset_name}")
    try:
        fields = get_metadata_fields(dataset_name)
        return {"dataset": dataset_name, "fields": fields, "count": len(fields)}
    except Exception as e:
        logger.error(f"Error fetching metadata fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return {"error": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

