import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
    SAMPLE_CSV_PATH: str = os.getenv("SAMPLE_CSV_PATH", str(BASE_DIR / "backend" / "sample_reviews.csv"))
    COLLECTION_NAME: str = "product_reviews"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    MIN_REVIEW_LENGTH: int = 10
    CHUNK_SIZE: int = 500
    BATCH_SIZE: int = 100
    EMBEDDING_BATCH_SIZE: int = 32
    CLUSTERING_THRESHOLD: float = 0.75
    MAX_CLUSTERS: int = 5
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {"csv"}
    CORS_ORIGINS: list = ["*"]
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
