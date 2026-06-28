import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from .config import settings
import uuid
import time
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

try:
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise


def safe_str(val) -> str:
    return str(val).strip() if pd.notna(val) else "N/A"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def detect_review_column(df: pd.DataFrame) -> str:
    common_names = [
        "review_text", "review", "text", "content",
        "description", "feedback", "comment", "message", "body"
    ]
    df_cols = df.columns.str.lower().tolist()
    for col_name in common_names:
        if col_name in df_cols:
            return col_name
    raise ValueError(
        f"Could not find review text column. "
        f"Expected one of: {', '.join(common_names)}. "
        f"Found columns: {', '.join(df.columns)}"
    )


def validate_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        raise ValueError("CSV file is empty")
    review_col = detect_review_column(df)
    df = df.dropna(subset=[review_col])
    df[review_col] = df[review_col].astype(str).str.strip()
    df = df[df[review_col].str.len() >= settings.MIN_REVIEW_LENGTH]
    if df.empty:
        raise ValueError(
            f"No valid reviews found. "
            f"Minimum text length: {settings.MIN_REVIEW_LENGTH} characters"
        )
    return df, review_col


def ingest_reviews(csv_path: str = None, dataset_name: str = None) -> int:
    """
    Ingest reviews from CSV with support for any column structure.
    
    Args:
        csv_path: Path to CSV file
        dataset_name: Name for the dataset
        
    Returns:
        Number of indexed reviews
    """
    path = csv_path or settings.SAMPLE_CSV_PATH
    
    if not dataset_name:
        dataset_name = "dataset_" + str(uuid.uuid4())[:8]
    
    dataset_name = dataset_name.replace(".csv", "").strip().lower()[:50]  # Limit name length
    
    logger.info(f"📂 Loading reviews from: {path}")
    logger.info(f"📁 Dataset: {dataset_name}")
    
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise ValueError(f"Database error: {str(e)}")
    
    total_indexed = 0
    
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    df_reader = None
    
    for encoding in encodings:
        try:
            logger.info(f"Attempting to read CSV with encoding: {encoding}")
            df_reader = pd.read_csv(
                path,
                chunksize=settings.CHUNK_SIZE,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
                dtype=str  # Read all as strings first
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            logger.warning(f"Failed with {encoding}: {e}")
            continue
    
    if df_reader is None:
        raise ValueError(
            "Could not read CSV file with any encoding. "
            "Please ensure the file is a valid CSV."
        )
    
    for chunk_idx, df in enumerate(df_reader):
        try:
            logger.info(f"Processing chunk {chunk_idx + 1}...")
            df = normalize_columns(df)
            df, review_col = validate_csv(df)
            if df.empty:
                logger.warning(f"Chunk {chunk_idx + 1} has no valid reviews, skipping")
                continue

            texts = df[review_col].tolist()
            logger.info(f"Generating embeddings for {len(texts)} reviews...")
            start = time.time()
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                batch_size=settings.EMBEDDING_BATCH_SIZE
            )
            elapsed = round(time.time() - start, 2)
            logger.info(f"Embedding time: {elapsed}s")

            ids = [str(uuid.uuid4()) for _ in range(len(df))]
            metadatas = []
            for _, row in df.iterrows():
                meta = {"review_text": row[review_col], "dataset": dataset_name}
                for col in df.columns:
                    if col != review_col:
                        meta[col] = safe_str(row[col])
                metadatas.append(meta)

            for i in range(0, len(ids), settings.BATCH_SIZE):
                collection.add(
                    ids=ids[i:i + settings.BATCH_SIZE],
                    embeddings=embeddings[i:i + settings.BATCH_SIZE].tolist(),
                    metadatas=metadatas[i:i + settings.BATCH_SIZE],
                    documents=texts[i:i + settings.BATCH_SIZE]
                )
            total_indexed += len(df)
            logger.info(f"Total indexed: {total_indexed}")
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            raise
    
    if total_indexed == 0:
        raise ValueError(
            "No valid reviews were indexed. Check CSV format and content."
        )
    
    logger.info(f"🎉 Successfully indexed {total_indexed} reviews")
    return total_indexed

