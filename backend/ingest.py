import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, SAMPLE_CSV_PATH
import uuid
import time

print("🔄 Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Model loaded successfully")


def ingest_reviews(csv_path: str = None, dataset_name: str = None):
    path = csv_path or SAMPLE_CSV_PATH

  
    if not dataset_name:
        dataset_name = "dataset_" + str(uuid.uuid4())[:6]

    dataset_name = dataset_name.replace(".csv", "").strip().lower()

    print(f"📂 Loading reviews from: {path}")
    print(f"📁 Dataset name: {dataset_name}")


    df = pd.read_csv(path)
    if df.empty:
    raise ValueError("CSV is empty or invalid.")
  
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "review_text" not in df.columns:
        raise ValueError(f"CSV must contain 'review_text'. Found: {list(df.columns)}")

  
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 10]

    print(f"✅ Loaded {len(df)} valid reviews")

   
    MAX_ROWS = 2000

    if len(df) > MAX_ROWS:
        print(f"⚠️ Trimming dataset from {len(df)} to {MAX_ROWS} rows")
        df = df.head(MAX_ROWS)

    # Generate embeddings
    texts = df["review_text"].tolist()
    print("🔢 Generating embeddings...")

    try:
        start = time.time()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        print(f"⏱ Embedding time: {round(time.time()-start,2)} sec")
    except Exception as e:
        print("❌ Embedding error:", str(e))
        raise

    # Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

 
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    
    def safe_str(val):
        return str(val) if pd.notna(val) else "N/A"

    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    metadatas = []

    for _, row in df.iterrows():
        meta = {
            "review_text": row["review_text"],
            "dataset": dataset_name
        }

        for col in ["product_name", "rating", "date", "reviewer_name", "category"]:
            if col in row:
                meta[col] = safe_str(row[col])

        metadatas.append(meta)

    
    batch_size = 100

    print("📦 Inserting into database...")
    for i in range(0, len(ids), batch_size):
        try:
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size].tolist(),
                metadatas=metadatas[i:i + batch_size],
                documents=texts[i:i + batch_size]
            )
            print(f"✔ Inserted batch {i // batch_size + 1}")
        except Exception as e:
            print("❌ ChromaDB insert error:", str(e))
            raise

    print(f"🎉 Indexed {len(df)} reviews (dataset={dataset_name})")

    return len(df)
