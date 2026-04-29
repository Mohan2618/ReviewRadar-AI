import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "chroma_db")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "sample_reviews.csv")

# ChromaDB collection name
COLLECTION_NAME = "product_reviews"

# Sentence Transformer model (lightweight + accurate)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Search defaults
DEFAULT_TOP_K = 10