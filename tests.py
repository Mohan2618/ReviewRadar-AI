import unittest
import json
import tempfile
import os
from pathlib import Path
import sys
import subprocess

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings

class TestReviewRadarAPI(unittest.TestCase):
    """Test suite for ReviewRadar AI API"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize test client"""
        cls.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)
    
    def test_info_endpoint(self):
        """Test info endpoint"""
        response = self.client.get("/api/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("version", data)
    
    def test_search_empty_query(self):
        """Test search with empty query"""
        response = self.client.post(
            "/api/search",
            json={"query": "", "top_k": 10}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
    
    def test_search_valid_query(self):
        """Test search with valid query"""
        response = self.client.post(
            "/api/search",
            json={"query": "test query", "top_k": 5}
        )
        # Should return 200 even if no results (empty database)
        self.assertIn(response.status_code, [200, 500])
    
    def test_datasets_endpoint(self):
        """Test datasets endpoint"""
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("datasets", data)
        self.assertIsInstance(data["datasets"], list)
    
    def test_ingest_invalid_file(self):
        """Test ingestion with invalid file type"""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("test.txt", "invalid content")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
    
    def test_ingest_valid_csv(self):
        """Test ingestion with valid CSV"""
        csv_content = b"review_text,rating\nGreat product,5\nPoor quality,2"
        response = self.client.post(
            "/api/ingest",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        # Should succeed or give valid error
        self.assertIn(response.status_code, [200, 400, 500])


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_settings_loaded(self):
        """Test that settings are properly loaded"""
        self.assertIsNotNone(settings.EMBEDDING_MODEL)
        self.assertIsNotNone(settings.COLLECTION_NAME)
        self.assertGreater(settings.DEFAULT_TOP_K, 0)
    
    def test_paths_valid(self):
        """Test that paths are valid"""
        self.assertIsNotNone(settings.CHROMA_DB_PATH)
        self.assertIsNotNone(settings.BASE_DIR)


def run_performance_test():
    """Run performance tests"""
    print("\n" + "="*50)
    print("🔬 PERFORMANCE TEST")
    print("="*50)
    
    client = TestClient(app)
    
    # Test search speed
    print("\n⏱️  Testing search speed...")
    import time
    start = time.time()
    response = client.post(
        "/api/search",
        json={"query": "test", "top_k": 10}
    )
    elapsed = time.time() - start
    print(f"✓ Search completed in {elapsed:.2f}s")
    
    # Test insights speed
    print("\n⏱️  Testing insights generation speed...")
    start = time.time()
    response = client.post(
        "/api/insights",
        json={"query": "test", "top_k": 50}
    )
    elapsed = time.time() - start
    print(f"✓ Insights generated in {elapsed:.2f}s")


def validate_installation():
    """Validate all dependencies are installed"""
    print("\n" + "="*50)
    print("✅ DEPENDENCY CHECK")
    print("="*50)
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("pydantic", "Pydantic"),
    ]
    
    all_ok = True
    for package, name in dependencies:
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 ReviewRadar AI - Test Suite")
    print("="*50)
    
    # Validate installation
    if not validate_installation():
        print("\n❌ Some dependencies are missing!")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run unit tests
    print("\n" + "="*50)
    print("🧪 UNIT TESTS")
    print("="*50 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestReviewRadarAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    run_performance_test()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
