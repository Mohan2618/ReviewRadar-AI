# 📊 ReviewRadar AI v2.0  
**Semantic Customer Review Analysis System**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

---

## 🚀 Overview

ReviewRadar AI is a production-ready semantic search and analysis engine for customer reviews. It uses advanced NLP to understand context rather than just keywords, making it perfect for analyzing large volumes of feedback.

### ✨ Key Improvements (v2.0)

- ✅ **Flexible Schema Support**: Accept ANY CSV structure (any columns, any format)
- ✅ **Auto-detection**: Automatically finds review text columns
- ✅ **Dynamic Metadata**: Preserves and indexes all custom fields
- ✅ **Production Ready**: Comprehensive error handling, logging, validation
- ✅ **Render Deployment**: Optimized for cloud deployment with proper configuration
- ✅ **Advanced Insights**: Entity extraction, detailed sentiment analysis, clustering
- ✅ **CORS Support**: Ready for multi-domain deployments
- ✅ **Better Performance**: Optimized batch processing, smart caching

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | Search reviews using natural language (not keywords) |
| 📁 **CSV Upload** | Upload any CSV format - we auto-detect the structure |
| 🤖 **Smart Detection** | Automatically finds review text in `review_text`, `review`, `text`, `content`, or `description` columns |
| 📊 **Sentiment Analysis** | Positive, Negative, Neutral breakdown with percentages |
| 🔤 **Keyword Extraction** | Identify important terms and trends automatically |
| 🏘️ **Smart Clustering** | Group similar reviews and complaints |
| 🎯 **Entity Detection** | Extract issues, features, and products mentioned |
| 📂 **Dataset Filtering** | Compare results across multiple datasets |
| 📈 **Advanced Analytics** | Deep insights including statistics and entity mapping |
| 🚀 **Cloud Ready** | Optimized for Render, Docker, and production deployments |

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   Frontend (UI)     │  HTML/CSS/JavaScript
│  (index.html)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   FastAPI Server    │  Backend API
│   (main.py)         │
└──────────┬──────────┘
           │
    ┌──────┴──────────┬────────────┬────────────┐
    ▼                 ▼            ▼            ▼
┌────────┐      ┌─────────┐  ┌────────┐  ┌─────────┐
│ Search │      │ Ingest  │  │Insights│  │Analysis │
│(vector)│      │(embed)  │  │(NLP)   │  │(stats)  │
└────┬───┘      └────┬────┘  └───┬────┘  └────┬────┘
     │               │           │           │
     └───────────────┼───────────┼───────────┘
                     ▼
            ┌────────────────────┐
            │  ChromaDB (Vector) │
            │  Persistent Store  │
            └────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **AI/ML** | Sentence Transformers, PyTorch |
| **Database** | ChromaDB (Vector DB) |
| **Data Processing** | Pandas, NumPy |
| **Deployment** | Docker, Render, Cloud |
| **Language** | Python 3.10+ |

---

## 📂 Project Structure

```
ReviewRadar-AI/
│
├── backend/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application & routes
│   ├── config.py             # Configuration management (ENV support)
│   ├── ingest.py             # CSV ingestion with flexible schema
│   ├── search.py             # Semantic search engine
│   └── insights.py           # Advanced analytics & NLP
│
├── frontend/
│   └── index.html            # Interactive web UI
│
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Local development setup
├── Procfile                  # Render deployment config
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version
├── start.sh                  # Startup script
│
├── .env.example              # Environment template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## ⚙️ Installation & Setup

### 1️⃣ Prerequisites
- Python 3.10+
- pip or conda
- 2GB free disk space (for models)
- Internet connection (first run downloads models)

### 2️⃣ Clone Repository
```bash
git clone https://github.com/Mohan2618/ReviewRadar-AI.git
cd ReviewRadar-AI
```

### 3️⃣ Create Virtual Environment
```bash
# Windows
python -3.10 -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3.10 -m venv venv
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ Run Application
```bash
# Development (with hot reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6️⃣ Access Application
Open http://127.0.0.1:8000 in your browser

---

## 📥 Usage Guide

### Uploading Data (Any Format!)

The app now supports **ANY CSV structure**:

#### Example 1: Simple Format
```csv
review_text
"Great product, highly recommend!"
"Terrible quality, broke after 1 day"
```

#### Example 2: Detailed Format
```csv
review_text,product_name,rating,date,category,reviewer_name
"Excellent camera!",iPhone 15,5,2024-01-15,Electronics,John_Doe
"Battery drains quickly",iPhone 15,2,2024-01-14,Electronics,Jane_Smith
"Love the design!",iPhone 15,5,2024-01-10,Design,Alex_Johnson
```

#### Example 3: Custom Columns
```csv
review_content,product_id,customer_tier,purchase_date,region,satisfaction
"Amazing experience",P001,Gold,2024-01-15,US,very_satisfied
"Poor support",P002,Silver,2024-01-14,EU,dissatisfied
```

**The app will:**
✅ Auto-detect the review column  
✅ Preserve all other columns as metadata  
✅ Make all fields searchable and filterable  
✅ Include custom fields in results  

### Searching
1. Enter a natural language query (e.g., "customers upset about battery life")
2. Adjust Top K (5, 10, 20 results)
3. Filter by dataset if needed
4. Click "Search"

### Analyzing
1. Click the "📊 Insights" button after a search
2. View:
   - Sentiment distribution
   - Top keywords (positive/negative/all)
   - Issue clustering
   - Entity extraction
   - Statistical analysis

### Comparing Datasets
Use the `/api/compare` endpoint to compare multiple datasets on a topic

---

## 🔌 API Endpoints

### Search
```http
POST /api/search
Content-Type: application/json

{
  "query": "battery issues",
  "top_k": 10,
  "dataset": null
}
```

### Insights
```http
POST /api/insights
Content-Type: application/json

{
  "query": "customer satisfaction",
  "top_k": 50,
  "dataset": "amazon_reviews"
}
```

### Ingest
```http
POST /api/ingest
multipart/form-data

file: <CSV file>
```

### Get Datasets
```http
GET /api/datasets
```

### Health Check
```http
GET /api/health
```

### Get Metadata Fields
```http
GET /api/datasets/{dataset_name}/metadata-fields
```

See API docs at `/api/docs` (Swagger UI) when running locally.

---

## 🐳 Docker Deployment

### Run Locally
```bash
# Using Docker Compose (recommended)
docker-compose up

# Or manual Docker
docker build -t reviewradar-ai .
docker run -p 8000:8000 reviewradar-ai
```

### Access
http://localhost:8000

---

## ☁️ Cloud Deployment (Render)

### Quick Start
1. Push code to GitHub
2. Connect repository to Render
3. Set environment variables
4. Deploy!

---

## 📋 Configuration

### Environment Variables
Create a `.env` file (see `.env.example`):

```env
# Application
ENV=production
LOG_LEVEL=INFO

# Server
PORT=8000

# AI Model (lightweight by default)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search settings
DEFAULT_TOP_K=10
MAX_TOP_K=100

# File upload
MAX_FILE_SIZE=52428800  # 50MB

# Clustering
CLUSTERING_THRESHOLD=0.75
MAX_CLUSTERS=5
```

---

## 🎓 Advanced Usage

### Custom CSV Columns
Upload any CSV with any columns - they'll be preserved:

```python
# Your CSV
review_text, brand, purchase_price, ship_time, defect_type

# The app will:
# ✓ Index reviews
# ✓ Keep brand, purchase_price, ship_time, defect_type as metadata
# ✓ Allow filtering by any field
# ✓ Return all fields in results
```

### Dynamic Result Fields
All CSV columns appear in results:

```json
{
  "results": [
    {
      "rank": 1,
      "review_text": "...",
      "similarity_score": 0.89,
      "dataset": "reviews_2024",
      "brand": "Samsung",
      "purchase_price": "$899",
      "ship_time": "2_days",
      "defect_type": "screen_issue",
      ...any_other_column
    }
  ]
}
```

### Batch Processing
Upload large CSVs (tested up to 10K+ reviews):
- Automatic chunking
- Progress tracking
- Error recovery
- Encoding auto-detection

---

## 🚀 Performance Tips

| Optimization | Impact |
|-------------|--------|
| Use `all-MiniLM-L6-v2` (default) | ✅ Lightweight, fast |
| Limit `MAX_TOP_K` | ⚡ Faster searches |
| Use dataset filters | 🎯 Precise results |
| Batch uploads | 📦 Efficient indexing |
| GPU instances | 🚀 10x faster (if available) |

---

## 🐛 Troubleshooting

### Issue: "Could not find review text column"
**Solution**: CSV must have a column like: `review_text`, `review`, `text`, `content`, `description`, or `feedback`

### Issue: "Encoding error"
**Solution**: App auto-tries UTF-8, Latin1, ISO-8859-1. If still failing, convert CSV to UTF-8.

### Issue: Slow indexing
**Solution**: 
- Use smaller batches (reduce `BATCH_SIZE`)
- Split large CSV files
- Use GPU if available

### Issue: Out of memory
**Solution**:
- Use lightweight embedding model (already default)
- Reduce batch sizes in config
- Deploy on higher-tier instance

---

## 📊 Example Workflows

### 1. Amazon Review Analysis
1. Export Amazon reviews as CSV
2. Upload via UI
3. Search: "delivery issues" → get relevant reviews
4. Click Insights → see sentiment, top complaints

### 2. Multi-Product Comparison
1. Upload reviews for Product A
2. Upload reviews for Product B  
3. Search: "feature X"
4. Use Compare API → see side-by-side stats

### 3. Issue Tracking
1. Upload support tickets
2. Search: "login problems"
3. Extract entities → get "issue: login" mention count
4. Export insights for product team

---

## 🛡️ Security

- ✅ Input validation on all endpoints
- ✅ File size limits
- ✅ Encoding detection
- ✅ Error message sanitization
- ✅ CORS configuration
- ✅ Environment variable management
- ✅ Production logging

---

## 📈 What's New in v2.0

- ✨ **Flexible CSV Schema**: Accept any column structure
- 🔧 **Auto-detection**: Intelligent review column detection
- 📝 **Dynamic Metadata**: Preserve all CSV fields
- 🐛 **Better Error Handling**: Comprehensive validation & messages
- 🚀 **Cloud Ready**: Render-optimized, Docker-ready
- 📊 **Advanced Analytics**: Entity extraction, detailed statistics
- 🔐 **Production Hardened**: Logging, CORS, error recovery
- 📚 **Better Documentation**: Deployment guide, API docs, examples

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

---

## 📄 License

MIT License - feel free to use commercially

---

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Real-time analytics dashboard
- [ ] User authentication
- [ ] Advanced filtering UI
- [ ] Export to PDF/CSV
- [ ] GraphQL API
- [ ] Mobile app
- [ ] Self-hosted LLM integration

---

## 📞 Support

- 📖 **Documentation**: This README contains setup, API, and deployment guidance.
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Email**: See GitHub profile

---

## 🌟 Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com)
- [ChromaDB](https://www.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- [PyTorch](https://pytorch.org)

---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**Status**: ✅ Production Ready

⭐ If you find this helpful, please star the repository!


## 📊 Example CSV Format
review_text,product_name,rating
"Battery drains fast","Phone A",2
"Great performance","Phone B",5

---

## ❌ Limitations
- Works only with CSV files  
- Requires review_text column  
- No real-time data integration  
- Basic sentiment model (can be improved)  

---

## 🔮 Future Scope
- Real-time review scraping  
- Advanced AI summarization  
- Multilingual support  
- Better UI dashboards  
- Cloud deployment  

---

## 📚 References
- https://fastapi.tiangolo.com  
- https://www.sbert.net  
- https://docs.trychroma.com  
- https://pandas.pydata.org  

---

## 👨‍💻 Author
Mohan  
B.Tech CSE
