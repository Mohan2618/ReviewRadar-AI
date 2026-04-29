📊 ReviewRadar AI

Semantic Customer Review Analysis System

🚀 Overview

ReviewRadar AI is a web-based application that helps analyze large volumes of customer reviews using Artificial Intelligence. It allows users to perform semantic search, extract insights, and understand customer feedback quickly.

Instead of traditional keyword search, this system uses semantic search, meaning it understands the context of the query and returns more relevant results.

🎯 Features
🔍 Semantic Search – Search reviews using natural language
📁 CSV Upload Support – Upload multiple review datasets
📊 Sentiment Analysis – Positive, Negative, Neutral breakdown
🧠 Keyword Extraction – Identify important terms and trends
⚠️ Complaint Clustering – Group similar issues
📂 Dataset Filtering – Filter results by dataset
⚖️ Product Comparison – Compare datasets based on reviews
📈 Insights Dashboard – Visual understanding of feedback
🏗️ Architecture
Frontend (HTML/CSS/JS)
        ↓
FastAPI Backend
        ↓
Embedding Model (Sentence Transformers)
        ↓
ChromaDB (Vector Database)
        ↓
Insights Engine (Sentiment + Keywords + Clustering)
🛠️ Tech Stack
Backend: FastAPI, Uvicorn
Frontend: HTML, CSS, JavaScript
AI Model: Sentence Transformers
Database: ChromaDB (Vector DB)
Data Processing: Pandas, NumPy
Language: Python
📂 Project Structure
reviewradar-ai/
│
├── backend/
│   ├── main.py
│   ├── ingest.py
│   ├── search.py
│   ├── insights.py
│   └── config.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── chroma_db/
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone <your-repo-link>
cd reviewradar-ai
2️⃣ Create Virtual Environment
py -3.10 -m venv venv

Activate:

.\venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Backend Server
uvicorn backend.main:app --reload
5️⃣ Open in Browser
http://127.0.0.1:8000
📥 Usage
Upload a CSV file with a review_text column
Enter a search query (e.g., "battery issues")
View relevant reviews
Click Insights to analyze sentiment & keywords
Use dataset filter to compare results
📊 Example CSV Format
review_text,product_name,rating
"Battery drains fast","Phone A",2
"Great performance","Phone B",5
❌ Limitations
Works only with CSV files
Requires review_text column
No real-time data integration
Basic sentiment model (can be improved)
🔮 Future Scope
Real-time review scraping
Advanced AI summarization
Multilingual support
Better UI dashboards
Cloud deployment
📚 References
FastAPI – https://fastapi.tiangolo.com
Sentence Transformers – https://www.sbert.net
ChromaDB – https://docs.trychroma.com
Pandas – https://pandas.pydata.org