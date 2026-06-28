#!/bin/bash
# ReviewRadar AI - Installation & Deployment Verification Script

echo "======================================"
echo "🔍 ReviewRadar AI v2.0 - Verification"
echo "======================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        ((failed++))
    fi
}

# Function to check directory
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        ((failed++))
    fi
}

# Function to check Python package
check_package() {
    python -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1 (NOT INSTALLED)"
        ((failed++))
    fi
}

echo "📁 Checking Project Structure..."
echo "================================"
check_dir "backend"
check_dir "frontend"
check_file "backend/main.py"
check_file "backend/config.py"
check_file "backend/ingest.py"
check_file "backend/search.py"
check_file "backend/insights.py"
check_file "frontend/index.html"
echo ""

echo "📄 Checking Configuration Files..."
echo "==================================="
check_file "Procfile"
check_file "Dockerfile"
check_file "docker-compose.yml"
check_file ".env.example"
check_file ".gitignore"
check_file "requirements.txt"
check_file "runtime.txt"
check_file "start.sh"
echo ""

echo "📚 Checking Documentation..."
echo "============================"
check_file "README.md"

echo ""

echo "🧪 Checking Test & Sample Files..."
echo "==================================="
check_file "tests.py"
check_file "sample_reviews.csv"
echo ""

echo "📦 Checking Dependencies..."
echo "============================"
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Note:${NC} Run 'pip install -r requirements.txt' if not done yet"
    echo ""
    echo "Key packages to install:"
    echo "  - fastapi"
    echo "  - uvicorn"
    echo "  - chromadb"
    echo "  - sentence-transformers"
    echo "  - torch"
    echo "  - pandas"
    ((passed++))
else
    echo -e "${RED}✗${NC} requirements.txt not found"
    ((failed++))
fi
echo ""

echo "======================================"
echo "📊 Verification Summary"
echo "======================================"
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Run locally: uvicorn backend.main:app --reload"
    echo "3. Open browser: http://localhost:8000"
    echo "4. Read README.md for usage and deployment notes"
    exit 0
else
    echo -e "${RED}❌ Some files are missing!${NC}"
    echo ""
    echo "Please ensure all required project files are present."
    echo "Read README.md for setup guidance."
    exit 1
fi
