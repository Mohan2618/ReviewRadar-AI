@echo off
REM ReviewRadar AI - Installation & Deployment Verification (Windows)

setlocal enabledelayedexpansion

echo.
echo ======================================
echo ^>^> ReviewRadar AI v2.0 - Verification
echo ======================================
echo.

set passed=0
set failed=0

REM Function to check file existence
:check_file
if exist "%~1" (
    echo [OK] %~1
    set /a passed+=1
) else (
    echo [FAIL] %~1 (MISSING)
    set /a failed+=1
)
goto :eof

echo Checking Project Structure...
echo ==========================
if exist backend (echo [OK] backend) else (echo [FAIL] backend & set /a failed+=1)
if exist frontend (echo [OK] frontend) else (echo [FAIL] frontend & set /a failed+=1)
if exist backend\main.py (echo [OK] backend\main.py) else (echo [FAIL] backend\main.py & set /a failed+=1)
if exist backend\config.py (echo [OK] backend\config.py) else (echo [FAIL] backend\config.py & set /a failed+=1)
if exist backend\ingest.py (echo [OK] backend\ingest.py) else (echo [FAIL] backend\ingest.py & set /a failed+=1)
if exist backend\search.py (echo [OK] backend\search.py) else (echo [FAIL] backend\search.py & set /a failed+=1)
if exist backend\insights.py (echo [OK] backend\insights.py) else (echo [FAIL] backend\insights.py & set /a failed+=1)
if exist frontend\index.html (echo [OK] frontend\index.html) else (echo [FAIL] frontend\index.html & set /a failed+=1)
echo.

echo Checking Configuration Files...
echo ==============================
if exist Procfile (echo [OK] Procfile) else (echo [FAIL] Procfile & set /a failed+=1)
if exist Dockerfile (echo [OK] Dockerfile) else (echo [FAIL] Dockerfile & set /a failed+=1)
if exist docker-compose.yml (echo [OK] docker-compose.yml) else (echo [FAIL] docker-compose.yml & set /a failed+=1)
if exist .env.example (echo [OK] .env.example) else (echo [FAIL] .env.example & set /a failed+=1)
if exist .gitignore (echo [OK] .gitignore) else (echo [FAIL] .gitignore & set /a failed+=1)
if exist requirements.txt (echo [OK] requirements.txt) else (echo [FAIL] requirements.txt & set /a failed+=1)
if exist runtime.txt (echo [OK] runtime.txt) else (echo [FAIL] runtime.txt & set /a failed+=1)
if exist start.sh (echo [OK] start.sh) else (echo [FAIL] start.sh & set /a failed+=1)
echo.

echo Checking Documentation...
echo ========================
if exist README.md (echo [OK] README.md) else (echo [FAIL] README.md & set /a failed+=1)
echo.

echo Checking Test and Sample Files...
echo ================================
if exist tests.py (echo [OK] tests.py) else (echo [FAIL] tests.py & set /a failed+=1)
if exist sample_reviews.csv (echo [OK] sample_reviews.csv) else (echo [FAIL] sample_reviews.csv & set /a failed+=1)
echo.

echo ======================================
echo Verification Summary
echo ======================================
echo Checks Passed: %passed%
echo Checks Failed: %failed%
echo.

if %failed% equ 0 (
    echo [SUCCESS] All checks passed!
    echo.
    echo Next steps:
    echo 1. Install dependencies: pip install -r requirements.txt
    echo 2. Run locally: uvicorn backend.main:app --reload
    echo 3. Open browser: http://localhost:8000
    echo 4. Read README.md for setup and deployment notes
    exit /b 0
) else (
    echo [ERROR] Some files are missing!
    echo.
    echo Please ensure all required project files are present.
    echo Read README.md for setup guidance.
    exit /b 1
)

endlocal
