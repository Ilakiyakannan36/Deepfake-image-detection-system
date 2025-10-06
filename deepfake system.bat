@echo off
echo ========================================
echo        Starting DeepGuard System
echo ========================================
echo.

REM Navigate to project directory
cd /d C:\Users\ilaki\OneDrive\Desktop\DeepGuard

REM Activate virtual environment
echo Activating Python environment...
call deepguard_env\Scripts\activate.bat

REM Check if environment activated
if errorlevel 1 (
    echo ERROR: Could not activate environment
    echo Please make sure Python is installed
    pause
    exit
)

REM Start the web app
echo.
echo Starting DeepGuard Web App...
echo.
echo Open your browser and go to:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
streamlit run app.py

pause