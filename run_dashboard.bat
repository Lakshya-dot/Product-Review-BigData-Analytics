@echo off
cd /d "%~dp0"
echo ============================================================
echo   ReviewSense Analytics Website Launcher
echo   Local URL:   http://localhost:8501
echo ============================================================
"C:\Users\laksh\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run dashboard\app.py --server.address 0.0.0.0 --server.port 8501
pause
