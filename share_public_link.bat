@echo off
cd /d "%~dp0"
echo ============================================================
echo   ReviewSense Analytics - Public Shareable Website Launcher
echo   Keep both windows open while teammates use the link.
echo ============================================================
start "ReviewSense Web Server" cmd /c ""C:\Users\laksh\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run dashboard\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"
timeout /t 4 /nobreak >nul
echo Starting Cloudflare Tunnel... copy the https://*.trycloudflare.com link below:
"%~dp0tools\cloudflared.exe" tunnel --url http://localhost:8501 --protocol http2 --no-autoupdate
pause
