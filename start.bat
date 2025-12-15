cat > start.bat << 'EOF'
@echo off
echo ======================================
echo Starting Screen Time Mental Health System
echo ======================================

echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && venv\Scripts\activate && python app.py"

timeout /t 3 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "python -m http.server 8000"

timeout /t 2 /nobreak > nul

echo Opening Browser...
start http://localhost:8000/frontend/index.html

echo.
echo ======================================
echo System Started Successfully!
echo ======================================
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:8000/frontend/index.html
echo.
echo Close the CMD windows to stop servers
echo ======================================
EOF