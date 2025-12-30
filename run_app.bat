@echo off
echo Starting Calories Burnt Prediction App...

:: Check if dependencies are installed
echo Checking dependencies...
pip list | findstr "fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

:: Start the API server in background
echo Starting API Server on port 8000...
start "API Server" cmd /k "uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait for server to start
timeout /t 5 /nobreak >nul

:: Start the frontend server in background
echo Starting Frontend Server on port 3000...
start "Frontend Server" cmd /k "cd frontend && python -m http.server 3000"

:: Wait for frontend to start
timeout /t 2 /nobreak >nul

:: Open the frontend in browser
echo Opening application in browser...
start http://localhost:3000

echo.
echo Application started successfully!
echo - API Server: http://localhost:8000
echo - Frontend: http://localhost:3000
echo.
echo Press any key to close this window (servers will continue running)...
pause >nul