# run_api.ps1
Write-Host "--- TRAFFIC_AI_CORE Backend Uplink ---" -ForegroundColor Cyan

# 1. Setup Virtual Environment
if (!(Test-Path "venv")) {
    Write-Host "[*] Creating Python Virtual Environment..." -ForegroundColor Yellow
    python -m venv venv
}

# 2. Activate and Install
Write-Host "[*] Updating Dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch Server
Write-Host "[!] Launching Command Center API (Port 8000)..." -ForegroundColor Green
Write-Host "[!] Press Ctrl+C to terminate session." -ForegroundColor Gray
uvicorn src.api.main:app --reload --port 8000
