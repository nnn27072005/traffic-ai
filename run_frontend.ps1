# run_frontend.ps1
Write-Host "--- TRAFFIC_AI_CORE Frontend Uplink ---" -ForegroundColor Cyan

Set-Location frontend

# 1. Install dependencies
Write-Host "[*] Synchronizing Asset Modules (npm install)..." -ForegroundColor Yellow
npm install

# 2. Launch Dev Server
Write-Host "[!] Launching Command Center UI (Port 3000)..." -ForegroundColor Green
Write-Host "[!] Press Ctrl+C to terminate session." -ForegroundColor Gray
npm run dev
