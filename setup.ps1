# setup.ps1
# First-time project setup for Interview Transcriber.
# Run once from the project root in PowerShell.
#
# Usage:
#   .\setup.ps1

Write-Host "Interview Transcriber — Project Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Confirm we are in the project root
if (-not (Test-Path ".\main.py")) {
    Write-Host "ERROR: Run this script from the project root directory (where main.py lives)." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it does not exist
if (-not (Test-Path ".\venv")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "`nVirtual environment already exists — skipping creation." -ForegroundColor Green
}

# Activate
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip (required for v26+ compatibility with PyAudioWPatch)
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create sessions directory
if (-not (Test-Path ".\sessions")) {
    New-Item -ItemType Directory -Path ".\sessions" | Out-Null
    Write-Host "`nCreated sessions\ directory." -ForegroundColor Green
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "To verify audio capture (Phase 1):" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run unit tests (no hardware required):" -ForegroundColor White
Write-Host "  python -m pytest tests\ -v" -ForegroundColor Yellow
Write-Host ""
