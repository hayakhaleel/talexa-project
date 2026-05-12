$ErrorActionPreference = "Stop"

Write-Host "Preparing NLLB translation model for Talexa..." -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv_rebuilt\Scripts\python.exe"
$Downloader = Join-Path $PSScriptRoot "download_nllb_translation_model.py"

Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $Python"
Write-Host "Downloader: $Downloader"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}

if (-not (Test-Path $Downloader)) {
    throw "Downloader script not found: $Downloader"
}

Write-Host "Checking Python version..." -ForegroundColor Cyan
& $Python --version

Write-Host "Installing requirements and downloading/caching NLLB model. This can take several minutes the first time..." -ForegroundColor Cyan
& $Python $Downloader
if ($LASTEXITCODE -ne 0) {
    throw "NLLB setup failed."
}

Write-Host "Done. NLLB translation model is ready." -ForegroundColor Green
