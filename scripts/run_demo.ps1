# KAVACH live demo launcher (Windows PowerShell). The router (Rules / TF-IDF / XLM-R) is chosen in the web UI.
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1          # Gemini (key from .env)
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Mock    # offline, no key
param([switch]$Mock, [int]$Port = 8000)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Mock -and -not (Test-Path ".env") -and -not $env:GEMINI_API_KEY) {
    Write-Host "No .env file found. Create one with:  Set-Content .env `"GEMINI_API_KEY=your-key`"" -ForegroundColor Yellow
    exit 1
}
# The XLM-R router needs torch + transformers; install them once if the model weights are present.
if (Test-Path "dl\best_router\model.safetensors") {
    python -c "import torch, transformers" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing torch + transformers for the XLM-R router (one time, ~250 MB)..." -ForegroundColor Cyan
        python -m pip install torch transformers
    }
}
$provider = if ($Mock) { "mock" } else { "gemini" }
Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 15; Start-Process "http://localhost:$p" } -ArgumentList $Port | Out-Null
python -m kavach.server --provider $provider --port $Port
