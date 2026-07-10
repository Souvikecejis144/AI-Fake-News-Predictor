$projectRoot = "c:\Users\souvi\OneDrive\Desktop\AI Fake News Prediction Model"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

Set-Location $projectRoot

Write-Host "=== Step 1: Training Model ===" -ForegroundColor Cyan
& $python training\train.py

Write-Host "`n=== Step 2: Running Diagnostic Tests ===" -ForegroundColor Cyan
& $python audit_model.py
