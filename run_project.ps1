param(
    [switch]$NoBrowser
)

$projectRoot = $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$next = Join-Path $frontendRoot "node_modules\next\dist\bin\next"

function Test-ListeningPort([int]$Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

if (-not (Test-Path -LiteralPath $next)) {
    throw "Next.js dependencies not found. Run npm install in the frontend directory first."
}

if (-not (Test-ListeningPort 8000)) {
    $backendCommand = "& '$python' 'backend\main.py'"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $backendCommand -WorkingDirectory $projectRoot
}

if (-not (Test-ListeningPort 3000)) {
    $frontendCommand = "& 'C:\Program Files\nodejs\node.exe' '$next' dev --hostname localhost --port 3000"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $frontendCommand -WorkingDirectory $frontendRoot
}

for ($attempt = 0; $attempt -lt 20; $attempt++) {
    if (Test-ListeningPort 3000) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not (Test-ListeningPort 3000)) {
    throw "The Next.js server did not start on port 3000. Check the frontend PowerShell window for errors."
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:3000"
}

Write-Host "Frontend: http://localhost:3000"
Write-Host "API: http://localhost:8000"
