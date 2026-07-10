$projectRoot = $PSScriptRoot
$datasetsDirectory = Join-Path $projectRoot "datasets"
$datasetPath = Join-Path $datasetsDirectory "WELFake_Dataset.csv"
$archivePath = Join-Path $datasetsDirectory "welfake.zip"
$datasetUrl = "https://www.kaggle.com/api/v1/datasets/download/saurabhshahane/fake-news-classification"

if (Test-Path -LiteralPath $datasetPath) {
    Write-Host "WELFake dataset already exists: $datasetPath"
    exit 0
}

New-Item -ItemType Directory -Path $datasetsDirectory -Force | Out-Null
Invoke-WebRequest -Uri $datasetUrl -OutFile $archivePath
Expand-Archive -LiteralPath $archivePath -DestinationPath $datasetsDirectory -Force
Remove-Item -LiteralPath $archivePath -Force

if (-not (Test-Path -LiteralPath $datasetPath)) {
    throw "Download completed but WELFake_Dataset.csv was not found in the archive."
}

Write-Host "Downloaded WELFake dataset to $datasetPath"
