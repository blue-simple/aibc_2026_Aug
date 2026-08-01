$ErrorActionPreference = "Stop"

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Environment is ready."
Write-Host "Run the app with: .\.venv\Scripts\python.exe -m streamlit run app.py"
