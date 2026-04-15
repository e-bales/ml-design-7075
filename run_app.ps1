# Run Streamlit app from the project virtual environment
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
if (-Not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Virtual environment not found. Create it with `python -m venv .venv` and install requirements."
    exit 1
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1
Write-Host "Starting Streamlit app..."
streamlit run streamlit_app.py