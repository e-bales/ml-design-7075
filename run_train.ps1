# Train the latest processed dataset and create models/registry.json
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
if (-Not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Virtual environment not found. Create it with `python -m venv .venv` and install requirements."
    exit 1
}
. .\.venv\Scripts\Activate.ps1
python .\run_train.py
