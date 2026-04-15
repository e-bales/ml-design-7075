# Generate processed feature CSVs from raw Alpha Vantage data
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
if (-Not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Virtual environment not found. Create it with `python -m venv .venv` and install requirements."
    exit 1
}
. .\.venv\Scripts\Activate.ps1
python .\run_feature_engineering.py $args
