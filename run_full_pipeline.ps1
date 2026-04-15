# Fetch raw data, create processed features, and train the model in one step
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
if (-Not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Virtual environment not found. Create it with `python -m venv .venv` and install requirements."
    exit 1
}
. .\.venv\Scripts\Activate.ps1
if ($args.Count -eq 0) {
    Write-Error "Please provide a ticker symbol, e.g. .\run_full_pipeline.ps1 AAPL"
    exit 1
}
python .\run_full_pipeline.py $args[0]
