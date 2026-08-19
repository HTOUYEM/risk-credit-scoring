$ErrorActionPreference = "Stop"

python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk-env --display-name "Python (Credit Risk)"

Write-Host "Environment ready."
Write-Host "Activate later with: .\.venv\Scripts\Activate.ps1"
