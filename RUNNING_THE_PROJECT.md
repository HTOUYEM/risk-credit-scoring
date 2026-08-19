# Running the Credit Risk project

## Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
jupyter lab
```

## macOS or Linux

```bash
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
jupyter lab
```

## Run the full pipeline

```bash
python run_pipeline.py
```

## Run only notebook 4

```bash
python run_pipeline.py --from-step 4 --to-step 4
```

## Project structure

```text
CreditRiskProject/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── requirements.txt
├── setup_env.ps1
├── setup_env.sh
└── run_pipeline.py
```
