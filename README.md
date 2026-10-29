# End-to-End Loss Cost Model

A Tweedie LightGBM model that predicts annual loss cost, deployed to Azure Container Apps and served through a FastAPI endpoint.


## Architecture

```
UI ──► FastAPI (/predict) ──► Feature engineering ──► LightGBM model
                                                            ▲
                                                    Azure Blob Storage
```

The model is loaded once from Azure Blob Storage and cached for the lifetime of the process.

<img width="626" height="572" alt="Screenshot 2026-10-29 4 40 25 PM" src="https://github.com/user-attachments/assets/2bdc12d4-78c2-47cb-b2c3-56308bdabc2f" />


## Project Structure

```
├── app.py                    
├── src/
│   ├── api/main.py           
│   ├── inference/predict.py  
│   ├── features.py           
│   ├── storage.py
|   ├── monitoring
|       ├── drift.py
|       ├── run_unlabeled.py
|       └── run_labeled.py     
│   └── config.py
|            
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci_cd.yml
```

## Quickstart

Requires Python 3.11 and access to the Azure storage account holding the model (authenticated via `DefaultAzureCredential`).

```bash
pip install -r requirements.txt

# Start the API
uvicorn src.api.main:app --port 8000

# Start the UI
streamlit run app.py
```

## API

**`GET /health`** returns `{"status": "ok"}`.

**`POST /predict`**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"RiskArea": 5.0, "BonusMalus": 100, "DrivAge": 35, "LicAge": 120, "VehUsage": "Private"}'
```

```json
{"loss_cost": 123.45}
```

## Configuration

| Variable                     | Default                         | Purpose                       |
|------------------------------|---------------------------------|-------------------------------|
| `MODEL_KEY`                  | `loss_cost_lgbm.txt`            | Model blob name               |
| `AZURE_CONTAINER`            | `models`                        | Blob container                |
| `AZURE_STORAGE_ACCOUNT_NAME` | `purepremst`                    | Storage account               |

## Docker

```bash
docker build -t loss-cost-model .
docker run -p 8000:8000 loss-cost-model
```
## Monitoring

Two jobs, split by whether ground truth has matured:

```bash
python -m src.monitoring.run_unlabeled   # PSI/CSI — no labels needed
python -m src.monitoring.run_labeled     # Mean Tweedie Deviance — needs labels
```

## Deployment

Every push to `main` triggers GitHub Actions to build the image, push it to Azure Container Registry, and update the `model-api` Container App. Required repository secrets: `AZURE_CREDENTIALS`, `ACR_REGISTRY`, `ACR_USERNAME`, `ACR_PASSWORD`.
