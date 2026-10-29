"""api.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict_loss_cost, Policy

app = FastAPI(title = 'Loss Cost Model')

class PredictRequest(BaseModel):
    """
    Prediction requests must contain 
        - RiskArea
        - BonusMalus
        - DrivAge
        - LicAge
        - VehUsage
    """
    RiskArea: float
    BonusMalus: int
    DrivAge: int 
    LicAge: int
    VehUsage: str

class PredictResponse(BaseModel):
    loss_cost: float


# end points
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(req: PredictRequest):
    response = predict_loss_cost(Policy(req.RiskArea, 
                                       req.BonusMalus,
                                       req.DrivAge,
                                       req.LicAge,
                                       req.VehUsage))
    return PredictResponse(loss_cost = response.loss_cost)
