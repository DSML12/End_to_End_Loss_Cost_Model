"""predict.py
"""

from __future__ import annotations
import pandas as pd
from src.storage import ModelStore
import lightgbm as lgb
from dataclasses import dataclass, asdict
from functools import lru_cache
from src.features import build_features
from src.config import Config

# define the policy and decision data classes
@dataclass
class Policy:
    RiskArea: float
    BonusMalus: int
    DrivAge: int
    LicAge: int
    VehUsage: str

@dataclass
class Decision:
    loss_cost: float

# load the model and cache it
class Scorer:

    def __init__(self,
                 config: Config = None,
                 model_store: ModelStore = None):
        self.config = config or Config()
        self.model_store = model_store or ModelStore(self.config)
        self.model = self.model_store.load_model()

    def score(self, feature_df: pd.DataFrame) -> float:
        return self.model.predict(feature_df)[0]


@lru_cache
def scorer():
  return Scorer()


def predict_loss_cost(policy: Policy) -> Decision:
    policy_df = pd.DataFrame([asdict(policy)])
    feature_df = build_features(policy_df)
    loss_cost = scorer().score(feature_df)
    return Decision(loss_cost = float(loss_cost))
