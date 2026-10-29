"""run_labeled.py
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance
from src.config import Config

def run(reference: pd.DataFrame, 
        labeled: pd.DataFrame, 
        ts: str,
        cadence: str = "M") -> None:

    periods = labeled[ts].dt.to_period(cadence).astype
    ref_loss_cost = np.clip(reference['loss_cost'], 0, None)
    ref_loss_cost_pred = = np.clip(reference['loss_cost_pred'], 1e-6, None),
    ref_score = mean_tweedie_deviance(y_true = ref_loss_cost,
                                      y_pred = ref_loss_cost_pred,
                                      sample_weight = reference['Exposure'],
                                      power = 1.5)
    lower = ref_score * 0.9
    report = {}
    for period, chunk in labeled.groupby(periods):
        loss_cost = np.clip(chunk['loss_cost'], 0, None)
        loss_cost_pred = np.clip(chunk['loss_cost_pred'], 1e-6, None)
        score = mean_tweedie_deviance(y_true = loss_cost,
                                      y_pred = loss_cost_pred,
                                      sample_weight = chunk['Exposure'],
                                      power = 1.5)
        entry = {'score': score}
        entry['breach'] = False
        if score < lower:
            entry['breach'] = True
        report[period] = entry
    
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cadence", default="M")
    p.add_argument("--reference", default="data/reference.csv")
    p.add_argument("--labeled", default="data/monitoring.csv")
    a = p.parse_args()

    reference = pd.read_csv(a.reference)
    labeled = pd.read_csv(a.monitoring)

    report = run(reference, labeled, cadence=a.cadence)
    print(json.dumps(report, indent=2))


