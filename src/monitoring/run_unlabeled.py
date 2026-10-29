"""run_unlabeled.py
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from src.monitoring.drift import band, bin_edges, csi_by_feature, psi
from src.config import Config

def run(reference: pd.DataFrame, 
        monitoring: pd.DataFrame, 
        ts: str,
        cadence: str = "M") -> None:

    periods = monitoring[ts].dt.to_period(cadence).astype(str)
    edges = bin_edges(reference['loss_cost_pred'].to_numpy())
    
    report = {}
    for period, chunk in monitoring.groupby(periods):
        score_psi = psi(reference['loss_cost_pred'].to_numpy(), 
                        chunk['loss_cost_pred'].to_numpy(), edges)
        entry = {'score_psi': score_psi}
        if band(score_psi) == 'material':
            csi = csi_by_feature(reference, chunk, Config().feature_columns)
            entry['csi'] = csi
        report[period] = entry
    
    return report 

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cadence", default="M")
    p.add_argument("--reference", default="data/reference.csv")
    p.add_argument("--monitoring", default="data/monitoring.csv")
    a = p.parse_args()

    reference = pd.read_csv(a.reference)
    monitoring = pd.read_csv(a.monitoring)

    report = run(reference, monitoring, cadence=a.cadence)
    print(json.dumps(report, indent=2))

