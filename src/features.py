"""features.py
"""

import numpy as np
import pandas as pd
from src.config import Config

def build_features(df: pd.DataFrame) -> pd.DataFrame:

  # add bonus malus feature
  df['bm_centered'] = df['BonusMalus'] - 100
  df['bm_bonus'] = np.maximum(0, 100 - df['BonusMalus'])
  df['bm_penality'] = np.maximum(0, df['BonusMalus'] - 100)
  df['bm_is_bonus'] = np.where(df['BonusMalus'] > 100, 1, 0)
  df['bm_is_penality'] = np.where(df['BonusMalus'] < 100, 1, 0)

  # add driver age features
  df['driv_age_sq'] = df['DrivAge'] ** 2
  df['inv_driv_age'] = 1 / df['DrivAge']
  df['young'] = (df['DrivAge'] < 25).astype(int)
  df['senior'] = (df['DrivAge'] > 65).astype(int)

  # add license age features
  df['licage_years'] = df['LicAge']/12
  df['first_year'] = (df['licage_years'] <= 1).astype(int)
  df['first_two_years'] = np.maximum(0, 2 - df['licage_years'])
  df['licage_log'] = np.log1p(df['licage_years'])

  # add experience feature
  df['experience_gap'] = df['DrivAge'] - (18 + df['licage_years'])
  df['late_starter'] = (df['experience_gap'] > 5).astype(int)

  # add interaction features
  df['young_x_risk_area'] = df['young'] * df['RiskArea']
  df['young_x_licAge_fst_2'] = df['young'] * df['first_two_years']
  df['bm_penality_x_risk_area'] = df['bm_penality'] * df['RiskArea']
  df['bm_x_age'] = df['bm_centered'] * df['DrivAge']

  df['bm_to_licage'] = df['bm_centered'] / df['licage_years']
  df['bm_to_age'] = df['bm_centered'] / df['DrivAge']

  df['VehUsage_lgb'] = df['VehUsage'].astype('category')

  features = Config().feature_columns
  df = df[features]
  return df.iloc[[0]]


