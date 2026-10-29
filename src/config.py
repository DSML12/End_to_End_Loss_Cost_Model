"""config.py
"""
import os

class Config:

    feature_columns = [
        'RiskArea', 'bm_centered',
        'bm_bonus', 'bm_penality', 'bm_is_bonus', 'bm_is_penality',
        'driv_age_sq', 'inv_driv_age', 'young', 'senior', 'licage_years',
        'first_year', 'first_two_years', 'licage_log', 'experience_gap',
        'late_starter', 'young_x_risk_area', 'young_x_licAge_fst_2',
        'bm_penality_x_risk_area', 'bm_x_age', 'bm_to_licage', 'bm_to_age',
        'VehUsage_lgb'
    ]

    model_key = os.getenv('MODEL_KEY', 'loss_cost_lgbm.txt')
    azure_container = os.getenv('AZURE_CONTAINER', 'models')
    azure_storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'purepremst')
