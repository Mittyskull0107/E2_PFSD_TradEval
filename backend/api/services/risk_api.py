# risk_api.py
# NOTE: The real risk classification logic lives in risk_model.py
# This file exists only for backwards compatibility — do not add logic here

from .risk_model import classify_risk, get_model_info

__all__ = ["classify_risk", "get_model_info"]
