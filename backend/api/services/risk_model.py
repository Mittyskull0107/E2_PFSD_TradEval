import joblib
import numpy as np
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[3]
MODEL_PATH  = BASE_DIR / "ml" / "model" / "risk_classifier.pkl"
SCALER_PATH = BASE_DIR / "ml" / "model" / "scaler.pkl"
META_PATH   = BASE_DIR / "ml" / "model" / "model_metadata.json""scaler.pkl"

# ── what features this model expects (document clearly) ───────────
FEATURE_NAMES = [
    "volatility",        # annualised volatility (e.g. 0.24 = 24%)
    "avg_daily_return",  # mean daily return (e.g. 0.001)
    "max_drawdown",      # max drawdown as negative float (e.g. -0.15)
    "sharpe_ratio",      # annualised Sharpe ratio (e.g. 1.2)
    "volume_ratio",      # recent vol / avg vol (e.g. 1.4 = 40% above avg)
]
EXPECTED_FEATURES = len(FEATURE_NAMES)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def _load_artifact(path: Path, name: str):
    """Load a joblib artifact safely. Returns None on any failure."""
    if not path.exists():
        print(f"[risk_model] WARNING: {name} not found at {path}")
        return None
    try:
        artifact = joblib.load(path)
        print(f"[risk_model] Loaded {name} from {path}")
        return artifact
    except Exception as e:
        print(f"[risk_model] ERROR loading {name}: {e}")
        return None


# load lazily so a bad file doesn't crash Django startup
_model  = None
_scaler = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_artifact(MODEL_PATH, "risk_classifier")
    return _model


def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = _load_artifact(SCALER_PATH, "scaler")
    return _scaler
    
def get_model_info() -> dict:
    """Return metadata about the loaded model."""
    import json
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {"note": "No metadata found. Run: cd ml && python train_model.py"}

def classify_risk(features: list) -> dict:
    """
    Predict risk level from numeric features.

    Expected features (in order):
        volatility, avg_daily_return, max_drawdown,
        sharpe_ratio, volume_ratio

    Returns:
        risk_level  : int   — 0=Low, 1=Medium, 2=High
        risk_label  : str
        confidence  : float — 0.0 to 1.0
        probabilities: dict — score per class
    """
    model = _get_model()

    if model is None:
        return {
            "error": "Model not loaded. Run: cd ml && python train_model.py",
            "risk_level": -1,
            "confidence": 0.0,
        }

    # ── validate feature count ────────────────────────────────────
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURES:
        return {
            "error": (
                f"Expected {EXPECTED_FEATURES} features: "
                f"{FEATURE_NAMES}. Got {len(features) if isinstance(features, list) else 'non-list'}."
            ),
            "risk_level": -1,
            "confidence": 0.0,
        }

    # ── validate all values are numeric ──────────────────────────
    try:
        features_array = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return {
            "error": "All features must be numeric values.",
            "risk_level": -1,
            "confidence": 0.0,
        }

    # ── apply scaler if available ─────────────────────────────────
   scaler = _get_scaler()
    if scaler is not None:
        features_array = scaler.transform(features_array)
    else:
        print("[risk_model] WARNING: No scaler found — predictions may be inaccurate")

    # ── predict ───────────────────────────────────────────────────
    try:
        prediction    = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0]
        confidence    = round(float(max(probabilities)), 4)

        # build per-class probability dict dynamically
        classes = model.classes_ if hasattr(model, "classes_") else range(len(probabilities))
        prob_dict = {
            RISK_LABELS.get(int(c), f"Class {c}"): round(float(p), 4)
            for c, p in zip(classes, probabilities)
        }

        return {
            "risk_level":    prediction,
            "risk_label":    RISK_LABELS.get(prediction, "Unknown"),
            "confidence":    confidence,
            "probabilities": prob_dict,
            "features_used": dict(zip(FEATURE_NAMES, features)),
        }

    except Exception as e:
        return {
            "risk_level": -1,
            "confidence": 0.0,
            "error": f"Prediction failed: {str(e)}",
        }
