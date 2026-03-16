"""
TradeEval – Risk Classifier Training Script
Run from the ml/ directory:  python train_model.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────
ML_DIR     = Path(__file__).resolve().parent
DATA_DIR   = ML_DIR.parent / "Data"
MODEL_DIR  = ML_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH    = MODEL_DIR / "risk_classifier.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

print(f"Data directory : {DATA_DIR}")
print(f"Model output   : {MODEL_PATH}")

# ── Load CSVs ──────────────────────────────────────────────────────
csv_files = [
    f for f in DATA_DIR.glob("*.csv")
    if "NIFTY" not in f.name and "metadata" not in f.name
]

if not csv_files:
    print("ERROR: No CSV files found in Data/")
    sys.exit(1)

print(f"Loading {len(csv_files)} stock CSVs ...")

frames = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        df = df.loc[:, ~df.columns.str.lower().isin(["symbol"])]
        df["symbol"] = f.stem
        frames.append(df)
    except Exception as e:
        print(f"  Skipping {f.name}: {e}")

if not frames:
    print("ERROR: No data loaded.")
    sys.exit(1)

data = pd.concat(frames, ignore_index=True)
print(f"Total rows loaded: {len(data)}")

# ── Normalise columns ──────────────────────────────────────────────
data.columns = [c.strip().lower() for c in data.columns]

if "close" not in data.columns:
    print(f"ERROR: No 'close' column. Available: {list(data.columns)}")
    sys.exit(1)

# ── Sort & clean ───────────────────────────────────────────────────
sort_cols = ["symbol", "date"] if "date" in data.columns else ["symbol"]
data = data.sort_values(sort_cols).reset_index(drop=True)
data["close"] = pd.to_numeric(data["close"], errors="coerce")
data.dropna(subset=["close"], inplace=True)

# ── Feature engineering ────────────────────────────────────────────
# These MUST match FEATURE_NAMES in risk_model.py exactly
data["return_1d"]  = data.groupby("symbol")["close"].pct_change(1)
data["return_5d"]  = data.groupby("symbol")["close"].pct_change(5)
data["return_20d"] = data.groupby("symbol")["close"].pct_change(20)

data["volatility"] = data.groupby("symbol")["return_1d"].transform(
    lambda x: x.rolling(20, min_periods=5).std()
)

# ── NEW: Sharpe ratio (20-day rolling) ────────────────────────────
data["sharpe_20d"] = data.groupby("symbol")["return_1d"].transform(
    lambda x: (
        x.rolling(20, min_periods=5).mean() /
        x.rolling(20, min_periods=5).std().replace(0, np.nan)
    ) * np.sqrt(252)
)

# ── NEW: Max drawdown (20-day rolling) ────────────────────────────
def rolling_max_drawdown(series, window=20):
    def mdd(x):
        peak = np.maximum.accumulate(x)
        dd   = (x - peak) / np.where(peak == 0, np.nan, peak)
        return dd.min()
    return series.rolling(window, min_periods=5).apply(mdd, raw=True)

data["max_drawdown_20d"] = data.groupby("symbol")["close"].transform(
    rolling_max_drawdown
)

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility",
    "sharpe_20d",
    "max_drawdown_20d",
]

data.dropna(subset=FEATURES, inplace=True)
print(f"Rows after feature engineering: {len(data)}")

if len(data) < 100:
    print("ERROR: Not enough data rows to train.")
    sys.exit(1)

# ── Risk labels (volatility + return combined) ─────────────────────
# Score = high volatility + negative return = highest risk
data["risk_score"] = (
    data["volatility"].rank(pct=True) * 0.6 +          # 60% weight on volatility
    (-data["return_20d"]).rank(pct=True) * 0.4          # 40% weight on poor return
)

data["risk_label"] = pd.qcut(
    data["risk_score"], q=3, labels=[0, 1, 2]           # even 33/33/33 split
).astype(int)

X = data[FEATURES].values
y = data["risk_label"].values

print(f"Training samples  : {len(X)}")
print(f"Class distribution: Low={int(np.sum(y==0))}  "
      f"Medium={int(np.sum(y==1))}  High={int(np.sum(y==2))}")

# ── Scale features ─────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train with cross-validation ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # prevent overfitting
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# cross-validation for honest accuracy
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
print(f"\nCross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

y_pred = model.predict(X_test)
report = classification_report(
    y_test, y_pred,
    target_names=["Low", "Medium", "High"],
    output_dict=True
)
print("\n── Classification Report ──────────────────────")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# ── Feature importance ─────────────────────────────────────────────
importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
print("── Feature importances ────────────────────────")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {feat:<22} {imp:.4f}")

# ── Save model + scaler + metadata ────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

metadata = {
    "trained_at":       datetime.now().isoformat(),
    "features":         FEATURES,           # ← single source of truth
    "num_features":     len(FEATURES),
    "training_samples": len(X_train),
    "cv_accuracy":      round(cv_scores.mean(), 4),
    "cv_std":           round(cv_scores.std(), 4),
    "class_report":     report,
    "feature_importances": importances,
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel  saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
print(f"Metadata     → {METADATA_PATH}")
print(f"File size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
