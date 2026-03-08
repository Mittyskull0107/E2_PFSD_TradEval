import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../ml/risk_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

def classify_risk(metrics):

    X = [[
        metrics["avg_return"],
        metrics["volatility"],
        metrics["max_drawdown"]
    ]]

    return model.predict(X)[0]