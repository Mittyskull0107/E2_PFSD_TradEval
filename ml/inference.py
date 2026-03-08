import joblib

model = joblib.load("ml/model/risk_classifier.pkl")

def predict(features):
    return model.predict([features])[0]