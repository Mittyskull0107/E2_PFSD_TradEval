import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("training_data.csv")

X = df[["avg_return","volatility","max_drawdown"]]

y = df["risk"]

model = RandomForestClassifier()

model.fit(X,y)

joblib.dump(model,"risk_classifier.pkl")