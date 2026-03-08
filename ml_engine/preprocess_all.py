import pandas as pd
import numpy as np
import os

# =========================
# 1. PATH CONFIGURATION
# =========================
DATA_PATH = "E:/PFSD/Project/Data"     # <-- CHANGE if needed
OUTPUT_PATH = "ml_engine/data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# 2. LOAD ALL CSV FILES
# =========================
all_data = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".csv"):
        symbol = file.replace(".csv", "")
        file_path = os.path.join(DATA_PATH, file)

        df = pd.read_csv(file_path)
        df["Symbol"] = symbol
        all_data.append(df)

# Combine all stocks
df = pd.concat(all_data, ignore_index=True)

print("✔ All CSV files loaded")
print(df.head())

# =========================
# 3. BASIC CLEANING
# =========================
df.dropna(inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by=["Symbol", "Date"])

# =========================
# 4. FEATURE ENGINEERING
# =========================

# Daily returns
df["daily_return"] = df.groupby("Symbol")["Close"].pct_change()

# Rolling volatility (14 days)
df["volatility"] = (
    df.groupby("Symbol")["daily_return"]
    .rolling(window=14)
    .std()
    .reset_index(0, drop=True)
)

# Rolling average return (14 days)
df["avg_return"] = (
    df.groupby("Symbol")["daily_return"]
    .rolling(window=14)
    .mean()
    .reset_index(0, drop=True)
)

# Cumulative returns
df["cum_return"] = df.groupby("Symbol")["daily_return"].cumsum()

# Maximum drawdown
df["max_drawdown"] = (
    df.groupby("Symbol")["cum_return"].cummax() - df["cum_return"]
)

# =========================
# 5. RISK LABEL CREATION
# =========================
def label_risk(row):
    if row["max_drawdown"] > 0.15 or row["volatility"] > 0.03:
        return "High"
    elif row["volatility"] > 0.02:
        return "Medium"
    else:
        return "Low"

df["risk"] = df.apply(label_risk, axis=1)

# =========================
# 6. FINAL ML DATASET
# =========================
ml_df = df[
    ["avg_return", "volatility", "max_drawdown", "risk"]
].dropna()

# =========================
# 7. SAVE OUTPUT
# =========================
output_file = os.path.join(OUTPUT_PATH, "final_ml_dataset.csv")
ml_df.to_csv(output_file, index=False)

print("✅ Preprocessing complete")
print(f"📁 Saved to: {output_file}")
print(ml_df.head())