import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load dataset
df = pd.read_csv("dataset.csv")

#Features
X = df[["Age", "Salary"]]

#Target
y = df["Approved"]

#Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

#Save model
joblib.dump(model, "load_model.joblib")

print("Model saved")