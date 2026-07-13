# import joblib
# model = joblib.load("model.pkl")

import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/breast_cancer_model/1")


