from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

try:
    model = joblib.load("/home/student/bootcamp/fastAPI_project/load_model.joblib")

except:
    model = None
class LoanRequest(BaseModel):
    Age: int = Field(..., ge=18, le=60, description="Age between 18 and 60")
    Salary: float = Field(..., gt=10000, description="Salary greater than 10000")

@app.get("/")
def home():
    return {
        "message": "Loan Prediction API is Running"
    }

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "Unhealthy",
            "model": "Not Loaded"
        }
    return {
        "status": "Healthy",
        "model": "Loaded"
    }

@app.post("/predict")
def predict(data: LoanRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model is not available"
        )
    
    if data.Salary > 1000000:
        raise HTTPException(
            status_code = 400,
            detail="Salary seems unrealistic."
        )

    try:
        input_data = [[data.Age, data.Salary]]
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"
        return {
            "prediction": result
        }
    except Exception:
        raise HTTPException(
            status_code=501,
            detail="Unable to generate prediction."
        )