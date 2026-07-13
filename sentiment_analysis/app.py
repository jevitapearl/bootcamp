from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from download_from_s3 import download_model

# ---------------------------------------------------
# Download model from S3 (if not present)
# ---------------------------------------------------

download_model()

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

try:

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "registered_model"
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "registered_model"
    )

    model.to(DEVICE)
    model.eval()

    print("Model Loaded Successfully")

except Exception as e:

    tokenizer = None
    model = None

    print(e)


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------

class ReviewRequest(BaseModel):

    review: str = Field(
        ...,
        min_length=3,
        description="Customer review",
    )


# ---------------------------------------------------
# Home
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Sentiment Analysis API Running"
    }


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "Unhealthy",
            "model": "Not Loaded",
        }

    return {
        "status": "Healthy",
        "model": "Loaded",
    }


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(data: ReviewRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model not available",
        )

    encoded = tokenizer(
        data.review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(**encoded)

        prediction = torch.argmax(
            outputs.logits,
            dim=1,
        ).item()

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }

    return {
        "review": data.review,
        "prediction": labels[prediction],
    }