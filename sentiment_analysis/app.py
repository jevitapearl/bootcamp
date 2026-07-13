from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:

    tokenizer = DistilBertTokenizerFast.from_pretrained("model")

    model = DistilBertForSequenceClassification.from_pretrained("model")

    model.to(DEVICE)

    model.eval()

except Exception:

    tokenizer = None
    model = None


class ReviewRequest(BaseModel):

    review: str = Field(
        ...,
        min_length=3,
        description="Customer review text",
    )


@app.get("/")
def home():

    return {
        "message": "Sentiment Analysis API Running"
    }


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


@app.post("/predict")
def predict(data: ReviewRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Prediction model unavailable.",
        )

    try:

        encoded = tokenizer(
            data.review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        with torch.no_grad():

            outputs = model(**encoded)

            prediction = torch.argmax(
                outputs.logits,
                dim=1,
            ).item()

        mapping = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }

        return {
            "review": data.review,
            "prediction": mapping[prediction],
        }

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction.",
        )