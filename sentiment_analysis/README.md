# Sentiment Analysis using DistilBERT

A production-ready NLP project that fine-tunes **DistilBERT** for multi-class sentiment analysis of product reviews. The project uses **MLflow** for experiment tracking, **AWS S3** for model storage, and **Docker** for containerized deployment.

---

## Features

- Fine-tune DistilBERT for sentiment classification
- Convert product ratings into sentiment labels
- Compare multiple hyperparameter configurations
- Track experiments with MLflow
- Automatically save the best-performing model
- Upload trained models to AWS S3
- Download models from AWS S3 when needed
- Docker support for deployment
- Reproducible training pipeline

---

## Tech Stack

- Python 3.11
- Hugging Face Transformers
- PyTorch
- Datasets
- Scikit-learn
- Pandas
- MLflow
- AWS S3 (Boto3)
- Docker
- FastAPI
- Uvicorn

---

## Project Structure

```
sentiment_analysis/
│
├── product_reviews_mock_data.csv
├── train.py                  # Train DistilBERT model
├── upload_to_s3.py           # Upload trained model to AWS S3
├── download_from_s3.py         # Download model from AWS S3
├── config.py                 # AWS configuration
├── app.py                    # FastAPI application
├── requirements.txt
├── Dockerfile
├── .env
│
├── registered_model/         # Saved best model
├── mlruns/                   # MLflow experiments
├── logs/
├── results/
│
└── README.md
```

---

## Dataset

The dataset contains product reviews and ratings.

| Column | Description |
|---------|-------------|
| ReviewText | Product review |
| Rating | Rating from 1–5 |

Ratings are mapped to sentiment labels:

| Rating | Label | Sentiment |
|---------|------:|-----------|
| 1–2 | 0 | Negative |
| 3 | 1 | Neutral |
| 4–5 | 2 | Positive |

---

## Model

- **Architecture:** DistilBERT
- **Pretrained Model:** `distilbert-base-uncased`
- **Task:** Multi-class Sentiment Classification
- **Maximum Sequence Length:** 128 tokens

---

## Training Pipeline

1. Load dataset
2. Convert ratings into sentiment labels
3. Split data into training and testing sets
4. Tokenize text using the DistilBERT tokenizer
5. Fine-tune DistilBERT
6. Evaluate model performance
7. Log metrics with MLflow
8. Save the best-performing model
9. Upload the model to AWS S3

---

## Hyperparameter Configurations

| Run | Learning Rate | Epochs | Batch Size |
|-----|---------------|--------|------------|
| run_1 | 2e-5 | 3 | 16 |
| run_2 | 3e-5 | 3 | 16 |
| run_3 | 5e-5 | 3 | 16 |

The model with the highest weighted F1-score is automatically selected.

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1-score

---

## MLflow Experiment Tracking

Each training run logs:

- Hyperparameters
- Evaluation metrics
- Best model artifacts

The experiments are stored locally in:

```
mlruns/
```

---

## AWS S3 Integration

After training, the best model can be uploaded to an Amazon S3 bucket.

### Upload Model

```bash
python upload_to_s3.py
```

### Download Model

```bash
python download_from_s3.py
```

The download script checks whether the model already exists locally before downloading it from S3.

---

## Environment Variables

Create a `.env` file in the project root.

```env
AWS_ACCESS_KEY=AWS_access_key
AWS_SECRET_KEY=AWS_secret_key
AWS_REGION=region
BUCKET_NAME=bucket_name
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/jevitapearl/bootcamp.git
```

Navigate to the project.

```bash
cd bootcamp/sentiment_analysis
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Training Script

```bash
python train.py
```

After training:

- The best model is stored in `registered_model/`
- MLflow logs are saved in `mlruns/`
- Training logs are stored in `logs/`
- Checkpoints are saved in `results/`

---

## Docker

Build the Docker image.

```bash
docker build -t sentiment-analysis .
```

Run the container.

```bash
docker run -p 8000:8000 sentiment-analysis
```

The FastAPI application will be available at:

```
http://localhost:8000
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Hyperparameter optimization with Optuna
- Early stopping
- Model Registry integration with MLflow
- CI/CD pipeline
- Kubernetes deployment
- Support for larger transformer models such as BERT and RoBERTa

---

## Author

**Jevita Pearl**

---

## License

This project is intended for educational and learning purposes.
