import os
import mlflow
import mlflow.pytorch
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# MLFlow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis DistilBERT")

# Load Dataset
df = pd.read_csv("product_reviews_mock_data.csv")

# Rating -> Sentiment
# 1,2 = Negative (0)
# 3 = Neutral (1)
# 4,5 = Positive (2)
def convert_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

df["label"] = df["Rating"].apply(convert_label)

df = df[["ReviewText", "label"]]

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["ReviewText"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Model

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
)

# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Training

# Training Configurations

configs = [
    {"name": "run_1", "lr": 2e-5, "epochs": 3, "batch_size": 16},
    {"name": "run_2", "lr": 3e-5, "epochs": 3, "batch_size": 16},
    {"name": "run_3", "lr": 5e-5, "epochs": 3, "batch_size": 16}
]
# MLFlow Logging

for config in configs:

    print(f"Training {config['name']}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
    )

    training_args = TrainingArguments(
        output_dir=f"./results/{config['name']}",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"./logs/{config['name']}",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=config["name"]):

        mlflow.log_params({
            "model": "DistilBERT",
            "learning_rate": config["lr"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
        })

        trainer.train()

        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        model_path = f"model/{config['name']}"
        os.makedirs(model_path, exist_ok=True)

        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        mlflow.log_artifacts(model_path, artifact_path="model")

print("All training runs completed.")