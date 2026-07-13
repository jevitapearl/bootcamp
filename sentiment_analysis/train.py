import os
import shutil

import mlflow
import pandas as pd

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# MLflow Configuration

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis DistilBERT")

# Load Dataset

df = pd.read_csv("product_reviews_mock_data.csv")


def convert_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    return 2


df["label"] = df["Rating"].apply(convert_label)
df = df[["ReviewText", "label"]]

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
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
    columns=["input_ids", "attention_mask", "label"],
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
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

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Hyperparameter Configurations

configs = [
    {
        "name": "run_1",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "run_2",
        "learning_rate": 3e-5,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "run_3",
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 16,
    },
]

# Best Model Tracking

best_f1 = -1
best_run = None

# Training Loop

for config in configs:

    print(f"\nTraining {config['name']}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
    )

    training_args = TrainingArguments(
        output_dir=f"results/{config['name']}",
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"logs/{config['name']}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=config["name"]):

        mlflow.log_params(config)

        trainer.train()

        metrics = trainer.evaluate()

        mlflow.log_metrics(metrics)

        current_f1 = metrics["eval_f1"]

        print(f"{config['name']} F1 Score : {current_f1:.4f}")

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_run = config["name"]

            if os.path.exists("registered_model"):
                shutil.rmtree("registered_model")

            trainer.save_model("registered_model")
            tokenizer.save_pretrained("registered_model")

            mlflow.log_artifacts(
                "registered_model",
                artifact_path="best_model",
            )

# Summary

print("\nTraining Completed")
print(f"Best Run : {best_run}")
print(f"Best F1  : {best_f1:.4f}")
print("\nBest model saved to registered_model/")