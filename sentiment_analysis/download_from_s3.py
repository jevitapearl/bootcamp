import os
import boto3

from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME,
)

LOCAL_MODEL_DIR = "registered_model"
S3_FOLDER = "registered_model"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)


def download_model():

    # Model already exists locally
    if os.path.exists(LOCAL_MODEL_DIR):
        print("Model already available locally.")
        return

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=S3_FOLDER,
    )

    if "Contents" not in response:
        raise Exception("No model found in S3.")

    for obj in response["Contents"]:

        key = obj["Key"]

        # Skip folder entry
        if key.endswith("/"):
            continue

        relative_path = os.path.relpath(key, S3_FOLDER)

        local_path = os.path.join(
            LOCAL_MODEL_DIR,
            relative_path,
        )

        os.makedirs(
            os.path.dirname(local_path),
            exist_ok=True,
        )

        print(f"Downloading {key}")

        s3.download_file(
            BUCKET_NAME,
            key,
            local_path,
        )

    print("Model downloaded successfully.")