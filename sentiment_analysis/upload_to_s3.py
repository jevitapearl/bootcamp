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


def upload_folder(local_folder, bucket, s3_folder):

    for root, _, files in os.walk(local_folder):

        for file in files:

            local_path = os.path.join(root, file)

            relative_path = os.path.relpath(
                local_path,
                local_folder,
            )

            s3_path = os.path.join(
                s3_folder,
                relative_path,
            ).replace("\\", "/")

            print(f"Uploading {relative_path}")

            s3.upload_file(
                local_path,
                bucket,
                s3_path,
            )

    print("\nUpload Completed Successfully")


if __name__ == "__main__":

    if not os.path.exists(LOCAL_MODEL_DIR):

        raise FileNotFoundError(
            "registered_model folder not found."
        )

    upload_folder(
        LOCAL_MODEL_DIR,
        BUCKET_NAME,
        S3_FOLDER,
    )