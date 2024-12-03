import os
from minio import Minio

if "MINIO_SERVER" not in os.environ:
    USE_MINIO = False

else:
    USE_MINIO = True

if USE_MINIO:
    if "MINIO_ACCESS_KEY" not in os.environ:
        raise ValueError("MINIO_ACCESS_KEY not found in environment variables")

    try:
        minio_client = Minio(
            os.environ["MINIO_SERVER"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            secure=False,
        )
    except KeyError as e:
        raise Exception(f"Missing environment variable: {e}")


def upload_to_minio(path):
    if USE_MINIO:
        try:
            minio_client.fput_object("img-search", "latest-checkpoint.pt", path)
        except Exception as e:
            print(f"Error uploading to minio: {e}")
            # raise e
