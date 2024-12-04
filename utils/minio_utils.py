from minio import Minio
from dotenv import dotenv_values as doten

ENV_FILE = ".env"

minio_config = doten(ENV_FILE)

if "MINIO_SERVER" not in minio_config:
    USE_MINIO = False

else:
    USE_MINIO = True

if USE_MINIO:
    if "MINIO_ACCESS_KEY" not in minio_config:
        raise ValueError("MINIO_ACCESS_KEY not found in environment variables")

    try:
        minio_client = Minio(
            minio_config["MINIO_SERVER"],
            access_key=minio_config["MINIO_ACCESS_KEY"],
            secret_key=minio_config["MINIO_SECRET_KEY"],
            secure=False,
        )
    except KeyError as e:
        raise Exception(f"Missing environment variable: {e}")


def upload_checkpoint_minio(path):
    if USE_MINIO:
        try:
            minio_client.fput_object("img-search", "latest-checkpoint.pt", path)
        except Exception as e:
            print(f"Error uploading to minio: {e}")


def get_client():
    client = Minio(
        minio_config["MINIO_SERVER"],
        access_key=minio_config["MINIO_ACCESS_KEY"],
        secret_key=minio_config["MINIO_SECRET_KEY"],
        secure=False,
    )
    return client


if __name__ == "__main__":
    pass
