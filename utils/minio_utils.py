from minio import Minio
from dotenv import load_dotenv
import os

load_dotenv()

# minio_addr = os.getenv("MINIO_ADDRESS")
# access_key = os.getenv("MINIO_ROOT_USER")
# secret_key = os.getenv("MINIO_PASSWORD")
minio_addr = "localhost:9000"
access_key = "admin"
secret_key = "development"


def get_client():
    client = Minio(
        minio_addr,  # Your MinIO server URL (host:port)
        access_key=access_key,
        secret_key=secret_key,
        secure=False,  # Use True for HTTPS, False for HTTP
    )
    return client
