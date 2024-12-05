from pathlib import Path
import json
import faiss
from minio import Minio
from dotenv import dotenv_values
from model import TextEncoder
import torch

config = dotenv_values(".env")

minio_client = Minio(
    config["MINIO_SERVER"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False,
)

bucket = config["MINIO_BUCKET"]

FAISS_INDEX_NAME = "index.faiss"
ID_LOOKUP_NAME = Path("filenames.txt")
WEIGHTS_NAME = Path("txt-encoder.pt")


def make_local_vol(name):
    return "local/" + name


minio_client.fget_object(bucket, FAISS_INDEX_NAME, make_local_vol(FAISS_INDEX_NAME))
minio_client.fget_object(bucket, ID_LOOKUP_NAME, make_local_vol(ID_LOOKUP_NAME))
minio_client.fget_object(bucket, ID_LOOKUP_NAME, make_local_vol(WEIGHTS_NAME))

faiss_index: faiss.IndexFlatIP = faiss.read_index(str(FAISS_INDEX_NAME))

with open(ID_LOOKUP_NAME, "r") as f:
    filenames = f.readlines()


txt_encoder = TextEncoder()
txt_encoder.load_state_dict(torch.load(WEIGHTS_NAME))


def top_k_images(query, k=20):
    encoding = txt_encoder(query)
    D, I = faiss_index.search(encoding, k)
    return [(d, filenames[i]) for d, i in zip(D[0], I[0])]
