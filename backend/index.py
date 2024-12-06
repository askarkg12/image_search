from pathlib import Path
import json
import faiss
from minio import Minio
from dotenv import dotenv_values
from model import TextEncoder
import torch
from transformers import BertTokenizer

device = "cpu"

config = dotenv_values(".env")

minio_client = Minio(
    config["MINIO_SERVER"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False,
)

bucket = config["MINIO_BUCKET"]

FAISS_INDEX_NAME = "index.faiss"
ID_LOOKUP_NAME = "filenames.txt"
WEIGHTS_NAME = "text_tower.pt"


def make_local_vol(name):
    return "local/" + name


minio_client.fget_object(bucket, FAISS_INDEX_NAME, make_local_vol(FAISS_INDEX_NAME))
minio_client.fget_object(bucket, ID_LOOKUP_NAME, make_local_vol(ID_LOOKUP_NAME))
minio_client.fget_object(bucket, WEIGHTS_NAME, make_local_vol(WEIGHTS_NAME))

faiss_index: faiss.IndexFlatIP = faiss.read_index(make_local_vol(FAISS_INDEX_NAME))

with open(make_local_vol(ID_LOOKUP_NAME), "r") as f:
    filenames = f.readlines()


txt_encoder = TextEncoder()
txt_encoder.load_state_dict(
    torch.load(make_local_vol(WEIGHTS_NAME), map_location=device)
)

tokeniser = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


def top_k_images(query, k=20):
    with torch.inference_mode():
        tokens = tokeniser(query, return_tensors="pt")
        encoding = txt_encoder(tokens)
        D, I = faiss_index.search(encoding, k)
        return [filenames[i] for d, i in zip(D[0], I[0])]
