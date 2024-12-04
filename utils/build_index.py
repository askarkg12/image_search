from datasets import load_dataset
from minio_utils import get_client
from minio import Minio
from pathlib import Path
import sys
import faiss
import pickle
import os
from PIL import Image
import torch

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.base.model import TwoTower

weights_dir = repo_dir / "weights"


def build_index(model, embed_dim: int):
    client = get_client()
    id_to_file = {}
    index = faiss.IndexFlatIP(embed_dim)
    image_bucket = "images"
    # Ensure the bucket exists
    if not client.bucket_exists(image_bucket):
        print(f"Bucket '{image_bucket}' does not exist.")
        exit()

    # List objects in the bucket and filter image files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    objects = client.list_objects(image_bucket)
    image_files = [
        obj.object_name
        for obj in objects
        if any(obj.object_name.lower().endswith(ext) for ext in image_extensions)
    ]

    with torch.inference_mode():
        for i, img_name in enumerate(image_files):
            # img = row["image"]
            _, ext = os.path.splitext(img_name)
            img_path = weights_dir / f"current_image{ext}"
            client.fget_object(image_bucket, img_name, img_path)
            img = Image.open(img_path)

            processed_img = model.preprocessing(img).unsqueeze(0)
            img_enc = model(processed_img).squeeze(0)
            index.add(img_enc)

            id_to_file[i] = img_name

    # save faiss index & send to minio
    faiss_path = str(weights_dir / "index.faiss")
    faiss.write_index(index, faiss_path)
    client.fput_object("image-search", "index.faiss", faiss_path)

    # save row-to-filename lookup table
    pickle_path = repo_dir / "weights/id_to_file.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(id_to_file, f)

    client.fput_object("image-search", "id_to_file.pkl", pickle_path)


if __name__ == "__main__":
    m = TwoTower()
    build_index(m.img_net, m.img_net.embed_dim)
