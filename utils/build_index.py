from tqdm import tqdm
from minio_utils import get_client
from pathlib import Path
import sys
import faiss
import pickle
from more_itertools import chunked
import os
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.base.model import TwoTower
from utils.load_model import load_model

weights_dir = repo_dir / "weights"

client = get_client()


def build_index(model, embed_dim: int):
    index = faiss.IndexFlatIP(embed_dim)
    image_bucket = "images"

    all_files_doc = repo_dir / "weights/id_to_file.txt"
    faiss_path = str(weights_dir / "index.faiss")

    if all_files_doc.exists():
        with open(all_files_doc, "r") as f:
            processed_files = f.readlines()
    else:
        processed_files = []

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
        if obj.object_name not in processed_files
    ]

    img_chunks = tqdm(chunked(image_files, 100), total=len(image_files) // 100)

    with torch.inference_mode():
        for chunk in img_chunks:
            chunk_enc = []
            chunk_files = []
            for img_name in chunk:
                # img = row["image"]
                _, ext = os.path.splitext(img_name)
                img_path = weights_dir / f"current_image{ext}"
                client.fget_object(image_bucket, img_name, img_path)
                img = Image.open(img_path)

                processed_img = model.preprocessing(img).unsqueeze(0).to(device)
                img_enc = model(processed_img)

                chunk_enc.append(img_enc / img_enc.norm(p=2))
                chunk_files.append(img_name)

            enc_matrix = torch.cat(chunk_enc, dim=0).cpu().numpy()
            index.add(enc_matrix)
            processed_files.extend(chunk_files)

            faiss.write_index(index, faiss_path)

            with open(all_files_doc, "w") as f:
                f.writelines(line + "\n" for line in processed_files)

    client.fput_object("image-search", "index.faiss", faiss_path)
    client.fput_object("image-search", "files_ordered.txt", all_files_doc)


if __name__ == "__main__":
    model, embed_dim = load_model("current_model.pt")
    model = model.to(device)
    model.eval()
    build_index(model, embed_dim)
