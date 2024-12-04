from tqdm import tqdm
from minio_utils import minio_client
from datasets import load_dataset


def save_from_ds():
    ds = load_dataset("nlphuji/flickr30k", split="test[:10]")

    id_to_filename = {}

    for i, row in enumerate(tqdm(ds)):
        img = row["image"]

        img.save("temp_file.png")

        img_name = row["filename"]
        id_to_filename[i] = img_name

        minio_client.fput_object("images", img_name, "temp_file.png")


def save_from_dir(directory):
    import os

    for i, file in enumerate(tqdm(os.listdir(directory))):
        if file.endswith(".jpg") or file.endswith(".png"):
            minio_client.fput_object("images", file, f"{directory}/{file}")


if __name__ == "__main__":
    dir = (
        "/home/askar/.cache/huggingface/datasets/downloads/extracted/"
        "28897b4e2f6ed8622eb38c8696f4c2f05e42e67c1ec12e1a535fd160c0fc5091/"
        "flickr30k-images"
    )
    save_from_dir(dir)
