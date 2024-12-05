from pathlib import Path
import sys
from minio import Minio
import torch
from .minio_utils import get_client

device = "cuda" if torch.cuda.is_available() else "cpu"

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.base.model import TwoTower

weights_dir = repo_dir / "weights"


def load_model(weight_file, use_local=False):
    client = get_client()
    weight_path = str(weights_dir / weight_file)
    if not use_local:
        client.fget_object("img-search", "latest-checkpoint.pt", weight_path)

    model = TwoTower()
    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))

    img_tower = model.img_net
    text_tower = model.text_net

    # save towers individually
    torch.save(img_tower.state_dict(), weights_dir / "img_tower.pt")
    if not use_local:
        client.fput_object("img-search", "img_tower.pt", weights_dir / "img_tower.pt")
    torch.save(text_tower.state_dict(), weights_dir / "text_tower.pt")
    if not use_local:
        client.fput_object("img-search", "text_tower.pt", weights_dir / "text_tower.pt")

    return img_tower, model.img_net.embed_dim
