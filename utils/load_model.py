from pathlib import Path
import sys
from minio import Minio
import torch
from .minio_utils import get_client

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.base.model import TwoTower

weights_dir = repo_dir / "weights"


def load_model(weight_file):
    client = get_client()
    weight_path = str(weights_dir / weight_file)
    client.fget_object("image-search", "latest-checkpoint.pt", weight_path)

    model = TwoTower()
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    img_tower = model.img_net
    text_tower = model.text_net

    # save towers individually
    torch.save(img_tower.state_dict(), weights_dir / "img_tower.pt")
    client.fput_object("image-search", "img_tower.pt", weights_dir / "img_tower.pt")
    torch.save(text_tower.state_dict(), weights_dir / "text_tower.pt")
    client.fput_object("image-search", "text_tower.pt", weights_dir / "text_tower.pt")

    return img_tower, model.img_net.embed_dim
