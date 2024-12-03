import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import wandb
import threading
from pathlib import Path


repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.base.model import TwoTower
from models.base.collate import collate_fn
from training.train_pass import train_pass
from training.val_pass import val_pass
from flickr_dataset import FlickrDatatset


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path(__file__).parent.parent / "weights"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
LAST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"

USE_WANDB = True
BATCH_SIZE = 32
VAL_BATCH_SIZE = 48
VAL_EVERY_N_DATA = 5000
CHECKPOINTS_PERIOD = 500
epoch = 0
datapoint_counter = 0  # Doesnt have to start from 0 if loaded from checkpoint
val_datapoint_counter = 0
train_ds = FlickrDatatset(split="train")
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

train_iter = iter(train_dl)

val_ds = FlickrDatatset(split="val")
val_dl = DataLoader(
    val_ds, batch_size=VAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)


model = TwoTower().to(device)

if LAST_CHECKPOINT.exists():
    model.load_state_dict(torch.load(LAST_CHECKPOINT))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
)

if USE_WANDB:
    config = {
        "model": "BaselineImgCaptionGen",
        "batch_size": BATCH_SIZE,
        "checkpoint_period": CHECKPOINTS_PERIOD,
        "lr": 1e-4,
    }
    wandb.init(project="img_caption", name="two-tower", config=config)

while True:
    try:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_dl)
            continue

        datapoint_counter += len(batch)
        val_datapoint_counter += len(batch)

        loss = train_pass(model, criterion, optimizer, batch)
        wandb.log(
            {
                "training_loss": loss,
                "epoch": epoch,
                "batch_num": datapoint_counter,
            }
        )

        if val_datapoint_counter >= VAL_EVERY_N_DATA:
            val_loss = val_pass(model, criterion, val_ds)
            wandb.log(
                {
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "batch_num": datapoint_counter,
                }
            )
            val_datapoint_counter = 0

            torch.save(model.state_dict(), LAST_CHECKPOINT)
            model.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            BATCH_SIZE = int(BATCH_SIZE * 0.9)
            train_dl = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=train_ds.collate_fn,
            )
            train_iter = iter(train_dl)
            continue
        else:
            raise e
