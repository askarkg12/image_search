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
from utils.minio_utils import upload_checkpoint_minio


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path(__file__).parent.parent / "weights"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
LAST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"

USE_WANDB = True
BATCH_SIZE = 32
VAL_BATCH_SIZE = 48
MINI_VAL_PERIOD = 100
CHECKPOINTS_PERIOD = 2000

USE_MINIO = True

epoch = 0
datapoint_counter = 0
val_datapoint_counter = 0
checkpoint_val_counter = 0

train_ds = FlickrDatatset(split="train", split_size=0.1)
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

train_iter = iter(train_dl)

val_ds = FlickrDatatset(split="val", split_size=0.1)
val_dl = DataLoader(
    val_ds, batch_size=VAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

val_iter = iter(val_dl)


model = TwoTower().to(device)

if LAST_CHECKPOINT.exists():
    model.load_state_dict(torch.load(LAST_CHECKPOINT))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1)
)

if USE_WANDB:
    wandb.init(project="img_search", name="two-tower")

try:
    batch = next(val_iter)
except StopIteration:
    val_iter = iter(val_dl)
val_loss = val_pass(model, criterion, batch)
wandb.log(
    {
        "mini_val_loss": val_loss,
        "epoch": epoch,
        "data_counter": datapoint_counter,
    }
)

big_val_loss = []
for batch in val_dl:
    big_val_loss.append(val_pass(model, criterion, batch))

val_loss = sum(big_val_loss) / len(big_val_loss)

wandb.log(
    {
        "big_val_loss": val_loss,
        "epoch": epoch,
        "data_counter": datapoint_counter,
    }
)
while True:
    try:
        if val_datapoint_counter >= MINI_VAL_PERIOD:
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dl)
                continue
            val_loss = val_pass(model, criterion, batch)
            wandb.log(
                {
                    "mini_val_loss": val_loss,
                    "epoch": epoch,
                    "data_counter": datapoint_counter,
                }
            )
            val_datapoint_counter = 0
            continue
        elif checkpoint_val_counter >= CHECKPOINTS_PERIOD:
            big_val_loss = []
            for batch in val_dl:
                big_val_loss.append(val_pass(model, criterion, batch))

            val_loss = sum(big_val_loss) / len(big_val_loss)

            wandb.log(
                {
                    "big_val_loss": val_loss,
                    "epoch": epoch,
                    "data_counter": datapoint_counter,
                }
            )

            torch.save(model.state_dict(), LAST_CHECKPOINT)

            if USE_MINIO:
                threading.Thread(
                    target=upload_checkpoint_minio, args=(LAST_CHECKPOINT,)
                ).start()

            checkpoint_val_counter = 0
            continue

        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_dl)
            continue

        loss = train_pass(model, criterion, optimizer, batch)
        wandb.log(
            {
                "training_loss": loss,
                "epoch": epoch,
                "data_counter": datapoint_counter,
            }
        )
        datapoint_counter += len(batch)
        val_datapoint_counter += len(batch)
        checkpoint_val_counter += len(batch)

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
