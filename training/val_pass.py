import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def val_pass(model, criterion, val_dl):
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch in val_dl:
            tokens, pos_patches, neg_patches = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

            text_enc, pos_enc, neg_enc = model(tokens, pos_patches, neg_patches)
            loss = criterion(text_enc, pos_enc, neg_enc)
            val_loss += loss.item()

    return val_loss / len(val_dl)
