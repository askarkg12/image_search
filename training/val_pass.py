import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def val_pass(model, criterion, batch):
    with torch.inference_mode():
        model.eval()
        tokens, pos_patches, neg_patches = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
        )

        text_enc, pos_enc, neg_enc = model(tokens, pos_patches, neg_patches)
        loss = criterion(text_enc, pos_enc, neg_enc)

        return loss.item()
