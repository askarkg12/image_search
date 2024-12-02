import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_pass(model, criterion, optimizer, batch):
    tokens, pos_patches, neg_patches = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
    )

    text_enc, pos_enc, neg_enc = model(tokens, pos_patches, neg_patches)
    loss = criterion(text_enc, pos_enc, neg_enc)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
