import torch
import torchvision
from transformers import BertTokenizer

transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
tokeniser = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


def collate_fn(batch):
    captions, pos_imgs, neg_imgs = zip(*batch)

    pos_imgs = torch.stack([transform(img) for img in pos_imgs])
    neg_imgs = torch.stack([transform(img) for img in neg_imgs])

    captions = tokeniser(captions, padding=True, truncation=True, return_tensors="pt")


    return captions, pos_imgs, neg_imgs


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("test.png")

    x = collate_fn(
        [
            ("this is a caption, hello world world asadf", img, img),
            ("a caption", img, img),
        ]
    )

    pass
