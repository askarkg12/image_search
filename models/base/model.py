import torch
import torch.nn as nn
from .text_tower import TextEncoder
from .image_tower import ImageEncoder


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_net = ImageEncoder()
        self.text_net = TextEncoder()

    def forward(self, tokens, pos_patches, neg_patches):
        pos_img = self.img_net(pos_patches)
        neg_img = self.img_net(neg_patches)
        text_enc = self.text_net(tokens)
        return text_enc, pos_img, neg_img
