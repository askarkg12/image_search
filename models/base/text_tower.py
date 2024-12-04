from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")

    def forward(self, tokens):
        inpud_ids = tokens.data["input_ids"]
        mask = tokens.data["attention_mask"]
        out = self.bert(inpud_ids, mask)
        return out.pooler_output


if __name__ == "__main__":
    model = TextEncoder()
