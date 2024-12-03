from tqdm import tqdm
from PIL import ImageEnhance
import random
from datasets import load_dataset
from torch.utils.data import Dataset


class FlickrDatatset(Dataset):
    def __init__(
        self,
        split: str = "train",
        split_size: int | float = None,
    ):
        if isinstance(split_size, float):
            if split_size > 1 or split_size < 0:
                raise ValueError("split_size should be between 0 and 1, if float")
            ratio = split_size
        hg_ds = load_dataset(
            "nlphuji/flickr30k",
            split="test",
        ).filter(lambda x: x["split"] == split)

        if isinstance(split_size, int):
            ratio = split_size / len(hg_ds)
        if split_size is not None:
            # Fake random sampling of subset
            hg_ds = hg_ds.train_test_split(test_size=ratio)["test"]
        self.id_to_img = {
            int(row["img_id"]): row["image"]
            for row in tqdm(hg_ds, desc="Preprocessing images")
        }

        self.img_caption_pairs: list[tuple[int, str]] = []
        for row in hg_ds:
            pairs = [(int(row["img_id"]), caption) for caption in row["caption"]]
            self.img_caption_pairs.extend(pairs)

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx):
        img_id, caption = self.img_caption_pairs[idx]
        img = self.id_to_img[img_id]

        neg_id = img_id
        while neg_id == img_id:
            neg_id = random.choice(list(self.id_to_img.keys()))
        neg_img = self.id_to_img[neg_id]

        img = img.rotate(random.uniform(-3, 3))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 2))

        neg_img = neg_img.rotate(random.uniform(-3, 3))
        neg_img = ImageEnhance.Color(neg_img).enhance(random.uniform(0.5, 2))
        return caption, img, neg_img


if __name__ == "__main__":
    dataset = FlickrDatatset(split="val", split_size=0.1)
    pass
