"""Text and Image datasets for multi-label genre classification."""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import IMG_ROOT, IMG_SIZE


class TextDataset(Dataset):
    """Tokenized text dataset for transformer models."""

    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = list(texts)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class ImageDataset(Dataset):
    """Screenshot dataset with standard ImageNet preprocessing."""

    def __init__(self, df, labels, split="train", is_train=True, size=IMG_SIZE):
        self.root = os.path.join(IMG_ROOT, split)
        self.fnames = df["image"].fillna("").values
        self.labels = labels

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        path = os.path.join(self.root, self.fnames[i])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y
