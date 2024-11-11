import sys
from functools import partial

import lightning.pytorch as pl
import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def preprocess_celebahq_caption(samples, transform):
    if not samples["text"] or not samples["image"]:
        return torch.Tensor([]), np.array([])

    prefix = "a photography of"
    samples["text"] = [i.lower().removeprefix(prefix).strip() for i in samples["text"]]
    samples["image"] = [transform(i) for i in samples["image"]]

    texts = np.stack(samples["text"])
    images = torch.stack(samples["image"])

    return images, texts


def collate_celebahq_caption(samples):
    images = torch.stack([sample["image"] for sample in samples])
    texts = np.stack([sample["text"] for sample in samples])

    return images, texts


class BatchHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, preprocess_fn, batch_size):
        """
        Custom Dataset to apply transformations in batch.

        Args:
            hf_dataset: Hugging Face dataset.
            preprocess_fn: preprocessing function.
            batch_size: Number of items in each batch.
        """
        self.hf_dataset = hf_dataset
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.hf_dataset))
        batch = self.hf_dataset[start_idx:end_idx]
        images, texts = self.preprocess_fn(batch)

        return images, texts


class CustomResizeAndCrop:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        if width > height:
            scale = self.target_size / width
        else:
            scale = self.target_size / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = transforms.Resize((new_height, new_width))(image)
        final_image = transforms.CenterCrop(self.target_size)(resized_image)

        return final_image


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seed: int,
        buffer_size: int,
        batch_size: int,
        image_size: int,
        num_workers: int,
        preprocess_batch_fn: str,
        collate_fn: str,
        train_val_split: int,
        cache_dir: str = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.cache_dir = cache_dir
        # https://pytorch.org/docs/stable/data.html
        # https://github.com/pytorch/pytorch/issues/13246

        self.transform = transforms.Compose(
            [
                CustomResizeAndCrop(target_size=image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        preprocess_batch = getattr(sys.modules[__name__], preprocess_batch_fn, None)
        self.preprocess_batch = partial(preprocess_batch, transform=self.transform)
        self.collate_fn = getattr(sys.modules[__name__], collate_fn, None)

    def prepare_data(self):
        self.dataset = load_dataset(
            self.data_path,
            split="train",
            cache_dir=self.cache_dir,
        )

    def setup(self, stage: str):
        if not hasattr(self, "dataset"):
            self.prepare_data()

        if stage == "fit":
            dataset = self.dataset.train_test_split(
                test_size=self.train_val_split, shuffle=True, seed=32
            )
            self.train_ds = BatchHuggingFaceDataset(
                dataset["train"], self.preprocess_batch, self.batch_size
            )
            self.val_ds = BatchHuggingFaceDataset(
                dataset["test"], self.preprocess_batch, self.batch_size
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
