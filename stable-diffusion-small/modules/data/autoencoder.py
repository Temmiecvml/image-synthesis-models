import os
import sys
from functools import partial

import lightning as L
import numpy as np
import torch
import pickle
from datasets import Dataset as HfDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def preprocess_celebahq_caption(sample, transform):
    prefix = "a photography of"
    image = transform(sample["image"])
    text = sample["text"].lower().removeprefix(prefix).strip()

    return image, text


def collate_celebahq_caption(samples):
    images, texts = zip(*samples)

    images = torch.stack(images)
    texts = np.stack(texts)

    return images, texts


class PytorchHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, preprocess_fn):
        """
        Custom Dataset to apply transformations in batch.

        Args:
            hf_dataset: Hugging Face dataset.
            preprocess_fn: Function to preprocess a batch of data.
        """
        self.hf_dataset = hf_dataset
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        data_point = self.hf_dataset[idx]
        image, text = self.preprocess_fn(data_point)

        return image, text


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


class AutoEncoderDataModule:
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
        """For downloading the dataset"""
        self.dataset = load_dataset(
            self.data_path,
            split="train",
            cache_dir=self.cache_dir,
        )

        dataset = self.dataset.train_test_split(
            test_size=self.train_val_split, shuffle=True, seed=32
        )

        with open("train.pkl", "wb") as file:
            pickle.dump(dataset["train"], file)

        with open("val.pkl", "wb") as file:
            pickle.dump(dataset["test"], file)

    def setup(self, stage: str):
        """For loading and processing the dataset

        Huggingface API for downloading and loading is the same
        """
        
        with open("train.pkl", "rb") as file:
            train_ds = pickle.load(file)

        with open("val.pkl", "rb") as file:
            val_ds = pickle.load(file)

        if stage == "fit":
            self.train_ds = PytorchHuggingFaceDataset(
                train_ds, self.preprocess_batch
            )
            self.val_ds = PytorchHuggingFaceDataset(
                val_ds, self.preprocess_batch
            )

        if stage == "test":
            self.test_ds = PytorchHuggingFaceDataset(
                val_ds, self.preprocess_batch
            )

    def train_dataloader(self, **kwargs):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            **kwargs,
        )

    def val_dataloader(self, **kwargs):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            **kwargs,
        )

    def test_dataloader(self, **kwargs):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            **kwargs,
        )
