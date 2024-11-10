import sys
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def preprocess_celebahq_caption(samples, transform):
    prefix = "a photography of"
    samples["text"] = [i.lower().removeprefix(prefix).strip() for i in samples["text"]]
    samples["image"] = [transform(i) for i in samples["image"]]

    return samples


def collate_celebahq_caption(samples):
    images = torch.stack([sample["image"] for sample in samples])
    texts = np.stack([sample["text"] for sample in samples])

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
        val_data_size: int,
        cache_dir: str = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_data_size = val_data_size
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
        self.train_dataset = load_dataset(
            self.data_path,
            split="train",
            streaming=True,
            cache_dir=self.cache_dir,
        ).skip(self.val_data_size)

        self.val_dataset = load_dataset(
            self.data_path,
            split="train",
            streaming=True,
            cache_dir=self.cache_dir,
        ).take(self.val_data_size)

    def setup(self, stage: str):
        self.prepare_data() # should be removed
        if stage == "fit":
            train_shuffled_dataset = self.train_dataset.shuffle(
                seed=self.seed,
                buffer_size=self.buffer_size,
            )

            val_shuffled_dataset = self.val_dataset.shuffle(
                seed=self.seed,
                buffer_size=self.buffer_size,
            )

            self.train_ds = train_shuffled_dataset.map(
                self.preprocess_batch, batch_size=self.batch_size, batched=True
            )

            self.val_ds = val_shuffled_dataset.map(
                self.preprocess_batch, batch_size=self.batch_size, batched=True
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )
