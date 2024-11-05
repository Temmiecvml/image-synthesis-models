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
    process_text = lambda x: x.lower().removeprefix(prefix).strip()
    process_image = lambda x: transform(x)
    samples["text"] = (
        [process_text(text) for text in samples["text"]]
        if type(samples["text"]) is list
        else process_text(samples["text"])
    )
    samples["image"] = (
        [process_image(image) for image in samples["image"]]
        if type(samples["image"]) is list
        else process_image(samples["image"])
    )
    return samples


def collate_celebahq_caption(samples):

    images = np.array([sample["image"] for sample in samples])
    # no string representation in torch
    texts = np.array([sample["text"] for sample in samples])
    images = torch.from_numpy(images / 255).to(torch.float32)
    # normalize
    images = (images - 0.5) / 0.5
    # put batch dimension to position 1
    images = images.permute(0, 3, 1, 2)

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
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        cache_dir: str = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.cache_dir = cache_dir
        # https://pytorch.org/docs/stable/data.html
        # https://github.com/pytorch/pytorch/issues/13246

        self.transform = transforms.Compose(
            [
                CustomResizeAndCrop(target_size=image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ToTensor(), # we convert to tensor in collation function
                # transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        preprocess_batch = getattr(sys.modules[__name__], preprocess_batch_fn, None)

        self.preprocess_batch = partial(preprocess_batch, transform=self.transform)

        self.collate_fn = getattr(sys.modules[__name__], collate_fn, None)

        if not callable(self.preprocess_batch):
            raise ValueError(
                f"Provided preprocess_batch_fn '{preprocess_batch_fn}' is not a callable function"
            )

    def prepare_data(self):
        self.dataset = load_dataset(
            self.data_path,
            split="train",
            streaming=True,
            cache_dir=self.cache_dir,
        )

    def setup(self, stage: str):
        shuffled_dataset = self.dataset.shuffle(
            seed=self.seed, buffer_size=self.buffer_size
        )

        total_samples = 1 / (1 - self.val_split_ratio - self.test_split_ratio)

        # Create iterators for train, val, and test splits
        self.train_ds = shuffled_dataset.take(
            int((1 - self.val_split_ratio - self.test_split_ratio) * total_samples)
        )
        self.val_ds = shuffled_dataset.skip(
            int((1 - self.val_split_ratio - self.test_split_ratio) * total_samples)
        ).take(int(self.val_split_ratio * total_samples))
        self.test_ds = shuffled_dataset.skip(
            int((1 - self.test_split_ratio) * total_samples)
        )

        # Apply batch preprocessing
        if self.preprocess_batch:
            self.train_ds = self.train_ds.map(
                self.preprocess_batch, batch_size=self.batch_size, batched=True
            )
            self.val_ds = self.val_ds.map(
                self.preprocess_batch, batch_size=self.batch_size, batched=True
            )
            self.test_ds = self.test_ds.map(
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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
