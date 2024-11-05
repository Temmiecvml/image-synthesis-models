import sys

import pytorch_lightning as pl
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def preprocess_celebahq_caption(samples):
    prefix = "a photography of"
    samples["text"] = [i.strip(prefix).strip("").lower() for i in samples["text"]]

    return samples


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

        self.transform = transforms.Compose(
            [
                CustomResizeAndCrop(target_size=image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.preprocess_batch = getattr(
            sys.modules[__name__], preprocess_batch_fn, None
        )
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
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
