import datetime
import importlib
import os
import random
from typing import Optional

import torch
import wandb
from PIL import Image
from torchvision import transforms


def log_reconstruction(
    logger,
    images,
    recon_x,
    epoch,
    step,
    num_samples=5,
) -> None:
    """Log Reconstruction"""
    random_indices = random.sample(range(images.size(0)), num_samples)
    images = [images[idx] for idx in random_indices]
    reconstructions = [recon_x[idx] for idx in random_indices]

    columns = ["original", "reconstruction"]
    data = [
        [wandb.Image(img), wandb.Image(recon)]
        for img, recon in zip(images, reconstructions)
    ]
    logger.log_table(
        key=f"reconstruction-{epoch:03d}/step-{step:06d}", columns=columns, data=data
    )


def instantiate_object(config, instantiate=True, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    module, cls = config["target"].rsplit(".", 1)

    module_imp = importlib.import_module(module)
    params = config.get("params", dict())
    params.update(kwargs)
    if instantiate:
        return getattr(module_imp, cls)(**params)

    return getattr(module_imp, cls)


def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_ckpt_dir(config, run_name: str):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    _ckpt_dir = config.train.checkpoint_dir.replace("{date}", now)
    _ckpt_dir = _ckpt_dir.replace("{run_name}", run_name)
    _ckpt_dir = _ckpt_dir.replace(
        "{data_name}", config.data.params.data_path.lower().replace("/", "_")
    )
    _ckpt_dir = _ckpt_dir.replace("{model_name}", config.train.model_name.lower())
    return _ckpt_dir


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def load_images_to_tensor(image_dir, image_width: int = 256, transformations=None):
    """
    Loads all images from the specified directory, preprocesses them, and
    converts them into a single PyTorch tensor.

    Parameters:
    - image_dir (str): Path to the directory containing images.
    - image_size (tuple): Desired size (width, height) for resizing images.

    Returns:
    - torch.Tensor: A 4D tensor of shape (N, C, H, W), where N is the number
                    of images, C is the number of channels, and H, W are height and width.
    """
    if not transformations:
        transform = transforms.Compose(
            [
                transforms.Resize(image_width),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    image_tensors = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            image_tensors.append(image_tensor)

    if len(image_tensors) > 0:
        return torch.stack(image_tensors)
    else:
        raise ValueError("No images found in the specified directory.")


def tensor_to_pil_images(tensor_batch):
    """
    Converts a batch of PyTorch tensors to a list of PIL images.

    Parameters:
    - tensor_batch (torch.Tensor): A 4D tensor of shape (N, C, H, W) where
                                   N is the batch size, C is 1 or 3 channels,
                                   and values are in range [0, 1] or [0, 255].

    Returns:
    - List[PIL.Image.Image]: A list of PIL images.
    """
    # Check if input is 4D tensor
    if tensor_batch.ndim != 4 or tensor_batch.shape[1] != 3:
        raise ValueError("Expected tensor of shape (N, C, H, W) with 1 or 3 channels.")

    pil_images = []
    for img_tensor in tensor_batch:
        if (
            img_tensor.dtype in [torch.float32, torch.float16]
            and img_tensor.max() <= 1.0
        ):
            img_tensor = img_tensor * 255

        img_tensor = img_tensor.to(torch.uint8).cpu()

        pil_image = transforms.ToPILImage()(img_tensor)
        pil_images.append(pil_image)

    return pil_images


def load_checkpoint(config, checkpoint_path: str):
    lightning_module = instantiate_object(config, instantiate=False)
    model = lightning_module.load_from_checkpoint(checkpoint_path=checkpoint_path)

    return model


def load_first_stage_component(ckpt_path: str, component: str = "encoder"):
    checkpoint = torch.load(ckpt_path)
    component_weights = {
        k.removeprefix(f"{component}."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(f"{component}.")
    }
    config = checkpoint["hyper_parameters"][f"{component}_config"]
    comp = instantiate_object(config)
    comp.load_state_dict(component_weights)

    return comp


def save_checkpoint(trainer, checkpoint_dir: str):
    trainer.save_checkpoint(checkpoint_dir)
