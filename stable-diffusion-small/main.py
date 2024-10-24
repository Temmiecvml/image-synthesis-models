import datetime
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from utils import instantiate_object


class Args:
    seed = 23
    base = ["hyperparameters/autoencoder/config.yaml"]
    resume = ""


if __name__ == "__main__":
    opt = Args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    seed_everything(opt.seed)

    try:
        config = OmegaConf.merge(*[OmegaConf.load(cfg) for cfg in opt.base])
        sample_input = torch.ones(2, 3, 256, 256)
        c = ["I am a demo", "I am a cat"]
        model = instantiate_object(config.autoencoder)
        recon_x, mean, log_var = model(sample_input)

        print("Model output shape: ", recon_x.shape)

    except Exception as e:
        print(e)
        raise e
