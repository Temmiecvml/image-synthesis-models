import argparse
import datetime

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils import instantiate_object, load_checkpoint, logger
from pytorch_lightning.strategies import FSDPStrategy


def train_model(config_path, ckpt: str):
    config = OmegaConf.load(config_path)
    config.data.seed = args.seed

    if ckpt:
        logger.info(f"Loading Model at: {ckpt}")
        model = load_checkpoint(ckpt)

    else:
        logger.info(f"Instatialing Model with config at: {config}")
        model = instantiate_object(config.model)

    data_module = instantiate_object(config.data)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        strategy=FSDPStrategy()
    )

    trainer.fit(model, datamodule=data_module)

    logger.info(
        f"Started training Model {config.model.target} with data: {config.data.target}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stable Diffusion Training Parameters")

    parser.add_argument(
        "--autoencoder_config",
        type=str,
        default="",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--ldm_config",
        type=str,
        default="",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--autoencoder_cpkt", type=str, default="", help="autoencoder checkpoint path"
    )

    parser.add_argument(
        "--ldm_cpkt",
        type=str,
        default="",
        help="latent diffusion model checkpoint path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    seed_everything(args.seed)

    try:
        if args.autoencoder_config:
            train_model(args.autoencoder_config, args.autoencoder_cpkt)

        if args.ldm_config:
            train_model(args.ldm_config, args.ldm_cpkt)

        # import torch

        # sample_input = torch.randn(2, 3, 512, 512)
        # c = ["I am a demo", "I am a cat", ""]
        # model = instantiate_object(config.model)
        # recon_x, mean, log_var = model(sample_input)

        # print("Model output shape: ", recon_x.shape)

        # diffusion_model = instantiate_object(config.ddpm)
        # o = diffusion_model(sample_input, c)

    except Exception as e:
        logger.error(e)
        raise e
