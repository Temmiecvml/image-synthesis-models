import argparse
import datetime

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils import instantiate_object, logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stable Diffusion Training Parameters")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="hyperparameters/autoencoder/config.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument("--cpkt", type=str, required=True, help="checkpoint path")

    parser.add_argument(
        "--resume", action="store_true", help="Resume from a previous run if specified."
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
        config = OmegaConf.load(args.config)
        # config.data.seed = args.seed
        # logger.info(f"Loading config at: {args.config}")
        # model = instantiate_object(config.model)
        # data_module = instantiate_object(config.data)
        # logger.info(
        #     f"Started training Model {config.model.target} at: {config.data.target}"
        # )

        # trainer = pl.Trainer(
        #     max_epochs=10,
        #     accelerator="mps",
        # )

        # trainer.fit(model, datamodule=data_module)
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
