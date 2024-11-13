import argparse
import datetime
from functools import partial

import lightning as L
import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from modules.autoencoder.attention_block import VAttentionBlock
from modules.autoencoder.residual_block import VResidualBlock
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from utils import instantiate_object, load_checkpoint, logger

load_dotenv()  # set WANDB_API_KEY as env

torch.set_float32_matmul_precision("medium")

MIN_FSDP_WRAP_PARAMS = 100_000


my_auto_wrap_policy = partial(
    size_based_auto_wrap_policy, min_num_params=MIN_FSDP_WRAP_PARAMS, recurse=True
)

activation_checkpointing_policy = {
    VResidualBlock,
    VAttentionBlock,
}


def train_model(config_path, ckpt: str, metric_logger):
    config = OmegaConf.load(config_path)
    config.data.seed = args.seed

    if ckpt:
        logger.info(f"Loading Model at: {ckpt}")
        model = load_checkpoint(ckpt)

    else:
        logger.info(f"Instatialing Model with config at: {config}")
        model = instantiate_object(config.model)

    data_module = instantiate_object(config.data)
    metric_logger.watch(model, log="all")

    trainer = L.Trainer(
        strategy=FSDPStrategy(
            auto_wrap_policy=my_auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy="FULL_SHARD",
        ),
        devices=torch.cuda.device_count(),
        precision="16-mixed",
        accumulate_grad_batches=2,
        max_epochs=10,
        accelerator="gpu",
        logger=metric_logger,
        callbacks=[RichProgressBar()],
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

    wandb_logger = WandbLogger(
        project="stable_diffusion_small", prefix="poc", save_dir="logs"
    )

    seed_everything(args.seed)

    try:
        if args.autoencoder_config:
            train_model(args.autoencoder_config, args.autoencoder_cpkt, wandb_logger)

        if args.ldm_config:
            train_model(args.ldm_config, args.ldm_cpkt, wandb_logger)

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
