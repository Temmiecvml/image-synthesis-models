import argparse
import datetime
from functools import partial

import lightning as L
import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from modules.autoencoder.attention_block import VAttentionBlock
from modules.autoencoder.residual_block import VResidualBlock
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from utils import instantiate_object, load_checkpoint, logger

load_dotenv()  # set WANDB_API_KEY as env

torch.set_float32_matmul_precision("medium")


def get_fsdp_strategy(
    model_name,
    min_wrap_params,
):

    my_auto_wrap_policy = partial(
        size_based_auto_wrap_policy, min_num_params=min_wrap_params, recurse=True
    )

    if model_name == "autoencoder":
        activation_checkpointing_policy = {
            VResidualBlock,
            VAttentionBlock,
        }

    return FSDPStrategy(
        auto_wrap_policy=my_auto_wrap_policy,
        activation_checkpointing_policy=activation_checkpointing_policy,
        sharding_strategy="FULL_SHARD",
    )


def train_model(config, ckpt: str, seed: int, metric_logger):

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    ckpt_dir = config.train.checkpoint_dir.replace("{date}", now)

    config.data.seed = seed
    config.model.ckpt_dir = ckpt_dir

    if ckpt:
        logger.info(f"Loading Model at: {ckpt}")
        model = load_checkpoint(ckpt)

    else:
        logger.info(f"Instatialing Model with config at: {config}")
        model = instantiate_object(config.model, ckpt_dir=ckpt_dir)

    data_module = instantiate_object(config.data)
    metric_logger.watch(model)

    trainer = L.Trainer(
        strategy=get_fsdp_strategy(
            config.train.model_name, config.train.min_wrap_params
        ),
        devices=torch.cuda.device_count(),
        precision=config.train.precision,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        max_epochs=config.train.max_epochs,
        val_check_interval=config.train.val_check_interval,
        accelerator=config.train.accelerator,
        logger=metric_logger,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                save_top_k=2,
                monitor="val_loss",
                mode="min",
                dirpath=ckpt_dir,
                filename=f"{config.train.model_name}" + "-{epoch:02d}-{val_loss:.2f}",
            ),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                check_finite=True,
                min_delta=config.train.early_stopping_min_delta,
                patience=config.train.early_stopping_patience,
                verbose=True,
            ),
        ],
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt,
    )

    logger.info(
        f"Started training Model {config.model.target} with data: {config.data.target}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stable Diffusion Training Parameters")

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument("--cpkt", type=str, default="", help="checkpoint path")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    config = OmegaConf.load(args.config)
    ckpt = args.cpkt if args.cpkt else None

    wandb_logger = WandbLogger(
        project=f"stable_diffusion_{config.train.model_name}",
        prefix="poc",
        save_dir="logs",
    )

    seed_everything(args.seed)

    try:
        train_model(config, ckpt, args.seed, wandb_logger)

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
