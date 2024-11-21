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
from utils import instantiate_object, logger

load_dotenv()  # set WANDB_API_KEY as env

torch.set_float32_matmul_precision("medium")


def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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


def get_ckpt_dir(config, run_name: str):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    _ckpt_dir = config.train.checkpoint_dir.replace("{date}", now)
    _ckpt_dir = _ckpt_dir.replace("{run_name}", run_name)
    _ckpt_dir = _ckpt_dir.replace(
        "{data_name}", config.data.params.data_path.lower().replace("/", "_")
    )
    _ckpt_dir = _ckpt_dir.replace("{model_name}", config.train.model_name.lower())
    return _ckpt_dir


def train_model(config, ckpt: str, seed: int, metric_logger):

    config.data.seed = seed
    config.train.accelerator = get_available_device()
    ckpt_dir = get_ckpt_dir(config, metric_logger.experiment.name)

    model = instantiate_object(config.model, ckpt_dir=ckpt_dir)
    data_module = instantiate_object(config.data)
    metric_logger.watch(model)

    trainer = L.Trainer(
        strategy=(
            "auto"
            if config.train.accelerator == "mps"
            else get_fsdp_strategy(
                config.train.model_name, config.train.min_wrap_params
            )
        ),
        devices=(
            "auto" if config.train.accelerator == "mps" else torch.cuda.device_count()
        ),
        precision=(
            "32-true" if config.train.accelerator == "mps" else config.train.precision
        ),
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        max_epochs=config.train.max_epochs,
        val_check_interval=config.train.val_check_interval,
        accelerator=config.train.accelerator,
        logger=metric_logger,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                save_top_k=2,
                monitor="ae_val/rec_loss",
                mode="min",
                dirpath=ckpt_dir,
                filename=f"{config.train.model_name}" + "-{epoch:02d}-{val_loss:.2f}",
            ),
            EarlyStopping(
                monitor="ae_val/rec_loss",
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

    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")

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

    if args.ckpt:
        ckpt = args.ckpt
        kwargs = {
            "id": next(
                run_part.split("=")[1]
                for run_part in ckpt.split("/")
                if run_part.startswith("run=")
            ),
            "resume": "must",
        }
    else:
        ckpt = None
        kwargs = {}

    wandb_logger = WandbLogger(
        project=f"poc_local_stable_diffusion_{config.train.model_name}",
        prefix="poc",
        save_dir="logs",
        **kwargs,
    )

    seed_everything(args.seed)

    try:
        train_model(config, ckpt, args.seed, wandb_logger)

    except Exception as e:
        logger.error(e)
        raise e
