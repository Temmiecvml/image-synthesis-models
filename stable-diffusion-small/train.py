import argparse
from functools import partial

import lightning as L
import torch
import re
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.fabric.strategies import FSDPStrategy as FabricFSDPStrategy
from modules.autoencoder.attention_block import VAttentionBlock
from modules.autoencoder.residual_block import VResidualBlock
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from utils import instantiate_object, logger, get_available_device, get_ckpt_dir

load_dotenv()  # set WANDB_API_KEY as env

torch.set_float32_matmul_precision("medium")


def get_epoch_and_global_step(ckpt_path):
    """
    Extracts the epoch and global step from a checkpoint file name.
    """
    pattern = r"autoencoder-epoch=(\d+)-step=(\d+)"
    match = re.search(pattern, ckpt_path)

    if not match:
        raise ValueError(f"Checkpoint path does not match expected format: {ckpt_path}")

    epoch = int(match.group(1))
    global_step = int(match.group(2))

    return epoch, global_step


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

        return FabricFSDPStrategy(
            auto_wrap_policy=my_auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy="FULL_SHARD",
        )

    elif model_name == "ldm":
        return FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy)


def train_autoencoder(config, ckpt: str, seed: int, metric_logger):

    metric_to_monitor = config.train.metric_to_monitor
    run_name = (
        metric_logger.experiment.name
        if isinstance(metric_logger.experiment.name, str)
        else "default"
    )

    ckpt_dir = get_ckpt_dir(config, run_name)

    with fabric.init_module(empty_init=True):
        model = instantiate_object(
            config.model,
            ckpt_dir=ckpt_dir,
            accumulate_grad_batches=config.train.accumulate_grad_batches,
        )

    optimizer_ae, optimizer_disc, scheduler_ae, scheduler_disc = (
        model.configure_optimizers()
    )

    data_module = instantiate_object(config.data)

    fabric = L.Fabric(
        accelerator=get_available_device(),
        devices=(
            "auto" if config.train.accelerator == "mps" else torch.cuda.device_count()
        ),
        precision=(
            "32-true" if config.train.accelerator == "mps" else config.train.precision
        ),
    )

    state = {
        "model": model,
        "optimizer_ae": optimizer_ae,
        "optimizer_disc": optimizer_disc,
        "scheduler_ae": scheduler_ae,
        "scheduler_disc": scheduler_disc,
        "epoch": epoch,
        "global_step": global_step,
    }

    if ckpt:
        epoch, global_step = get_epoch_and_global_step(ckpt)
        state["epoch"] = epoch
        state["global_step"] = global_step
        fabric.load(model, state)
        model.epoch = epoch
        model.global_step = global_step

    fabric.launch()
    fabric.seed_everything(seed)

    with fabric.rank_zero_first():
        data_module.prepare_data()

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model.generator, optimizer_ae = fabric.setup(model.generator, optimizer_ae)
    model.discriminator, optimizer_disc = fabric.setup(
        model.discriminator, optimizer_disc
    )
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)

    model.train_model(
        fabric,
        train_loader,
        val_loader,
        config.train.max_epochs,
        config.train.val_check_interval,
        metric_to_monitor,
    )


def train_ldm(config, ckpt: str, seed: int, metric_logger):

    config.data.seed = seed
    config.train.accelerator = get_available_device()
    run_name = (
        metric_logger.experiment.name
        if isinstance(metric_logger.experiment.name, str)
        else "default"
    )

    ckpt_dir = get_ckpt_dir(config, run_name)
    seed_everything(seed)
    model = instantiate_object(
        config.model,
        ckpt_dir=ckpt_dir,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
    )
    data_module = instantiate_object(config.data)
    metric_logger.watch(model)

    metric_to_monitor = config.train.metric_to_monitor

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
                monitor=metric_to_monitor,
                mode="min",
                dirpath=ckpt_dir,
                filename=f"{config.train.model_name}"
                + "-{epoch:02d}-{metric_to_monitor:.2f}".replace(
                    "metric_to_monitor", metric_to_monitor
                ),
            ),
            EarlyStopping(
                monitor=metric_to_monitor,
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

    try:
        train_model(config, ckpt, args.seed, wandb_logger)

    except Exception as e:
        logger.error(e)
        raise e
