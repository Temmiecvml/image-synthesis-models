import argparse
import re
from functools import partial

import lightning as L
import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.fabric.strategies import FSDPStrategy as FabricFSDPStrategy
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)
from wandb.integration.lightning.fabric import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from modules.autoencoder.attention_block import VAttentionBlock
from modules.autoencoder.residual_block import VResidualBlock
from utils import (get_available_device, get_ckpt_dir, instantiate_object,
                   logger)

load_dotenv()  # set WANDB_API_KEY as env

torch.set_float32_matmul_precision("medium")


def configure_optimizers(
    generator,
    discriminator,
    lr,
):
    opt_ae = torch.optim.Adam(
        list(generator.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )
    opt_disc = torch.optim.Adam(
        discriminator.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    scheduler_ae = torch.optim.lr_scheduler.StepLR(
        opt_ae,
        step_size=5,  # Number of epochs after which LR is reduced
        gamma=0.5,  # Multiplicative factor for LR reduction
    )
    scheduler_disc = torch.optim.lr_scheduler.StepLR(
        opt_disc,
        step_size=5,
        gamma=0.5,
    )

    return opt_ae, opt_disc, scheduler_ae, scheduler_disc


def get_epoch_and_step(ckpt_path):
    """
    Extracts the epoch and global step from a checkpoint file name.
    """
    pattern = r"autoencoder-epoch=(\d+)-step=(\d+)"
    match = re.search(pattern, ckpt_path)

    if not match:
        raise ValueError(f"Checkpoint path does not match expected format: {ckpt_path}")

    epoch = int(match.group(1))
    step = int(match.group(2))

    return epoch, step


def get_fsdp_strategy(
    model_name,
    min_wrap_params,
):
    device = get_available_device()
    if device in ["mps", "cpu"]:
        return "auto"

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
    config.data.seed = seed
    config.train.accelerator = get_available_device()
    metric_to_monitor = config.train.metric_to_monitor
    run_name = (
        metric_logger.experiment.name
        if isinstance(metric_logger.experiment.name, str)
        else "default"
    )

    ckpt_dir = get_ckpt_dir(config, run_name)

    fabric = L.Fabric(
        accelerator=config.train.accelerator,
        devices="auto",
        precision=(
            "32-true"
            if config.train.accelerator == "mps" or torch.cuda.device_count() == 1
            else config.train.precision
        ),
        strategy=get_fsdp_strategy("autoencoder", config.train.min_wrap_params),
    )

    fabric.launch()

    with fabric.init_module(empty_init=True):
        model = instantiate_object(
            config.model,
            ckpt_dir=ckpt_dir,
            accumulate_grad_batches=config.train.accumulate_grad_batches,
        )

    model.metric_logger = metric_logger

    data_module = instantiate_object(config.data)

    # state = {
    #     "model": model,
    #     "opt_ae": model.opt_ae,
    #     "opt_disc": model.opt_disc,
    #     "scheduler_ae": model.scheduler_ae,
    #     "scheduler_disc": model.scheduler_disc,
    #     "epoch": model.epoch,
    #     "step": model.step,
    # }
    state = {
        "model": model,
        "epoch": model.epoch,
        "step": model.step,
    }

    if ckpt:
        epoch, step = get_epoch_and_step(ckpt)
        state["epoch"] = epoch
        state["step"] = step
        fabric.load(model, state)
        model.epoch = epoch
        model.step = step

    fabric.seed_everything(seed)

    with fabric.rank_zero_first():
        data_module.prepare_data()

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model.generator = fabric.setup_module(model.generator)
    model.discriminator = fabric.setup_module(model.discriminator)

    opt_ae, opt_disc, model.scheduler_ae, model.scheduler_disc = configure_optimizers(
        model.generator, model.discriminator, model.lr
    )
    model.opt_ae = fabric.setup_optimizers(opt_ae)
    model.opt_disc = fabric.setup_optimizers(opt_disc)
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
        project=f"bugfix_autoencoder_{config.train.model_name}",
        prefix="poc",
        save_dir="logs",
        **kwargs,
    )

    try:
        if "autoencoder" in args.config:
            train_autoencoder(config, ckpt, args.seed, wandb_logger)
        else:
            train_ldm(config, ckpt, args.seed, wandb_logger)

    except Exception as e:
        logger.error(e)
        raise e
