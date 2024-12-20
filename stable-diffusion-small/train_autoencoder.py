import argparse
import os
import time
from datetime import datetime
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

from utils import get_ckpt_dir, instantiate_object, logger


class Posterior:
    def __init__(self, z, mu, log_var):
        self.z = z
        self.mu = mu
        self.log_var = log_var

    def kl(self):
        return -0.5 * torch.sum(
            1 + self.log_var - self.mu.pow(2) - self.log_var.exp(), dim=[1, 2, 3]
        )


def configure_optimizers(model):
    opt_ae = torch.optim.Adam(
        list(model.generator.parameters()),
        lr=model.lr,
        betas=(0.5, 0.9),
    )
    opt_disc = torch.optim.Adam(
        model.discriminator.discriminator.parameters(), lr=model.lr, betas=(0.5, 0.9)
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


def should_validate(val_check_interval, steps_per_epoch, epoch, step):
    if val_check_interval > 0 and val_check_interval < 1:
        val_check_interval = int(steps_per_epoch * val_check_interval)
        if step == val_check_interval:
            return True

    elif val_check_interval >= 1:
        if val_check_interval == epoch:
            return True
    else:
        raise ValueError(
            f"val_check_interval: {val_check_interval} should be greater than 0"
        )

    return False


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def setup_model(config):
    model = instantiate_object(config.model)
    return model


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    logger.info(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3  # Number of bytes in a gigabyte
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def train(
    model,
    rank,
    world_size,
    train_loader,
    opt_ae,
    opt_disc,
    epoch,
    metric_logger,
    sampler=None,
):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])

    if sampler:
        sampler.set_epoch(epoch)

    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )

    all_metrics = {}
    with autocast(dtype=torch.bfloat16):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(local_rank)
            opt_ae.zero_grad()
            recon_x, z, mu, log_var = model.generator(x)
            posterior = Posterior(z, mu, log_var)

            aeloss, log_dict_ae = model.discriminator(
                x,
                recon_x,
                posterior,
                optimizer_idx=0,
                global_step=model.step,
                last_layer=model.get_last_layer(),
                split="train",
            )

            aeloss.backward()
            opt_ae.step()

            opt_disc.zero_grad()
            discloss, log_dict_disc = model.discriminator(
                x,
                recon_x,
                posterior,
                optimizer_idx=1,
                global_step=model.step,
                last_layer=model.get_last_layer(),
                split="train",
            )

            discloss.backward()
            opt_disc.step()

            log_dict_ae["ae_loss"] = aeloss.item()
            log_dict_disc["disc_loss"] = discloss.item()

            if batch_idx == 0:
                all_metrics = {
                    **log_dict_ae,
                    **log_dict_disc,
                    "ae_loss": aeloss.item(),
                    "disc_loss": discloss.item(),
                }
            else:
                for key in log_dict_ae.keys():
                    all_metrics[key] += log_dict_ae[key]

                for key in log_dict_disc.keys():
                    all_metrics[key] += log_dict_disc[key]

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(all_metrics, op=dist.ReduceOp.SUM)

    if rank == 0:
        for key in all_metrics.keys():
            all_metrics[key] = all_metrics[key] / len(train_loader) / world_size
        all_metrics["ae_lr"] = opt_ae.param_groups[0]["lr"]
        all_metrics["disc_lr"] = opt_disc.param_groups[0]["lr"]

        inner_pbar.close()
        logger.info(f"--> epoch {epoch} completed...entering save and stats zone")
        for key in all_metrics.keys():
            logger.info(f"{key}: {all_metrics[key]}")
            metric_logger.log(key, all_metrics[key])
            logger.info(f"completed save and stats")

    return all_metrics


def validation(model, rank, world_size, val_loader, epoch, metric_logger):
    model.eval()
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(local_rank)
            recon_x, z, mu, log_var = model.generator(x)
            posterior = Posterior(z, mu, log_var)

            aeloss, log_dict_ae = model.discriminator(
                x,
                recon_x,
                posterior,
                optimizer_idx=0,
                global_step=model.step,
                last_layer=model.get_last_layer(),
                split="train",
            )

            discloss, log_dict_disc = model.discriminator(
                x,
                recon_x,
                posterior,
                optimizer_idx=1,
                global_step=model.step,
                last_layer=model.get_last_layer(),
                split="train",
            )

            log_dict_ae["ae_loss"] = aeloss.item()
            log_dict_disc["disc_loss"] = discloss.item()

            if batch_idx == 0:
                all_metrics = {
                    **log_dict_ae,
                    **log_dict_disc,
                    "ae_loss": aeloss.item(),
                    "disc_loss": discloss.item(),
                }
            else:
                for key in log_dict_ae.keys():
                    all_metrics[key] += log_dict_ae[key]

                for key in log_dict_disc.keys():
                    all_metrics[key] += log_dict_disc[key]

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(all_metrics, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        for key in all_metrics.keys():
            all_metrics[key] = all_metrics[key] / len(val_loader) / world_size

        inner_pbar.close()
        logger.info(f"--> epoch {epoch} completed...entering save and stats zone")
        for key in all_metrics.keys():
            logger.info(f"{key}: {all_metrics[key]}")
            metric_logger.log(key, all_metrics[key])
            logger.info(f"completed save and stats")

    return all_metrics


def fsdp_main(args, metric_logger):

    config = OmegaConf.load(args.config)

    model = setup_model(config)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        run_name = (
            metric_logger.name
            if isinstance(metric_logger.name, str)
            else "default"
        )

        ckpt_dir = get_ckpt_dir(config, run_name)
    else:
        run_name = "default"
        ckpt_dir = "default"

    data = instantiate_object(config.data)
    data.prepare_data()
    data.setup("fit")

    if rank == 0:
        logger.info("Size of train dataset: ", len(data.train_ds))
        logger.info("Size of Validation dataset: ", len(data.val_ds))

    train_sampler = DistributedSampler(
        data.train_ds, rank=rank, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(data.val_ds, rank=rank, num_replicas=world_size)

    setup(rank, world_size)

    train_kwargs = {"sampler": train_sampler}
    test_kwargs = {"sampler": val_sampler}

    data.train_loader = data.train_dataloader(**train_kwargs)
    data.val_loader = data.val_dataloader(**test_kwargs)

    my_auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=config.train.min_wrap_params,
        recurse=True,
    )
    sharding_strategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(local_rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    if (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and dist.is_nccl_available()
    ):
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None  # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True, 
    )

    if rank == 0:
        metric_logger.watch(model)

    model.opt_ae, model.opt_disc, model.scheduler_ae, model.scheduler_disc = configure_optimizers(model)


    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        mem_alloc_tracker = []
        mem_reserved_tracker = []
        best_val_loss = float("inf")
        curr_val_loss = float("inf")
        early_stopping_counter = 0

    for epoch in range(1, config.train.max_epochs + 1):
        t0 = time.time()
        all_train_metrics = train(
            model,
            rank,
            world_size,
            data.train_loader,
            model.opt_ae,
            model.opt_disc,
            epoch,
            metric_logger,
            sampler=train_sampler,
        )
        if should_validate(
            config.train.val_check_interval, len(data.train_loader), epoch, model.step
        ):
            all_val_metrics = validation(
                model, rank, world_size, data.val_loader, epoch, metric_logger
            )
            curr_val_loss = all_val_metrics["ae_loss"]

        model.scheduler_ae.step()
        model.scheduler_disc.step()

        dist.barrier()

        if rank == 0:
            logger.info(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            mem_alloc_tracker.append(
                format_metrics_to_gb(torch.cuda.memory_allocated())
            )
            mem_reserved_tracker.append(
                format_metrics_to_gb(torch.cuda.memory_reserved())
            )

            if curr_val_loss < best_val_loss:
                delta_loss = best_val_loss - curr_val_loss
                logger.info(f"--> New Val Loss Record: {curr_val_loss}")
                if delta_loss > config.train.early_stopping_min_delta:
                    early_stopping_counter = 0
                    best_val_loss = curr_val_loss
                    save_policy = FullStateDictConfig(
                        offload_to_cpu=True, rank0_only=True
                    )
                    with FSDP.state_dict_type(
                        model, StateDictType.FULL_STATE_DICT, save_policy
                    ):
                        cpu_state = model.state_dict()
                        logger.info(f"--> saving model ...")
                        currEpoch = (
                            "-"
                            + str(epoch)
                            + "-"
                            + str(round(curr_val_loss.item(), 4))
                            + ".pt"
                        )
                        logger.info(f"--> attempting to save model prefix {currEpoch}")
                        save_name = "autoencoder" + "-" + time_of_run + "-" + currEpoch
                        logger.info(f"--> saving as model name {save_name}")

                        torch.save(cpu_state, os.path.join(ckpt_dir, save_name))

                else:
                    early_stopping_counter += 1
                    logger.info(f"--> Early Stopping Counter: {early_stopping_counter}")
                    if early_stopping_counter >= config.train.early_stopping_patience:
                        logger.info(f"--> Early Stopping at Epoch {epoch}")
                        break
        model.epoch += 1
    init_end_event.record()

    cleanup()


if __name__ == "__main__":
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

    args = parser.parse_args()

    if int(os.environ["RANK"]) == 0:
        metric_logger = wandb.init(
            project="stable_diffusion_autoencoder",
            config=args,
        )
    else:
        metric_logger = None

    torch.manual_seed(args.seed)

    fsdp_main(args, metric_logger)
