import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import instantiate_object, log_reconstruction


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


class Posterior:
    def __init__(self, z, mu, log_var):
        self.z = z
        self.mu = mu
        self.log_var = log_var

    def kl(self):
        return -0.5 * torch.sum(
            1 + self.log_var - self.mu.pow(2) - self.log_var.exp(), dim=[1, 2, 3]
        )


class Generator(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = instantiate_object(encoder_config)
        self.decoder = instantiate_object(decoder_config)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z, mean, log_var


class VAutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        loss_config,
        ckpt_dir: str,
        lr: float,
        min_beta: float,
        max_beta: float,
        kl_anneal_epochs: int,
        accumulate_grad_batches: int,
    ):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.kl_anneal_epochs = kl_anneal_epochs
        self.accumulate_grad_batches = accumulate_grad_batches

        self.generator = Generator(encoder_config, decoder_config)
        self.discriminator = instantiate_object(loss_config)

        self.epoch = 1
        self.step = 1

        self.save_hyperparameters()

    def encode(self, x):
        z, mean, log_var = self.generator.encoder(x)
        return z, mean, log_var

    def decode(self, z):
        recon_x = self.generator.decoder(z)
        return recon_x

    def get_last_layer(self):
        return self.generator.decoder.conv_out.weight

    def train_model(
        self,
        fabric,
        train_dataloader,
        val_dataloader,
        max_epochs,
        val_check_interval,
        metric_to_monitor,
    ):
        steps_per_epoch = len(train_dataloader) // train_dataloader.batch_size

        for epoch in tqdm(range(self.epoch, max_epochs + 1)):
            print(f"Starting at Epoch {epoch} and Step {self.step}")
            self.epoch = epoch

            for b_idx, (x, _) in enumerate(train_dataloader):
                self.opt_ae.zero_grad()
                recon_x, z, mu, log_var = self.generator(x)
                posterior = Posterior(z, mu, log_var)
                aeloss, log_dict_ae = self.discriminator(
                    x,
                    recon_x,
                    posterior,
                    optimizer_idx=0,
                    global_step=self.step,
                    last_layer=self.get_last_layer(),
                    split="train",
                )
                fabric.backward(aeloss)
                self.opt_ae.step()
                log_dict_ae["ae_lr"] = self.opt_ae.param_groups[0]["lr"]
                fabric.log_dict(log_dict_ae, step=self.step)
                fabric.log("aeloss", aeloss, step=self.step)
                self.opt_disc.zero_grad()
                discloss, log_dict_disc = self.discriminator(
                    x,
                    recon_x,
                    posterior,
                    optimizer_idx=1,
                    global_step=self.step,
                    last_layer=None,
                    split="train",
                )

                fabric.backward(discloss)
                log_dict_disc["disc_lr"] = self.opt_disc.param_groups[0]["lr"]
                fabric.log_dict(log_dict_disc, step=self.step)
                fabric.log("discloss", discloss, step=self.step)

                if should_validate(
                    val_check_interval, steps_per_epoch, self.epoch, self.step
                ):
                    self.validate_model(fabric, val_dataloader, metric_to_monitor)

                self.step += 1

            if (self.epoch + 1) % 5 == 0:
                self.scheduler_ae.step()
                self.scheduler_disc.step()

    def validate_model(
        self,
        fabric,
        val_dataloader,
        metric_to_monitor,
    ):

        metric_values = [1e10, 1e10]
        for batch_idx, (x, _) in enumerate(val_dataloader):
            recon_x, z, mu, log_var = self.generator(x)
            posterior = Posterior(z, mu, log_var)
            aeloss, log_dict_ae = self.discriminator(
                x,
                recon_x,
                posterior,
                0,
                self.step,
                last_layer=self.get_last_layer(),
                split="val",
            )
            fabric.log_dict(log_dict_ae, step=self.step)
            fabric.log("aeloss", aeloss, step=self.step)
            discloss, log_dict_disc = self.discriminator(
                x,
                recon_x,
                posterior,
                1,
                self.step,
                last_layer=self.get_last_layer(),
                split="val",
            )
            fabric.log_dict(log_dict_disc, step=self.step)
            fabric.log("discloss", discloss, step=self.step)
            if fabric.is_global_zero and batch_idx == 0:
                log_reconstruction(
                    self.metric_logger, x, recon_x, self.epoch, self.step
                )
        metric = log_dict_ae.get(metric_to_monitor, "")
        if not metric:
            metric = log_dict_disc[metric_to_monitor]
        
        if fabric.is_global_zero:
            state = {
                "model": self,
                "opt_ae": self.opt_ae,
                "opt_disc": self.opt_disc,
                "scheduler_ae": self.scheduler_ae,
                "scheduler_disc": self.scheduler_disc,
                "epoch": self.epoch,
                "step": self.step,
            }

            # save if metric better than previous values
            second_best_prev = max(metric_values)
            id_second_best_prev = metric_values.index(second_best_prev)
            print(f"second_best_prev {second_best_prev}")
            print(f"id_second_best_prev {id_second_best_prev}")

            if metric < second_best_prev:
                fabric.save(
                    f"{self.ckpt_dir}/autoencoder-epoch={self.epoch:02d}-step={self.step:06d}-{metric_to_monitor}={metric:.2f}.ckpt",
                    state,
                )
                metric_values[id_second_best_prev] = metric
                print(f"Saved model with improved metric {metric}")
        
        fabric.barrier()
