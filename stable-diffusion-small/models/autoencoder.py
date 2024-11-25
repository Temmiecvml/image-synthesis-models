import lightning as L
import torch
import torch.nn.functional as F
from utils import instantiate_object, log_reconstruction


class Posterior:
    def __init__(self, z, mu, log_var):
        self.z = z
        self.mu = mu
        self.log_var = log_var

    def kl(self):
        return -0.5 * torch.sum(
            1 + self.log_var - self.mu.pow(2) - self.log_var.exp(), dim=[1, 2, 3]
        )


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
    ):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.loss = instantiate_object(loss_config)
        self.encoder = None
        self.decoder = None
        self.ckpt_dir = ckpt_dir

        self.lr = lr
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.kl_anneal_epochs = kl_anneal_epochs
        self.automatic_optimization = False

        self.save_hyperparameters()

    def configure_model(self):
        if self.encoder is not None or self.decoder is not None:
            return

        self.encoder = instantiate_object(self.encoder_config)
        self.decoder = instantiate_object(self.decoder_config)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z, mean, log_var

    def encode(self, x):
        z, mean, log_var = self.encoder(x)
        return z, mean, log_var

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()

        x, _ = batch
        recon_x, z, mu, log_var = self(x)

        posterior = Posterior(z, mu, log_var)

        aeloss, log_dict_ae = self.loss(
            x,
            recon_x,
            posterior,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        self.log("ae_loss", aeloss, prog_bar=True, logger=True)

        for key, value in log_dict_ae.items():
            self.log(f"ae_{key}", value, logger=True)

        lr_ae = opt_ae.param_groups[0]["lr"]
        self.log("lr_ae", lr_ae, prog_bar=True, logger=True)

        self.manual_backward(aeloss)
        opt_ae.step()
        opt_ae.zero_grad()

        discloss, log_dict_disc = self.loss(
            x,
            recon_x,
            posterior,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        self.log("disc_loss", discloss, prog_bar=True, logger=True)
        for key, value in log_dict_disc.items():
            self.log(f"disc_{key}", value, logger=True)

        lr_disc = opt_disc.param_groups[0]["lr"]
        self.log("lr_disc", lr_disc, prog_bar=True, logger=True)

        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, z, mu, log_var = self(x)

        posterior = Posterior(z, mu, log_var)

        aeloss, log_dict_ae = self.loss(
            x,
            recon_x,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        for key, value in log_dict_ae.items():
            self.log(f"ae_{key}", value, logger=True)

        discloss, log_dict_disc = self.loss(
            x,
            recon_x,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        for key, value in log_dict_disc.items():
            self.log(f"disc_{key}", value, logger=True)

        if batch_idx == 0:
            log_reconstruction(
                self.logger, x, recon_x, self.current_epoch, self.global_step
            )

    def get_kl_weight(self):
        return min(
            self.max_beta,
            self.min_beta
            + (self.current_epoch / self.kl_anneal_epochs) * self.max_beta,
        )

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="mean")

        KLD = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)),
            dim=0,
        )

        beta = self.get_kl_weight()

        return MSE, KLD, beta

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9)
        )

        scheduler_ae = torch.optim.lr_scheduler.StepLR(
            opt_ae,
            step_size=10,  # Number of epochs after which LR is reduced
            gamma=0.5,  # Multiplicative factor for LR reduction
        )
        scheduler_disc = torch.optim.lr_scheduler.StepLR(
            opt_disc,
            step_size=10,
            gamma=0.5,
        )

        return (
            [opt_ae, opt_disc],
            [
                {"scheduler": scheduler_ae, "interval": "epoch", "frequency": 1},
                {"scheduler": scheduler_disc, "interval": "epoch", "frequency": 1},
            ],
        )
