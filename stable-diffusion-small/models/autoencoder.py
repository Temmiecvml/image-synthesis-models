import lightning as L
import torch
import torch.nn.functional as F
from utils import instantiate_object, log_reconstruction


class VAutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        ckpt_dir,
        lr: float = 1e-3,
        beta_scale: float = 1,
    ):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.encoder = None
        self.decoder = None
        self.ckpt_dir = ckpt_dir

        self.lr = lr
        self.beta_scale = beta_scale

        self.save_hyperparameters()

    def configure_model(self):
        if self.encoder is not None or self.decoder is not None:
            return

        self.encoder = instantiate_object(self.encoder_config)
        self.decoder = instantiate_object(self.decoder_config)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mean, log_var

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar, self.beta_scale)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log("val_loss", loss)
        if batch_idx == 0:
            log_reconstruction(
                self.logger, x, recon_x, self.current_epoch, self.global_step
            )

    def test_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log("test_loss", loss)

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        MSE = F.mse_loss(recon_x, x, reduction="mean")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize KL divergence by batch size and apply beta scaling
        KLD = KLD / x.size(0)
        return MSE + beta * KLD

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of epochs for the first restart
            T_mult=2,  # Multiplier for the number of epochs between restarts
            eta_min=1e-6,  # Minimum learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
