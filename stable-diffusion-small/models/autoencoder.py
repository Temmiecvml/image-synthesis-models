import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from utils import instantiate_object


class VAutoEncoder(pl.LightningModule):
    def __init__(
        self, encoder_config, decoder_config, lr: float = 1e-3, beta_scale: float = 1
    ):
        super().__init__()
        self.save_hyperparameters()
        rank = self.global_rank

        if rank % 2 == 0:
            self.encoder = instantiate_object(encoder_config).to(rank)
            self.decoder = instantiate_object(decoder_config).to(rank + 1)

        self.lr = lr
        self.beta_scale = beta_scale

    def forward(self, x):
        rank = self.global_rank
        if rank % 2 == 0:
            x = x.to(rank)
            z, mean, log_var = self.encoder(x)
            z = z.to(rank + 1)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
