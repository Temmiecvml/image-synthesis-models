import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils import instantiate_object


class VAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = instantiate_object(encoder_config)
        self.decoder = instantiate_object(decoder_config)
        self.learning_rate = learning_rate

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
        loss = self.loss_function(recon_x, x, mu, logvar)
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

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

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
