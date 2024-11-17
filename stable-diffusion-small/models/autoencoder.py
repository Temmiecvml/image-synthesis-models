import lightning as L
import torch
import torch.nn.functional as F
from utils import instantiate_object, log_reconstruction


class VAutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        ckpt_dir: str,
        lr: float,
        min_beta: float,
        max_beta: float,
        kl_anneal_epochs: int,
    ):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.encoder = None
        self.decoder = None
        self.ckpt_dir = ckpt_dir

        self.lr = lr
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.kl_anneal_epochs = kl_anneal_epochs

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

    def encode(self, x):
        z, mean, log_var = self.encoder(x)
        return z, mean, log_var

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        mse, kld, beta = self.loss_function(recon_x, x, mu, logvar)
        loss = (mse + beta * kld).mean()

        self.log("mse", mse, batch_size=x.size[0])
        self.log("kld", kld, batch_size=x.size[0])
        self.log("beta", beta, batch_size=x.size[0])
        self.log("train_loss", loss, batch_size=x.size[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        mse, kld, beta = self.loss_function(recon_x, x, mu, logvar)
        loss = (mse + beta * kld).mean()
        self.log("mse", mse, batch_size=x.size[0])
        self.log("kld", kld, batch_size=x.size[0])
        self.log("beta", beta, batch_size=x.size[0])
        self.log("val_loss", loss, batch_size=x.size[0])

        if batch_idx == 0:
            log_reconstruction(
                self.logger, x, recon_x, self.current_epoch, self.global_step
            )

        return loss

    def get_kl_weight(self):
        return min(
            self.max_beta,
            self.min_beta
            + (self.current_epoch / self.kl_anneal_epochs) * self.max_beta,
        )

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        MSE = MSE / recon_x.size(0)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / recon_x.size(0)

        beta = self.get_kl_weight()

        return MSE, KLD, beta

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
