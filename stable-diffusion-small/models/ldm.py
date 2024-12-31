from functools import partial
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from utils import (
    extract_into_tensor,
    instantiate_object,
    load_first_stage_component,
    logger,
    make_beta_schedule,
    timestep_embedding,
)


class LDM(L.LightningModule):
    def __init__(
        self,
        unet_config,
        timestep_config,
        text_conditioner_config,
        first_stage_encoder_ckpt,
        lr,
        beta_schedule,
        num_timesteps,
        loss_type,
        v_posterior,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight,
        elbo_weight,
        ckpt_dir,
    ):
        super().__init__()

        self.unet_config = unet_config
        self.text_conditioner_config = text_conditioner_config
        self.first_stage_encoder_ckpt = first_stage_encoder_ckpt
        self.lr = lr

        self.unet = None
        self.text_conditioner = None
        self.first_stage_encoder = None

        self.time_embedder = instantiate_object(timestep_config)

        self.num_timesteps = int(num_timesteps)
        self.loss_type = loss_type
        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight
        self.elbo_weight = elbo_weight

        self.ckpt_dir = ckpt_dir

        self.logvar = torch.nn.Parameter(
            torch.full((num_timesteps,), 0.0), requires_grad=True
        )
        self.create_schedule(beta_schedule, num_timesteps)

    def configure_model(self):
        if (
            self.unet is not None
            or self.text_conditioner is not None
            or self.first_stage_encoder is not None
        ):
            return

        self.unet = instantiate_object(self.unet_config)

        self.text_conditioner = (
            instantiate_object(
                self.text_conditioner_config,
                device=str(self.device),
            )
            .requires_grad_(False)
            .eval()
        )
        self.first_stage_encoder = (
            load_first_stage_component(self.first_stage_encoder_ckpt)
            .requires_grad_(False)
            .eval()
        )

    def idle_component(self, component: str, idle: bool = True):
        def is_on_cpu(comp):
            for param in comp.parameters():
                return param.device.type == "cpu"

        if component not in ["text_conditioner", "first_stage_encoder"]:
            raise ValueError(f"Invalid component {component}")

        comp = getattr(self, component)

        if idle and is_on_cpu(comp):
            logger.info(f"{component} is already idle.")
            setattr(self, f"{component}_idle", True)
            return

        if idle and not is_on_cpu(comp):
            logger.info(f"Idling {component}")
            setattr(self, f"{component}_idle", True)
            comp.to("cpu")
            return

        if not idle and is_on_cpu(comp):
            logger.info(f"Activating {component}")
            setattr(self, f"{component}_idle", False)
            comp.to(self.device)
            return

        if not idle and not is_on_cpu(comp):
            logger.info(f"{component} is already active.")
            setattr(self, f"{component}_idle", False)
            return

    def create_schedule(self, beta_schedule, num_timesteps):
        betas = make_beta_schedule(
            beta_schedule,
            num_timesteps,
            linear_start=1e-4,
            linear_end=2e-2,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

        lvlb_weights = self.betas**2 / (
            2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
        )
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)

    def get_context_embedding(self, context: List[str]):
        """
        Get the conditioning text embedding
        Conditioner is
            - CLIP text model: we take the entire sequence embeddings
              including paddings not the embedding of the most probable token
              as is done in the original paper
            - Multihead attention: we use the entire sequence
            - Linear projection: we project the entire sequence

            for a max sequence length of 77, and output dims of 640
            we get an output shape of (batch_size, 77, 640)å
        """

        with torch.no_grad():
            context = self.text_conditioner(context)

        return context

    def get_timestep_embedding(self, t):
        """Get the timestep embedding"""
        dims = self.time_embedder.dims
        t = timestep_embedding(t, dims, max_period=10000)
        t_embed = self.time_embedder(t)

        return t_embed

    def sample_noisy_image(self, x_start, t, noise):
        """Diffusion forward pass"""
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cum_prod_t = extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cum_prod_t = extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        # simmilar to the reparameterization trick: mean + std * noise
        noisy_image = (
            sqrt_alphas_cum_prod_t * x_start + sqrt_one_minus_alphas_cum_prod_t * noise
        )

        return noisy_image

    def predict_noise(self, x_noisy: torch.Tensor, context: np.array, t: torch.Tensor):
        """Predict the noise in a noisy latent
            conditioned on a text

        Args:
            x_noisy (torch.Tensor): _description_
            context (np.array): _description_
            t (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """

        if isinstance(context, np.ndarray):
            context = list(context)

        context = self.get_context_embedding(context).to(x_noisy.dtype)
        t_embeddings = self.get_timestep_embedding(t)
        noise = self.unet(x_noisy, context, t_embeddings)

        return noise

    def forward(
        self,
        x: torch.Tensor,
        context: List[str],
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        x_noisy = self.sample_noisy_image(x, t, noise)
        x_output = self.predict_noise(x_noisy, context, t)

        return x_output

    def get_loss(
        target: torch.Tensor,
        prediction: torch.Tensor,
        loss_type: str,
        mean: bool = False,
    ):
        if loss_type == "l2":
            reduction = "mean" if mean else "none"
            loss = F.mse_loss(target, prediction, reduction=reduction)
        else:
            loss = (target - prediction).abs()
            if mean:
                loss = loss.mean()

        return loss

    def compute_loss(
        self,
        x: torch.Tensor,
        context: np.ndarray,
        noise: torch.Tensor,
        t: torch.Tensor,
    ):
        """Calculate the loss"""

        predicted_noise = self(x, context, t, noise)

        loss_simple = self.get_loss(
            noise, predicted_noise, self.loss_type, mean=False
        ).mean(dim=(1, 2, 3))
        logvar_t = self.logvar[t].to(loss_simple.device)
        logvar_t = torch.clamp(logvar_t, -10.0, 10.0)
        scaled_loss = loss_simple / torch.exp(logvar_t) + logvar_t

        loss = self.l_simple_weight * scaled_loss.mean()

        self.log(f"loss_simple", loss, on_epoch=True, logger=True)

        loss_vlb = self.get_loss(
            noise, predicted_noise, self.loss_type, mean=False
        ).mean(dim=(1, 2, 3))
        loss_vlb = self.elbo_weight * (self.lvlb_weights[t] * loss_vlb).mean()
        self.log(f"loss_vlb", loss_vlb, on_epoch=True, logger=True)

        loss += loss_vlb

        return loss

    def training_step(self, batch, batch_idx):
        x, context = batch

        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        # Forward pass through the model
        with torch.no_grad():
            x_start, _, _ = self.first_stage_encoder(x)
            noise = torch.randn_like(x_start, device=self.device)

        loss = self.compute_loss(x_start, context, noise, t)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
