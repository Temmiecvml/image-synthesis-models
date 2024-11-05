from functools import partial
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from utils import (
    extract_into_tensor,
    instantiate_object,
    make_beta_schedule,
    timestep_embedding,
)


class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        timestep_config,
        text_conditioner_config,
        beta_schedule,
        num_timesteps,
    ):
        super().__init__()
        self.unet = instantiate_object(unet_config)
        self.time_embedder = instantiate_object(timestep_config)
        self.text_conditioner = instantiate_object(
            text_conditioner_config, device=str(self.device)
        )
        self.text_conditioner_idle = False
        self.num_timesteps = int(num_timesteps)
        self.create_schedule(beta_schedule, num_timesteps)

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

    def idle_text_conditioner(self, idle_device: str, idle: bool = True):
        if idle:
            self.text_conditioner = self.text_conditioner.set_device(idle_device)
            self.text_conditioner_idle = True
        else:
            self.text_conditioner = self.text_conditioner.set_device(self.device)
            self.text_conditioner_idle = False

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
        if not self.text_conditioner_idle:
            context = self.text_conditioner(context)
        else:
            raise ValueError("Text conditioner is idle")

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

    def forward(
        self,
        x: torch.Tensor,
        context: List[str],
        noise: Optional[torch.Tensor] = None,
    ):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        context = self.get_context_embedding(context)
        x_noisy = self.sample_noisy_image(x, t, noise)
        t_embeddings = self.get_timestep_embedding(t)
        x_output = self.unet(x_noisy, context, t_embeddings)

        return x_output

    def apply_model(self, x_noisy, context, t):
        context = self.get_context_embedding(context)
        t_embeddings = self.get_timestep_embedding(t)
        x_output = self.unet(x_noisy, context, t_embeddings)

        return x_output

    def training_step(self, batch, batch_idx):
        loss = 0

        return loss
