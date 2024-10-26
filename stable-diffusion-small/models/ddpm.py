import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils import extract_into_tensor, instantiate_object


class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        timestep_config,
        num_timesteps,
        beta=0.9,
        noise_level=0.1,
    ):
        super().__init__()
        self.unet = instantiate_object(unet_config)
        self.time_embedder = instantiate_object(timestep_config)
        self.num_timesteps = num_timesteps
        self.beta = beta
        self.noise_level = noise_level

    def get_context_embedding(self, context):
        bs = len(context)
        context = torch.randn(bs, 77, 640, device=self.device)

        return context

    def get_timestep_embedding(self, t):
        bs = len(t)
        # consine embedding passed through linear embedding with 192 * 4 output channels
        t = torch.randn(bs, 192, device=self.device)
        t = self.time_embedder(t)

        return t

    def sample_noisy_image(self, x, t):
        noise = torch.randn_like(x, device=self.device)
        # noisy_image =  (
        #     extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        #     + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        #     * noise
        # )

        return noise

    def _forward(self, x, context, t):
        x_noisy = self.sample_noisy_image(x, t)
        t_embeddings = self.get_timestep_embedding(t)
        x_output = self.unet(x_noisy, context, t_embeddings)

        loss = 0

        return loss

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ):

        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        context = self.get_context_embedding(context)
        loss = self._forward(x, context, t)

        return loss
