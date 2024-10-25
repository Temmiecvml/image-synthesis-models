import torch
import torch.nn as nn

from utils import instantiate_object


class DDPM(nn.Module):
    def __init__(
        self,
        time_embedding_config,
        unet_config,
        unet_output_config,
        beta=0.9,
        noise_level=0.1,
    ):
        super().__init__()
        self.time_embedding_config = instantiate_object(time_embedding_config)
        self.unet = instantiate_object(unet_config)
        self.beta = beta
        self.noise_level = noise_level

    def forward(self, latent: torch.Tensor, context: torch.Tensor, t: torch.Tensor):

        t = self.time_embedding(t)
        x = self.unet(latent, context, t)
        return x
