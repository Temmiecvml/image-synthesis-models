from functools import partial
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import (extract_into_tensor, load_first_stage_encoder,
                   load_images_to_tensor, logger, make_beta_schedule,
                   tensor_to_pil_images)


def process_prompt(prompt: str, cfg_scale: float):
    if prompt:
        prompt = [p.strip() for p in prompt.split("\n")]
        logger.info(f"Prompt: {prompt}")
    else:
        prompt = [""]

    if cfg_scale > 0:
        logger.info(f"cfg_scale: {cfg_scale}, therefore adding uncond_prompt to prompt")
        prompt += ["" for _ in range(len(prompt))]

    return prompt


def process_image_paths(
    autoencoder_model, image_paths: str, image_width: int, latent_dim: int
):
    images = load_images_to_tensor(image_paths, image_width)
    logger.info(f"Images loaded: {images.shape}")
    assert images.shape[2] == image_width, "Image width mismatch with config"
    image_embeddings, _, _ = autoencoder_model(images)
    logger.info(f"Image embeddings: {image_embeddings.shape}")
    assert (
        image_embeddings.shape[2] == latent_dim
    ), "embeddings dimension mismatch with config"

    return image_embeddings


def load_models(config):
    autoencoder_model = load_first_stage_encoder(config.autoencoder)
    ddpm_model = load_first_stage_encoder(config.ddpm)

    return {"autoencoder": autoencoder_model, "ddpm": ddpm_model}


def generate(
    config: str,
    prompt: Optional[str] = None,
    image_paths: Optional[str] = None,
    models: dict = None,
    strength: float = 0.8,
    cfg_scale: float = 7.5,
    sampler: str = "ddpm",
    n_inference_steps: int = 50,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
):
    """Generates images from text prompts using the diffusion model.

    Args:
        prompt (List[str]): individual prompts are seperated by "\n"
        input_image (Optional[torch.Tensor], optional): _description_. Defaults to None.
        strength (float, optional): _description_. Defaults to 0.8.
        cfg_scale (float, optional): _description_. Defaults to 7.5.
        sampler (str, optional): _description_. Defaults to "ddpm".
        n_inferences_steps (int, optional): _description_. Defaults to 50.
        seed (Optional[int], optional): _description_. Defaults to None.
        device (Optional[str], optional): _description_. Defaults to None.
        idle_device (Optional[str], optional): _description_. Defaults to None.
    """
    config = OmegaConf.load(config)
    image_width = config.inference.image_width
    latent_width = config.inference.latent_width
    latent_dim = config.inference.latent_dim

    autoencoder_model = models["autoencoder"]
    prompt = process_prompt(prompt, cfg_scale)
    fac = 2 if cfg_scale > 0 else 1

    if image_paths:
        image_embeddings = process_image_paths(
            autoencoder_model, image_paths, image_width, latent_dim
        )
        if not prompt:
            prompt = ["" for i in range(image_embeddings.shape[0] * fac)]
        else:
            assert (
                len(prompt) == image_embeddings.shape[0] * fac
            ), "Prompt length mismatch with number of images"

    with torch.no_grad():
        assert 0 < strength <= 1, "Strength must be in (0, 1]"
        num_images = len(prompt) // fac
        generator = torch.Generator(device=idle_device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        if sampler == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler} not supported.")

        if image_paths:
            sampler.set_strength(strength)
            latents = sampler.add_noise(image_embeddings, sampler.timesteps[0])
        else:
            latents = torch.randn(
                num_images,
                latent_dim,
                latent_width,
                latent_width,
                device=device,
                generator=generator,
            )

        ddpm_model = models["ddpm"]

        for i, timestep in tqdm(enumerate(sampler.timesteps)):
            if cfg_scale > 0:
                latents = latents.repeat(2, 1, 1, 1)

            logger.info(f"Generating image at timestep: {timestep}")
            output = ddpm_model.predict_noise(latents, prompt, timestep)

            if cfg_scale > 0:
                cond_output, uncond_output = torch.chunk(output, 2, dim=0)
                output = cfg_scale * (cond_output - uncond_output) + uncond_output

            # sampler removes noise predicted by model
            latents = sampler.step(timestep, latents, output)

        # offload ddpm_model to cpu
        tensor_images = autoencoder_model.decode(latents)
        images = tensor_to_pil_images(tensor_images)

        return images


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        n_inference_steps: int,
        schedule: str = "linear",
        num_timesteps: int = 1000,
    ):
        self.generator = generator
        self.create_schedule(schedule, num_timesteps)
        self.set_inference_steps(n_inference_steps, num_timesteps)

    def create_schedule(self, beta_schedule, num_timesteps):
        self.betas = make_beta_schedule(
            beta_schedule,
            num_timesteps,
            linear_start=1e-4,
            linear_end=2e-2,
        )
        self.alphas = torch.tensor(1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor(1.0), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def set_inference_steps(self, n_inference_steps, num_timesteps):
        self.n_inference_steps = n_inference_steps
        step_ratio = num_timesteps // n_inference_steps
        self.timesteps = (
            torch.arange(0, num_timesteps, step_ratio).round().to(torch.int32).flip(0)
        )

    def set_strength(self, strength):
        self.strength = strength
        self.start_step = int(self.n_inference_steps * (1 - strength))
        self.timesteps = self.timesteps[self.start_step :]

    def add_noise(self, latents, t):
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device=latents.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device=latents.device
        )
        noise = torch.randn_like(latents, generator=self.generator)

        sqrt_alphas_cum_prod_t = extract_into_tensor(
            sqrt_alphas_cumprod, t, latents.shape
        )
        sqrt_one_minus_alphas_cum_prod_t = extract_into_tensor(
            sqrt_one_minus_alphas_cumprod, t, latents.shape
        )
        # simmilar to the reparameterization trick: mean + std * noise
        noisy_image = (
            sqrt_alphas_cum_prod_t * latents + sqrt_one_minus_alphas_cum_prod_t * noise
        )

        return noisy_image

    def step(self, t, x_t, predicted_noise, variant="scaled_beta"):
        """
        Sample previous step latent of diffusion model.
        Paper: https://arxiv.org/pdf/2006.11239
        Denoising Diffusion Probabilistic Models

        Args:
            t (int): current timestep
            x_t (Tensor): current noisy latent
            predicted_noise (Tensor): model's noise prediction
            variant (str, optional): use scaled_beta or standard variant.
        Returns:
            x_t_minus_1 (Tensor): latent at previous timestep t-1
        """
        to_device = partial(torch.tensor, device=x_t.device)
        alphas_t = to_device(extract_into_tensor(self.alphas, t, x_t.shape))
        betas_t = to_device(extract_into_tensor(self.betas, t, x_t.shape))
        sqrt_alphas_cumprod_t = to_device(
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        )
        sqrt_one_minus_alphas_cumprod_t = to_device(
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )

        # Mean for the reverse process (Equation 11 in the DDPM paper)
        mu_t = (
            x_t - (betas_t * predicted_noise) / sqrt_one_minus_alphas_cumprod_t
        ) / sqrt_alphas_cumprod_t

        # Variance of the reverse process
        var_t = torch.clamp(betas_t, min=1e-20)

        if variant == "scaled_beta":
            alphas_cumprod_prev_t = to_device(
                extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
            )
            alphas_cumprod_t = to_device(
                extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
            )

            # Alternative mean calculation (Equation 7 in the DDPM paper)
            x_o = (
                x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise
            ) / sqrt_alphas_cumprod_t

            # Equation for mean under scaled_beta variant
            mu_t = (
                x_o * torch.sqrt(alphas_cumprod_prev_t) * betas_t
                + x_t * torch.sqrt(alphas_t) * (1 - alphas_cumprod_prev_t)
            ) / (1 - alphas_cumprod_t)

            var_t = betas_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            var_t = torch.clamp(var_t, min=1e-20)

        # Add noise for stochasticity, except at the final step
        noise = (
            torch.randn_like(predicted_noise, generator=self.generator) if t > 0 else 0
        )

        sigma_t = torch.sqrt(var_t)
        x_t_minus_1 = mu_t + sigma_t * noise

        return x_t_minus_1
