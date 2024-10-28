import logging

from .diffusion import make_beta_schedule, timestep_embedding
from .utils import (extract_into_tensor, instantiate_object,
                    load_images_to_tensor, load_trained_model,
                    tensor_to_pil_images)


def setup_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)

        # Define a format for the log messages
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger


logger = setup_logger()
