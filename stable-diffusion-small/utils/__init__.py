import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from .diffusion import make_beta_schedule, timestep_embedding
from .utils import (
    extract_into_tensor,
    instantiate_object,
    load_checkpoint,
    load_first_stage_encoder,
    load_images_to_tensor,
    log_reconstruction,
    save_checkpoint,
    tensor_to_pil_images,
)


def setup_logger(name=__name__, level=logging.INFO, log_dir="logs"):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(log_dir, today)
        os.makedirs(log_path, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=os.path.join(log_path, f"{name}.log"),
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=5,  # Keep 5 backup files
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = setup_logger()
