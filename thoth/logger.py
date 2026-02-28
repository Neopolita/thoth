import coloredlogs
import logging

logger = logging.getLogger("thoth")
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S", logger=logger
)


def get_logger() -> logging.Logger:
    return logger


def log_tensor(name, tensor, level: str = "DEBUG") -> None:
    log = f"{name}, {tensor.shape}, {tensor.dtype}, {tensor.device}, m: {tensor.mean():.5f}, s: {tensor.std():.5f}, n: {tensor.norm():.5f}"
    logger.log(getattr(logging, level.upper()), log)
