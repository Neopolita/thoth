import contextlib
import logging
import os


@contextlib.contextmanager
def suppress_logs():
    logging.disable(logging.CRITICAL)
    with (
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        try:
            yield
        finally:
            logging.disable(logging.NOTSET)


def count_tokens(text: str) -> int:
    return len(text) // 4


def truncate(text: str, max_length: int = 256) -> str:
    """Truncate text to max_length, appending '...' if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
