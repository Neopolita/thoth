from thoth.d2l.doc_to_lora import process_doc_to_lora
from thoth.logger import get_logger
from thoth.signatures import QueryMemorySignature
from thoth.utils import suppress_logs
import atexit
import dspy
import os
import subprocess

logger = get_logger()


qm_lm = dspy.LM(
    "huggingface/neopolita/Mistral-7B-Instruct-v0.2-mlx",
    api_base="http://127.0.0.1:8080/v1",
    api_key="",
    temperature=0.0,
)

query_memory_mod = dspy.Predict(QueryMemorySignature)


_memory_server_process = None


def _cleanup_memory_server():
    global _memory_server_process
    if _memory_server_process is not None:
        _memory_server_process.terminate()
        _memory_server_process.wait()
        _memory_server_process = None


atexit.register(_cleanup_memory_server)


def restart_memory_server():
    if not os.path.exists(".tmp/d2l"):
        return

    global _memory_server_process
    _cleanup_memory_server()
    _memory_server_process = subprocess.Popen(
        [
            "mlx_lm.server",
            "--model",
            "neopolita/Mistral-7B-Instruct-v0.2-mlx",
            "--adapter-path",
            ".tmp/d2l",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def query_memory(query: str) -> str:
    """Query memory with a natural language query."""
    try:
        res = query_memory_mod(query=query, lm=qm_lm)
        return res.result
    except Exception as e:
        logger.error(f"Error querying memory: {str(e)}")
        return f"Error: {str(e)}"


def add_memory(data: str) -> str:
    """Add given data to memory."""
    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")
    with open(".tmp/memory.txt", "a") as f:
        f.write(data + "\n")
    with open(".tmp/memory.txt", "r") as f:
        memory = f.read()
    try:
        with suppress_logs():
            process_doc_to_lora(memory, output_dir=".tmp/d2l")
        restart_memory_server()
        return "Memory added successfully."
    except Exception as e:
        logger.error(f"Error adding memory: {str(e)}")
        return f"Error: {str(e)}"
