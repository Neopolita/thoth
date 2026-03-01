from click import clear
from dspy.utils.callback import BaseCallback
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from thoth.display import draw_logo, draw_stats
from thoth.logger import get_logger
from thoth.memory import clear_memory, add_memory, query_memory, restart_memory_server
from thoth.signatures import AgentSignature
from thoth.tools import web_fetch, web_search
from thoth.utils import count_tokens, truncate
from typing import Any
import dspy
import logging
import os

logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)
logger = get_logger()
console = Console()

API_KEY = os.getenv("MISTRAL_API_KEY")


class ToolMonitor(BaseCallback):
    def __init__(self):
        self._active_tools: dict[str, Any] = {}

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        """logger.debug(
            f"Tool {instance.name} called with inputs: {truncate(str(inputs))}"
        )"""
        self._active_tools[call_id] = instance
        if instance.name != "finish":
            pass
            # console.print(f"[dim]Using {instance.name}...[/dim]")

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        instance = self._active_tools.pop(call_id, None)
        name = instance.name if instance else "unknown"
        if name == "finish":
            return

        if exception:
            console.print(f"[red]Error in {name}: {exception}[/red]")
        else:
            console.print(f"\n[dim]{name} → {truncate(str(outputs))}[/dim]")


clear_memory()

TOOLS = [add_memory, query_memory, web_search, web_fetch]

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
lm = dspy.LM("mistral/mistral-large-2512", api_key=API_KEY, max_tokens=16384)
dspy.configure(lm=lm, callbacks=[ToolMonitor()])

mod = dspy.ReAct(AgentSignature, tools=TOOLS)


def get_history_size() -> int:
    size = 0
    for item in history.messages:
        size += count_tokens(item["prompt"]) + count_tokens(item["result"])
    return size


def get_memory_size() -> int:
    if not os.path.exists(".tmp/memory.txt"):
        return 0
    with open(".tmp/memory.txt", "r") as f:
        memory = f.read()
    return count_tokens(memory)


console.clear()
draw_logo(console)
console.print("[magenta]type quit to exit[/magenta]")
console.print(
    "[yellow]Built for the Mistral AI Worldwide Hackathon 2026[/yellow]\nThoth uses Sakana.ai Doc-to-LoRA to ease context size pressure, instead of feeding entire documents into the context window, it converts them into LoRA adapters on the fly, freeing up the context for reasoning and conversation while retaining document knowledge in the model's weights.\n",
    justify="right",
)

with open("data/context.md") as f:
    context = f.read()

history = dspy.History(messages=[])


def process_commands(prompt: str) -> bool:
    if not prompt.startswith("/"):
        return False
    command = prompt[1:].split(" ")[0]
    if command == "clear_memory":
        clear_memory()
        console.print("\n[cyan]⏺ Memory cleared[/cyan]")
    elif command == "add_memory":
        data = prompt[len("/add_memory ") :]
        res = add_memory(data)
        console.print(f"\n[cyan]⏺ {res}[/cyan]")
    else:
        console.print(f"\n[magenta]⏺ Unknown command: {command}[/magenta]")

    return True


while True:
    draw_stats(console, get_history_size(), get_memory_size())
    prompt = console.input("\n[bold cyan]> [/bold cyan]")
    if prompt.strip().lower() in ("quit", "exit"):
        break

    if process_commands(prompt):
        continue

    res = mod(context=context, history=history, task=prompt)
    console.print("\n")
    console.print(Panel(res.result, title="Thoth", border_style="magenta"))
    history_item = {"prompt": prompt, "result": res.result}
    history.messages.append(history_item)
