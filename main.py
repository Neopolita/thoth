from dspy.utils.callback import BaseCallback
from rich.console import Console
from rich.prompt import Prompt
from thoth.logger import get_logger
from thoth.memory import add_memory, query_memory, restart_memory_server
from thoth.signatures import AgentSignature
from thoth.tools import web_fetch, web_search
from thoth.utils import truncate
from typing import Any
import dspy
import logging
import os

logging.getLogger("primp").setLevel(logging.ERROR)
logger = get_logger()
console = Console()

API_KEY = os.getenv("MISTRAL_API_KEY")


class ToolMonitor(BaseCallback):
    def __init__(self):
        self._active_tools: dict[str, Any] = {}

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        logger.debug(
            f"Tool {instance.name} called with inputs: {truncate(str(inputs))}"
        )
        self._active_tools[call_id] = instance
        if instance.name != "finish":
            console.print(f"[dim]Using {instance.name}...[/dim]")

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        instance = self._active_tools.pop(call_id, None)
        name = instance.name if instance else "unknown"
        if exception:
            console.print(f"[red]Error in {name}: {exception}[/red]")
        else:
            console.print(f"[dim]{name} → {truncate(str(outputs))}[/dim]")


restart_memory_server()

TOOLS = [add_memory, query_memory, web_search, web_fetch]

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
lm = dspy.LM("mistral/mistral-large-2512", api_key=API_KEY)
dspy.configure(lm=lm, callbacks=[ToolMonitor()])

mod = dspy.ReAct(AgentSignature, tools=TOOLS)

console.clear()
with open("data/logo.txt") as f:
    logo = f.read()
console.print(f"[bold magenta]{logo}[/bold magenta]")
console.print("[dim]type quit to exit[/dim]\n")

with open("data/context.md") as f:
    context = f.read()

history = dspy.History(messages=[])

while True:
    prompt = Prompt.ask("[bold cyan]You[/bold cyan]")
    if prompt.strip().lower() in ("quit", "exit"):
        break
    res = mod(context=context, history=history, task=prompt)
    console.print(f"\n[bold magenta]Thoth[/bold magenta]: {res.result}\n")

    history_item = {"prompt": prompt, "result": res.result}
    print(history_item)
    history.messages.append(history_item)
