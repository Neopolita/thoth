from rich.console import Console
from rich.style import Style
from rich.text import Text


def draw_logo(console: Console) -> None:
    with open("data/logo.txt") as f:
        logo = f.read()
    start_rgb = (180, 40, 220)  # purple
    end_rgb = (40, 200, 220)  # cyan
    gradient = Text()
    for line in logo.splitlines():
        max_col = max(len(line) - 1, 1)
        for i, ch in enumerate(line):
            t = i / max_col
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
            gradient.append(ch, Style(color=f"rgb({r},{g},{b})", bold=True))
        gradient.append("\n")
    console.print(gradient)


def draw_stats(console: Console, history_size: int, memory_size: int) -> None:
    console.print(
        f"[bold yellow]Context[/bold yellow]: ~{history_size} [dim]|[/dim] [bold yellow]LoRA memory[/bold yellow]: ~{memory_size}",
        justify="right",
    )
