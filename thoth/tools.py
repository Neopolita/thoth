from ddgs import DDGS
from rich.prompt import Confirm
import functools
import trafilatura


def require_permission(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        answer = Confirm.ask(f"Allow {func.__name__}({kwargs})?")
        if not answer:
            return "Tool call denied by user."
        return func(*args, **kwargs)

    return wrapper


@require_permission
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web and return top results."""
    results = DDGS().text(query, max_results=max_results)
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


@require_permission
def web_fetch(url: str) -> str:
    """Fetch and extract the main content of a web page."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return "Error: could not fetch URL"
    text = trafilatura.extract(downloaded)
    return text or "Error: could not extract content"
