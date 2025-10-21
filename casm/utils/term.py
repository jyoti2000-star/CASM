from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
from casm.utils.colors import Colors

console = Console()

def _is_minimal() -> bool:
    env = os.environ.get('CASM_MINIMAL_UI')
    if env is not None:
        return env.strip() in ('1', 'true', 'yes', 'on')
    return bool(getattr(Colors, 'MINIMAL', False))

def print_stage(step: int, total: int, message: str):
    if _is_minimal():
        console.print(f"[{step}/{total}] {message}")
    else:
        console.print(f"[cyan]●[/cyan] [bold]{step}/{total}[/bold] {message}")

def print_info(message: str):
    if _is_minimal():
        return
    console.print(f"[yellow]Info:[/yellow] {message}")

def print_error(message: str):
    if _is_minimal():
        console.print(f"[ERROR] {message}")
        return
    console.print(f"[red]Error:[/red] {message}")

def print_warning(message: str):
    if _is_minimal():
        console.print(f"[WARN] {message}")
        return
    console.print(f"[#9b59b6]Warning:[/#9b59b6] {message}")

def print_success(message: str):
    if _is_minimal():
        console.print(f"[OK] {message}")
        return
    console.print(f"[green]Success:[/green] {message}")
