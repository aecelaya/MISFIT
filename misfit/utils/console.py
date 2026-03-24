"""Shared Rich console and output formatting helpers for MISFIT.

All user-facing messages should go through these helpers so that formatting
(colours, prefixes, spacing) is consistent across every CLI command and
module. Import the shared ``console`` singleton rather than creating new
Console instances per module.
"""
from rich.console import Console

# Module-level singleton — import and use directly from any MISFIT module.
console = Console()


def print_section_header(title: str) -> None:
    """Print a bold section header with surrounding newlines.

    Args:
        title: Header text.
    """
    console.print(f"\n[bold]{title}[/bold]\n")


def print_info(msg: str) -> None:
    """Print a plain informational message.

    Args:
        msg: Message text.
    """
    console.print(msg)


def print_warning(msg: str) -> None:
    """Print a yellow warning message.

    Args:
        msg: Warning text.
    """
    console.print(f"[yellow]Warning:[/yellow] {msg}")


def print_error(msg: str) -> None:
    """Print a bold-red error message.

    Args:
        msg: Error text.
    """
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str) -> None:
    """Print a green success message with a checkmark.

    Args:
        msg: Success text.
    """
    console.print(f"[green]\u2713[/green] {msg}")
