"""Rich logging utilities for the voice assistant."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


# Initialize Rich Console
console = Console()

# Store logs in memory
log_messages: List[str] = []

# Logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def log(message: str, title: str, style: str) -> None:
    """Log a message to the Rich console and persist it for later export."""
    console.print(Panel(Markdown(f"**{message}**"), border_style=style, expand=False, title=title))
    log_messages.append(f"[{title}] {message}")


def save_log() -> None:
    """Save collected log messages to a timestamped file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    # Organize logs by year/month
    year_month_dir = LOGS_DIR / str(now.year) / f"{now.month:02d}"
    year_month_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%d-%H%M%S")
    filename = year_month_dir / f"{timestamp}.log"

    with open(filename, "w", encoding="utf-8") as f:
        for message in log_messages:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    # Update latest.log symlink
    latest_link = LOGS_DIR / "latest.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)
    except OSError:
        pass  # Symlinks may not work on all platforms

    print(f"Log saved to {filename}")
