"""Compatibility entrypoint for the real VK differential-equation bot.

Historically this file was a tiny echo bot. Keep the filename usable, but
delegate to src/bot.py so running it exercises the actual MAS pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bot import main  # noqa: E402


if __name__ == "__main__":
    main()
