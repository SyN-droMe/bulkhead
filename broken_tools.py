"""Broken variants of the `calculator` tool, one per failure scenario.

Each function has the same signature as `tools.calculator` so it can be
swapped in as a drop-in replacement when benchmarking the harness.
"""

import time
from pathlib import Path

from tools import calculator


def calculator_timeout(expression: str) -> str:
    """Hangs well past any reasonable deadline."""
    time.sleep(9999)
    return calculator(expression)  # never reached


def calculator_slow(expression: str, marker_path: str = "", sleep_seconds: float = 3.0) -> str:
    """Sleeps past the harness deadline, then proves whether it survived.

    If the harness killed the work, `marker_path` never appears. If it only stopped
    waiting (thread mode), the file shows up seconds after the harness moved on.
    """
    time.sleep(sleep_seconds)
    if marker_path:
        Path(marker_path).write_text("tool woke up and finished", encoding="utf-8")
    return calculator(expression)


def calculator_malformed(expression: str) -> str:
    """Returns a garbage, non-numeric string instead of a result."""
    return "{not a number, definitely broken output!!"


def calculator_empty(expression: str) -> str:
    """Silently returns nothing."""
    return ""


def calculator_exception(expression: str) -> str:
    """Raises mid-call."""
    raise RuntimeError("arithmetic engine offline")


def calculator_loop(expression: str) -> str:
    """Always asks to be called again with the same input, inducing a loop."""
    return "NEED_MORE_INFO: please retry this exact same calculation again."


def calculator_wrong(expression: str) -> str:
    """Succeeds, but silently returns an off-by-one wrong answer. The hardest case: there
    is no error signal at all, so the harness's retry/timeout/error-surfacing machinery
    cannot catch this by design. See the README limitations section.
    """
    correct = float(calculator(expression))
    return str(correct + 1)
