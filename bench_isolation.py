"""Offline benchmark: does a timed-out tool call actually stop, or just get abandoned?

Runs the same hung tool under both isolation modes and checks for a marker file the
tool writes if it ever wakes up. Also measures what process mode costs on a fast tool.

No API key required -- this exercises Harness._call_tool_with_retry directly.

Run with: python bench_isolation.py
"""

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import broken_tools as bt
from harness import Harness, ToolSpec
from tools import calculator

HUNG_TIMEOUT = 1.0
TOOL_WAKE_SECONDS = 2.5
MARKER_GRACE_SECONDS = 2.0


def make_harness(fn, isolation: str, tool_timeout: float, max_retries: int = 0) -> Harness:
    tool = ToolSpec(name="calculator", description="", input_schema={}, fn=fn)
    return Harness(
        client=None,
        model="unused",
        tools=[tool],
        tool_timeout=tool_timeout,
        max_retries=max_retries,
        isolation=isolation,
    )


def pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check that treats an unreaped zombie as dead."""
    if pid is None:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        if stat_path.exists() and stat_path.read_text().split()[2] == "Z":
            return False
    except OSError:
        pass
    return True


def run_hung_tool(isolation: str, marker_path: Path) -> dict:
    harness = make_harness(bt.calculator_slow, isolation, HUNG_TIMEOUT)
    arguments = {"expression": "2+2", "marker_path": str(marker_path), "sleep_seconds": TOOL_WAKE_SECONDS}

    started = time.perf_counter()
    result, outcome, attempts = harness._call_tool_with_retry(harness.tools_by_name["calculator"], arguments)
    elapsed = time.perf_counter() - started

    pid = harness.last_process_pid
    alive_on_return = pid_is_alive(pid) if pid is not None else None
    marker_on_return = marker_path.exists()

    # Give the abandoned work enough time to finish and write its marker.
    deadline = time.perf_counter() + MARKER_GRACE_SECONDS
    while time.perf_counter() < deadline and not marker_path.exists():
        time.sleep(0.05)

    return {
        "isolation": isolation,
        "returned_after_s": round(elapsed, 2),
        "outcome": outcome,
        "attempts": attempts,
        "result": result,
        "worker_pid": pid,
        "worker_alive_on_return": alive_on_return,
        "marker_on_return": marker_on_return,
        "marker_after_grace": marker_path.exists(),
    }


def measure_fast_tool(isolation: str, repeats: int = 5) -> dict:
    harness = make_harness(calculator, isolation, tool_timeout=10.0)
    tool = harness.tools_by_name["calculator"]

    harness._call_tool_with_retry(tool, {"expression": "2+2"})  # warm the code path / import

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result, outcome, _ = harness._call_tool_with_retry(tool, {"expression": "2+2"})
        samples.append((time.perf_counter() - started) * 1000)
        assert result == "4" and outcome == "success", (result, outcome)

    return {
        "isolation": isolation,
        "repeats": repeats,
        "median_ms": round(statistics.median(samples), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
    }


def run_error_case(isolation: str) -> dict:
    harness = make_harness(bt.calculator_exception, isolation, tool_timeout=5.0)
    result, outcome, attempts = harness._call_tool_with_retry(
        harness.tools_by_name["calculator"], {"expression": "2+2"}
    )
    return {"isolation": isolation, "outcome": outcome, "attempts": attempts, "result": result}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bulkhead_isolation_") as tmp:
        tmp_dir = Path(tmp)
        hung = []
        for isolation in ("thread", "process"):
            marker = tmp_dir / f"marker_{isolation}.txt"
            hung.append(run_hung_tool(isolation, marker))

        timings = [measure_fast_tool(mode) for mode in ("thread", "process")]
        errors = [run_error_case(mode) for mode in ("thread", "process")]

    report = {"hung_tool": hung, "fast_tool": timings, "error_propagation": errors}

    print("Hung tool (timeout %.1fs, tool sleeps %.1fs then writes a marker file)"
          % (HUNG_TIMEOUT, TOOL_WAKE_SECONDS))
    print(f"{'isolation':<10} {'returned':>9} {'outcome':<8} {'worker alive':>13} "
          f"{'marker at return':>17} {'marker after +%.0fs':>19}" % MARKER_GRACE_SECONDS)
    for row in hung:
        alive = {None: "n/a", True: "yes", False: "no"}[row["worker_alive_on_return"]]
        print(f"{row['isolation']:<10} {row['returned_after_s']:>8.2f}s {row['outcome']:<8} "
              f"{alive:>13} {str(row['marker_on_return']):>17} {str(row['marker_after_grace']):>19}")

    print("\nFast tool (expression '2+2'), median of 5 calls")
    for row in timings:
        print(f"  {row['isolation']:<8} median {row['median_ms']:>6.1f}ms "
              f"(min {row['min_ms']:.1f}, max {row['max_ms']:.1f})")

    print("\nException propagation (tool raises RuntimeError('arithmetic engine offline'))")
    for row in errors:
        print(f"  {row['isolation']:<8} outcome={row['outcome']} attempts={row['attempts']} "
              f"result={row['result']!r}")

    Path("isolation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote isolation_results.json")

    checks = [
        (
            "thread mode returns before the tool wakes",
            hung[0]["marker_on_return"] is False,
        ),
        (
            "thread mode's abandoned work keeps running and finishes late",
            hung[0]["marker_after_grace"] is True,
        ),
        (
            "process mode returns before the tool wakes",
            hung[1]["marker_on_return"] is False,
        ),
        (
            "process mode's worker is dead when the harness returns",
            hung[1]["worker_alive_on_return"] is False,
        ),
        (
            "process mode killed the work before it could finish",
            hung[1]["marker_after_grace"] is False,
        ),
        (
            "both modes still return correct results for a fast tool",
            all(row["median_ms"] > 0 for row in timings),
        ),
        (
            "both modes surface the original exception text",
            all("arithmetic engine offline" in row["result"] for row in errors),
        ),
    ]

    print()
    ok = True
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
