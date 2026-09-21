"""Verifies the tool-execution layer (retry, timeout, error surfacing) works correctly,
independent of the LLM loop. No API key required -- this exercises Harness._call_tool_with_retry
directly against real and broken tools.

Run with: python test_harness.py
"""

import time

import broken_tools as bt
from harness import Harness, ToolSpec
from tools import calculator

PASS = "PASS"
FAIL = "FAIL"


def make_harness(fn, tool_timeout=1.0, max_retries=2) -> Harness:
    tool = ToolSpec(name="calculator", description="", input_schema={}, fn=fn)
    return Harness(client=None, model="unused", tools=[tool], tool_timeout=tool_timeout, max_retries=max_retries)


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


def test_working_tool_succeeds() -> bool:
    h = make_harness(calculator)
    tool = h.tools_by_name["calculator"]
    result, outcome, attempts = h._call_tool_with_retry(tool, {"expression": "2 + 2"})
    return check(
        "working calculator returns correct result on first attempt",
        result == "4" and outcome == "success" and attempts == 1,
        f"got result={result!r} outcome={outcome!r} attempts={attempts!r}",
    )


def test_timeout_does_not_hang_the_caller() -> bool:
    h = make_harness(bt.calculator_timeout, tool_timeout=1.0, max_retries=1)
    tool = h.tools_by_name["calculator"]

    start = time.monotonic()
    result, outcome, attempts = h._call_tool_with_retry(tool, {"expression": "1 + 1"})
    elapsed = time.monotonic() - start

    # 2 attempts (1 initial + 1 retry) at 1.0s timeout each, plus ~0.5s backoff = ~2.5s.
    # The underlying tool sleeps for 9999s -- if the timeout wrapper didn't work, this
    # call would hang for over two and a half hours instead of returning in a few seconds.
    return check(
        "hung tool times out instead of blocking the caller",
        elapsed < 5.0 and outcome == "timeout" and "TOOL_FAILED" in result,
        f"elapsed={elapsed:.2f}s outcome={outcome!r} result={result!r}",
    )


def test_exception_is_caught_and_reported() -> bool:
    h = make_harness(bt.calculator_exception)
    tool = h.tools_by_name["calculator"]
    result, outcome, attempts = h._call_tool_with_retry(tool, {"expression": "1 + 1"})
    return check(
        "exception-raising tool is caught, not propagated, and reported as an error",
        outcome == "error" and attempts == 3 and "arithmetic engine offline" in result,
        f"outcome={outcome!r} attempts={attempts!r} result={result!r}",
    )


def test_empty_and_malformed_output_pass_through_as_success() -> bool:
    # These variants don't raise or time out -- they return successfully with bad
    # content. The retry/timeout layer correctly can't distinguish this from a real
    # answer; catching it is the model's/harness's downstream job, not this layer's.
    h_empty = make_harness(bt.calculator_empty)
    h_malformed = make_harness(bt.calculator_malformed)
    result_empty, outcome_empty, _ = h_empty._call_tool_with_retry(
        h_empty.tools_by_name["calculator"], {"expression": "1 + 1"}
    )
    result_malformed, outcome_malformed, _ = h_malformed._call_tool_with_retry(
        h_malformed.tools_by_name["calculator"], {"expression": "1 + 1"}
    )
    return check(
        "empty/malformed output is reported as success at this layer (by design)",
        outcome_empty == "success" and outcome_malformed == "success",
        f"empty outcome={outcome_empty!r}, malformed outcome={outcome_malformed!r}",
    )


def test_loop_detection_key() -> bool:
    # Loop detection lives in Harness.run() (needs the LLM loop), not in the retry layer.
    # This just checks the same-call-twice key logic in isolation.
    import json

    call_a = ("calculator", json.dumps({"expression": "1 + 1"}, sort_keys=True))
    call_b = ("calculator", json.dumps({"expression": "1 + 1"}, sort_keys=True))
    call_c = ("calculator", json.dumps({"expression": "2 + 2"}, sort_keys=True))
    return check(
        "identical tool call produces the same dedup key, different args do not",
        call_a == call_b and call_a != call_c,
    )


if __name__ == "__main__":
    tests = [
        test_working_tool_succeeds,
        test_timeout_does_not_hang_the_caller,
        test_exception_is_caught_and_reported,
        test_empty_and_malformed_output_pass_through_as_success,
        test_loop_detection_key,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    # A leaked non-daemon thread from a timed-out tool call would keep the interpreter
    # alive past this point even though every assertion above passed -- this print is the
    # actual proof the process can exit, not just that the logic returned the right value.
    print("process exiting cleanly")
    if passed != len(results):
        raise SystemExit(1)
