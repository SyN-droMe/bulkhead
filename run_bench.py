"""Runs one fixed task through the harness once per tool variant (a working
calculator, then each broken variant) and records what actually happened.

This is the evidence for "reliable" -- not a claim, a results table.
"""

import argparse
import json

from groq import Groq

import broken_tools as bt
from harness import Harness, ToolSpec
from tools import CALCULATOR_SPEC, LOOKUP_SPEC, calculator, lookup

TASK = (
    "Calculate (23 + 19) * 3 using the calculator tool. Then use the lookup tool to find "
    "out how many days are in a week, and tell me how many weeks (23 + 19) * 3 days is. "
    "Give me just the final number of weeks."
)
EXPECTED_ANSWER = "18"  # (23 + 19) * 3 = 126 days / 7 days-per-week = 18 weeks

VARIANTS = {
    "baseline (working calculator)": calculator,
    "timeout": bt.calculator_timeout,
    "malformed_output": bt.calculator_malformed,
    "silent_empty": bt.calculator_empty,
    "exception": bt.calculator_exception,
    "loop_inducing": bt.calculator_loop,
    "plausible_wrong_answer": bt.calculator_wrong,
}

FAILURE_MARKERS = ("fail", "error", "unable", "cannot", "couldn't", "not able", "issue", "wasn't able")


def build_tools(calc_fn) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculator",
            description=CALCULATOR_SPEC["description"],
            input_schema=CALCULATOR_SPEC["input_schema"],
            fn=calc_fn,
        ),
        ToolSpec(
            name="lookup",
            description=LOOKUP_SPEC["description"],
            input_schema=LOOKUP_SPEC["input_schema"],
            fn=lookup,
        ),
    ]


def classify(run_result, expected: str) -> str:
    if run_result.status == "loop_detected":
        return "loop caught (harness stopped it)"
    if run_result.status == "max_steps":
        return "gave up (hit step cap without answering)"

    text = run_result.final_text or ""
    got_it_right = expected in text
    any_tool_failed = any(tc.outcome != "success" for tc in run_result.tool_calls)
    mentions_failure = any(marker in text.lower() for marker in FAILURE_MARKERS)

    if got_it_right and any_tool_failed:
        return "recovered (retried past the failure to the right answer)"
    if got_it_right and not any_tool_failed:
        return "correct (no failure occurred)"
    if not got_it_right and any_tool_failed and mentions_failure:
        return "clear error (harness/model flagged the failure honestly)"
    if not got_it_right and any_tool_failed:
        return "unclear failure (tool errored, model didn't explain)"
    return "SILENTLY WRONG (tool reported success but the answer is wrong)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    client = Groq()
    results = []

    for variant_name, calc_fn in VARIANTS.items():
        print(f"\n=== {variant_name} ===")
        harness = Harness(
            client=client,
            model=args.model,
            tools=build_tools(calc_fn),
            max_steps=6,
            max_retries=2,
            tool_timeout=3.0,
        )
        try:
            result = harness.run(TASK)
            outcome = classify(result, EXPECTED_ANSWER)
            row = {
                "variant": variant_name,
                "status": result.status,
                "outcome": outcome,
                "steps": result.steps,
                "tool_call_outcomes": [tc.outcome for tc in result.tool_calls],
                "final_text": (result.final_text or "")[:300],
            }
        except Exception as exc:
            row = {
                "variant": variant_name,
                "status": "CRASHED",
                "outcome": "crashed (unhandled exception escaped the harness)",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(row, indent=2))
        results.append(row)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
