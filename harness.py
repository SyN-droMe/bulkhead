"""Bulkhead: a small, reliable tool-calling harness for Groq's OpenAI-compatible API.

Owns the parts a demo agent usually skips: retrying a failing tool call,
enforcing a timeout so one hung tool can't hang the whole run, surfacing
tool failures to the model honestly instead of pretending nothing went
wrong, and detecting when the model is stuck calling the same tool with
the same arguments in a loop.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    fn: callable  # fn(**kwargs) -> str


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    arguments: dict
    attempts: int
    outcome: str  # "success" | "error" | "timeout"
    result: str


@dataclass
class RunResult:
    final_text: str | None
    steps: int
    status: str  # "completed" | "max_steps" | "loop_detected"
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class Harness:
    def __init__(
        self,
        client,
        model: str,
        tools: list[ToolSpec],
        max_steps: int = 6,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        tool_timeout: float = 3.0,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_by_name = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.tool_timeout = tool_timeout

    def _tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools_by_name.values()
        ]

    def _call_tool_once(self, tool: ToolSpec, arguments: dict) -> str:
        # Run on a daemon thread, not a ThreadPoolExecutor: a genuinely hung tool call
        # can't be force-killed either way, but a daemon thread at least doesn't keep the
        # whole process alive at interpreter shutdown once the harness has moved on. An
        # earlier version used ThreadPoolExecutor here and the leaked non-daemon thread
        # from a timed-out call kept the process running indefinitely after `run()`
        # returned -- caught by test_harness.py, not by inspection.
        result_box: queue.Queue = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                result_box.put(("ok", tool.fn(**arguments)))
            except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
                result_box.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.tool_timeout)
        if thread.is_alive():
            raise TimeoutError(f"tool '{tool.name}' did not return within {self.tool_timeout}s")

        status, payload = result_box.get()
        if status == "error":
            raise payload
        return payload

    def _call_tool_with_retry(self, tool: ToolSpec, arguments: dict) -> tuple[str, str, int]:
        """Returns (result_text, outcome, attempts_used)."""
        last_error = "unknown error"
        outcome = "error"
        for attempt in range(1, self.max_retries + 2):  # 1 initial try + max_retries retries
            try:
                result = self._call_tool_once(tool, arguments)
                return str(result), "success", attempt
            except TimeoutError as exc:
                last_error = str(exc)
                outcome = "timeout"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                outcome = "error"
            if attempt <= self.max_retries:
                time.sleep(self.retry_backoff_base * (2 ** (attempt - 1)))
        return (
            f"TOOL_FAILED after {self.max_retries + 1} attempts: {last_error}",
            outcome,
            self.max_retries + 1,
        )

    def run(self, task: str) -> RunResult:
        messages = [{"role": "user", "content": task}]
        tool_calls: list[ToolCallRecord] = []
        seen_calls: set[tuple] = set()

        for step in range(1, self.max_steps + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                tools=self._tool_schemas(),
                messages=messages,
            )
            message = response.choices[0].message

            assistant_msg = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if not message.tool_calls:
                return RunResult(
                    final_text=message.content or "", steps=step, status="completed", tool_calls=tool_calls
                )

            loop_hit = False
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                call_key = (name, json.dumps(arguments, sort_keys=True))
                if call_key in seen_calls:
                    loop_hit = True
                    break
                seen_calls.add(call_key)

                tool = self.tools_by_name.get(name)
                if tool is None:
                    result_text, outcome, attempts = f"unknown tool '{name}'", "error", 0
                else:
                    result_text, outcome, attempts = self._call_tool_with_retry(tool, arguments)

                tool_calls.append(
                    ToolCallRecord(
                        step=step,
                        tool_name=name,
                        arguments=arguments,
                        attempts=attempts,
                        outcome=outcome,
                        result=result_text,
                    )
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            if loop_hit:
                return RunResult(final_text=None, steps=step, status="loop_detected", tool_calls=tool_calls)

        return RunResult(final_text=None, steps=self.max_steps, status="max_steps", tool_calls=tool_calls)
