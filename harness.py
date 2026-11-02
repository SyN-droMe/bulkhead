"""Bulkhead: a small, reliable tool-calling harness for Groq's OpenAI-compatible API.

Owns the parts a demo agent usually skips: retrying a failing tool call,
enforcing a timeout so one hung tool can't hang the whole run, surfacing
tool failures to the model honestly instead of pretending nothing went
wrong, and detecting when the model is stuck calling the same tool with
the same arguments in a loop.

Tool execution has two modes:

- ``isolation="thread"`` (default): a daemon thread per call. Cheap, but a
  timed-out call cannot be killed -- Python has no thread cancellation, so the
  work keeps running in the background until the process exits.
- ``isolation="process"``: a child process per call. Costs a spawn (tens to hundreds
  of ms, and the child re-imports your entry module) and requires the tool function to
  be importable/picklable, but a timed-out call can be terminated for real, and a
  crashing tool cannot take the harness down with it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field


def _process_worker(fn, arguments: dict, send_conn) -> None:
    """Child-process entry point. Sends exactly one ("ok"|"error", payload) message."""
    try:
        send_conn.send(("ok", fn(**arguments)))
    except BaseException as exc:  # noqa: BLE001 - serialize every failure back to the parent
        send_conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        send_conn.close()


class ToolError(RuntimeError):
    """A tool call failed inside the isolation layer, with the message already formatted.

    Exists so a child process's `"RuntimeError: boom"` string isn't re-prefixed into
    `"ToolError: RuntimeError: boom"` by the retry loop.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.formatted = message


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
    outcome: str  # "success" | "error" | "timeout" | "loop_blocked"
    result: str


@dataclass
class RunResult:
    final_text: str | None
    steps: int
    status: str  # "completed" | "max_steps"
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
        isolation: str = "thread",
    ) -> None:
        if isolation not in ("thread", "process"):
            raise ValueError(f"isolation must be 'thread' or 'process', got {isolation!r}")
        self.client = client
        self.model = model
        self.tools_by_name = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.tool_timeout = tool_timeout
        self.isolation = isolation
        # PID of the most recent child process, or None in thread mode. Exposed so tests
        # and callers can verify the worker really died rather than being abandoned.
        self.last_process_pid: int | None = None
        self._mp_context = mp.get_context("spawn") if isolation == "process" else None

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
        if self.isolation == "process":
            return self._call_tool_in_process(tool, arguments)
        return self._call_tool_in_thread(tool, arguments)

    def _call_tool_in_thread(self, tool: ToolSpec, arguments: dict) -> str:
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

    def _call_tool_in_process(self, tool: ToolSpec, arguments: dict) -> str:
        """Run the tool in a child process so a timeout can actually kill the work.

        Spawn (not fork) keeps this portable to Windows and avoids inheriting the
        parent's threads and locks. The tradeoff is that `tool.fn` must be a
        module-level, picklable function, and each call pays a spawn cost.
        """
        recv_conn, send_conn = self._mp_context.Pipe(duplex=False)
        process = self._mp_context.Process(
            target=_process_worker, args=(tool.fn, arguments, send_conn), daemon=True
        )
        process.start()
        send_conn.close()  # parent only reads; holding this open would delay EOF
        self.last_process_pid = process.pid

        # poll() does the waiting without a busy loop: returns True as soon as the child
        # writes its payload or closes the pipe (including by dying without writing).
        if not recv_conn.poll(self.tool_timeout):
            self._terminate(process)
            recv_conn.close()
            raise TimeoutError(f"tool '{tool.name}' did not return within {self.tool_timeout}s")

        try:
            status, payload = recv_conn.recv()
        except (EOFError, OSError):
            # Child died before sending anything (segfault, os._exit, killed externally).
            process.join(0.5)
            recv_conn.close()
            raise ToolError(
                f"tool '{tool.name}' worker exited with code {process.exitcode} without returning a result"
            )
        finally:
            recv_conn.close()

        # The worker sends its result before exiting; give it a moment to finish, then
        # make sure a child that lingers after replying doesn't outlive the call.
        process.join(0.5)
        if process.is_alive():
            self._terminate(process)
        if status == "error":
            raise ToolError(payload)
        return payload

    @staticmethod
    def _terminate(process) -> None:
        """Escalate SIGTERM -> SIGKILL so a tool that ignores soft termination still dies."""
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)

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
                last_error = getattr(exc, "formatted", f"{type(exc).__name__}: {exc}")
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

            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                call_key = (name, json.dumps(arguments, sort_keys=True))
                if call_key in seen_calls:
                    # Don't silently re-run it (wasteful for a hung tool, pointless for a
                    # broken one) and don't just abort the run with no answer either -- tell
                    # the model plainly what happened and let it adapt, same honesty
                    # principle as every other failure path here.
                    blocked_msg = (
                        f"BLOCKED_DUPLICATE_CALL: '{name}' was already called with these exact "
                        "arguments earlier in this run and was not re-executed. Use the result "
                        "you already received, try different arguments, or explain that you "
                        "cannot complete the task."
                    )
                    tool_calls.append(
                        ToolCallRecord(
                            step=step,
                            tool_name=name,
                            arguments=arguments,
                            attempts=0,
                            outcome="loop_blocked",
                            result=blocked_msg,
                        )
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": blocked_msg})
                    continue
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

        return RunResult(final_text=None, steps=self.max_steps, status="max_steps", tool_calls=tool_calls)
