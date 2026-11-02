# Bulkhead

A small tool-calling harness for LLM agents, built to answer one question: when a tool
call breaks, does the system recover, fail honestly, or fail silently?

Most "build an agent" examples stop at the happy path: call the API, if it wants a
tool run it, feed the result back, repeat. That loop is maybe 20 lines and it's the
easy part. The actual engineering is in what happens when a tool hangs, throws,
returns garbage, or gets called in an unproductive loop. Bulkhead is that layer,
built and benchmarked against six ways a tool can fail.

The name comes from the [bulkhead pattern](https://en.wikipedia.org/wiki/Bulkhead_(partition))
in resilience engineering: isolate a failure so it doesn't sink the whole system.

## What it does

- **The loop**: send messages, parse any tool calls, execute them, feed results back,
  repeat until the model gives a final answer or a step cap is hit.
- **Retry with backoff**: a failing tool call gets up to 2 retries with exponential
  backoff before the harness gives up on it.
- **Timeout enforcement**: each tool call runs with a hard timeout, so one hung tool
  can't hang the whole run (or the process, see below). Two isolation modes:
  `isolation="thread"` (default) runs the call on a daemon thread; `isolation="process"`
  runs it in a child process that can be killed for real.
- **Honest error surfacing**: when a tool fails after retries, the model is told
  exactly what happened, in plain text, instead of the harness pretending nothing
  went wrong.
- **Loop handling**: if the model calls the same tool with the same arguments twice,
  the harness doesn't re-execute it (pointless for a broken tool, wasteful for a slow
  one) or silently abort the run. It tells the model the call was blocked and lets
  it keep trying.

## Architecture

```
run(task)
  |
  v
send messages + tool schemas to the model
  |
  v
model wants a tool? --no--> return final text, done
  |
 yes
  |
  v
for each requested tool call:
    seen before (same name + args)?
        yes -> feed back "BLOCKED_DUPLICATE_CALL", don't execute, continue
        no  -> run with a timeout
               thread mode:  daemon thread, abandoned on timeout
               process mode: child process, terminated/killed on timeout
               success -> feed back the result
               fail    -> retry with backoff, up to 2 times
                          still failing -> feed back "TOOL_FAILED: <reason>"
  |
  v
loop back to the model with all tool results
  (or return "max_steps" if the step cap is hit first)
```

## The two real tools

- `calculator(expression)`: evaluates basic arithmetic safely via Python's `ast`
  module (parses and walks the tree itself, no `eval()`).
- `lookup(query)`: looks up a small fact (e.g. "days in a week") from a fixed
  in-memory table. A stand-in for a real search/retrieval tool.

## Proving it: 6 broken-tool scenarios

`broken_tools.py` defines broken drop-in replacements for `calculator`, one per
failure mode. `run_bench.py` runs one fixed task
("calculate (23+19)*3, then convert that many days to weeks", correct answer: 18)
through the harness once per variant and records what actually happened.

| Variant | Status | Outcome |
|---|---|---|
| baseline (working calculator) | completed | correct, no failure |
| timeout (hangs forever) | completed | **recovered**: timed out, retried, adapted to the right answer |
| malformed_output (garbage string) | completed | **loop blocked, then recovered**: retried the same call once, harness blocked the repeat, model adapted anyway |
| silent_empty (returns `""`) | completed | **loop blocked, then recovered**: model retried the empty call, harness blocked the repeat, model answered anyway; the harness can't tell "empty but technically successful" from real success, and doesn't try to |
| exception (raises mid-call) | completed | **recovered**: caught, retried, reported, model adapted |
| loop_inducing (always says "retry this") | completed | **loop blocked, then recovered**: one duplicate blocked, model adapted |
| plausible_wrong_answer (silently off by +1) | completed | **SILENTLY WRONG**: final answer `18.14...` instead of `18`, no error anywhere |

Full machine-readable output in `results.json`. These are single runs with default
sampling, not averages: recovery behavior varies between runs, and the `loop_inducing`
row below is the clearest example of that.

## What this actually shows

- **The harness's job is narrower than "always succeed," and it did that job.** For
  every failure that produces *any* signal (a timeout, an exception, garbage output
  the model notices, a repeated call), the model either recovered or the harness
  surfaced the failure honestly. Nothing hung, nothing crashed uncaught, nothing
  silently returned a blank result to the caller.
- **`loop_inducing` is the most run-dependent variant, and that is the honest result.** In
  this run, blocking one duplicate call was enough and the model answered correctly. An
  earlier run of the same benchmark hit the step cap instead. The harness can guarantee
  that the pointless repeat is blocked and that a failure to finish is reported as
  `max_steps`, but it cannot guarantee the model finds its way out within a fixed step
  budget. A smarter version would detect "no forward progress after N interventions" and
  escalate faster instead of just capping total steps.
- **`plausible_wrong_answer` is the one this harness cannot catch, by design.** It's because the retry/timeout/error-surfacing layer only
  reacts to signals like exceptions, timeouts and repeated calls. A tool that
  returns successfully with a wrong number produces none of those signals. In this
  run `(23 + 19) * 3` came back as `127.0` instead of `126`, and the run finished with
  `18.14...` instead of `18`, without a single error anywhere in the trace. Catching
  this class of failure needs answer verification or cross-checking, which is a
  different (and harder) problem than reliable tool orchestration.

## A bug this project actually caught

While building the timeout wrapper, an early version used
`concurrent.futures.ThreadPoolExecutor` and passed all of `test_harness.py`'s
assertions, but the process itself never exited. The leaked thread from a timed-out
call (Python can't force-kill a thread) was non-daemon, so the interpreter waited on
it at shutdown, indefinitely. `test_harness.py` was rerun with a wall-clock check and
caught it directly; the fix was switching to a raw `daemon=True` thread per call. The
tests passing wasn't proof of correctness on its own: proof was watching the process
actually exit.

## Killing a hung tool for real

The thread-mode timeout above stops *waiting*; it does not stop the *work*. Python has
no way to cancel a running thread, so a timed-out call keeps burning CPU, holding locks,
or writing files until the process exits. `isolation="process"` closes that gap:

```python
Harness(client=client, model=model, tools=tools, tool_timeout=2.0, isolation="process")
```

Each call now runs in a `spawn`ed child process. On timeout the harness escalates
`terminate()` to `kill()` and reaps it, then reports a normal `TOOL_FAILED` timeout to
the model, so nothing above this layer changes.

`bench_isolation.py` proves the difference offline: a tool sleeps past a 1s deadline and
then writes a marker file. If the file appears, the work survived; if it doesn't, the
work was killed.

| Mode | Returned after | Worker alive at return | Marker at return | Marker after +2s |
|---|---|---|---|---|
| thread | 1.00s | n/a | no | **yes, the work finished late** |
| process | 1.02s | no | no | **no, the work was killed** |

Both modes returned control at the timeout and both surfaced the failure identically.
The difference is only what happened to the abandoned work.

That hard-kill is not free:

| Mode | Median call time, fast tool (`2+2`, 5 calls) |
|---|---|
| thread | 0.3ms |
| process | 150.6ms |

The process cost is `spawn` startup (measured on Windows; the shape is the same
elsewhere, only the constant changes). It also imposes two constraints: tool functions
must be module-level and picklable, and scripts using process mode need the standard
`if __name__ == "__main__":` guard. That is why thread mode stays the default: most
tools return in milliseconds, where a 150ms tax per call is real, and most "hangs" are
finite. Process mode is the right choice when a tool can block forever, holds a
resource you need back, or might crash the interpreter (a segfault in a child process is
reported as an error instead of killing the harness).

`spawn` has a third cost that is easy to miss: every child re-imports the entry module.
The first live process-mode run of `run_bench.py` imported the Groq SDK at module level,
so each tool call paid that import in the child, startup crossed the 3s tool timeout,
and calls reported false timeouts (they succeeded on retry). Moving the import inside
`main()` fixed it. Keep entry-point imports light, and leave timeout headroom above the
spawn cost.

With that fixed, the full 7-variant benchmark ran end to end in process mode and matched
thread mode where it matters: every signal-producing failure surfaced the same way and
recovered, and the `timeout` variant returned at the same point. Differences in
individual runs came from model sampling, not from the isolation layer.

## Limitations

- **The default thread mode doesn't kill the work, it just stops waiting for it.** A
  genuinely hung tool call keeps running in the background for the life of the process.
  `isolation="process"` fixes this at the cost of a child-process spawn per call (~150ms
  here) and a picklability requirement on tool functions. The default stays thread mode
  because that cost applies to every call, not just the hung ones.
- **Process mode also isolates state.** Each call gets a fresh child process, so a tool
  cannot keep in-memory state between calls (caches, connection pools, counters). That
  isolation is the point, but it means process mode suits stateless tools best.
- **Process mode pays `spawn` cost on every call, including re-importing your entry
  module.** A heavyweight top-level import in the script running the harness is paid by
  every child; in this repo's first process-mode benchmark that alone pushed child
  startup past the tool timeout. Keep entry scripts light and leave headroom.
- **Loop detection is one heuristic**: exact (tool name, arguments) repeats. A tool
  that varies its unproductive response slightly each time would not be caught by
  this check.
- **The silently-wrong-answer case is out of scope for this layer**, as shown above:
  documented here rather than papered over.
- **One provider, one model** (Groq, `openai/gpt-oss-120b`). Retry/timeout/loop
  behavior is provider-agnostic by construction, but how *often* the model recovers
  from a given failure is not: a different model could do better or worse on the
  same broken tools.

## Future Improvements

- Escalate faster on "no forward progress" instead of relying only on the step cap.
- A second model/provider run, to see how much of the recovery behavior is
  harness-driven versus model-driven.
- Reuse a worker process across calls (or a small pool) to amortize the spawn cost of
  process isolation.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # free tier at console.groq.com, no card required
python test_harness.py        # offline: verifies retry/timeout/error logic directly
python bench_isolation.py     # offline: thread vs process timeout behavior, no API key
python run_bench.py           # live: runs the 7-variant benchmark, writes results.json
python run_bench.py --isolation process   # same benchmark, hard-killable tool execution
```

## Files

```
harness.py           Core loop: retry, timeout, error surfacing, loop handling
tools.py             calculator, lookup: the two real tools
broken_tools.py      6 broken calculator variants, one per failure mode
run_bench.py         Runs the fixed task through each variant, classifies the outcome
bench_isolation.py   Offline thread-vs-process comparison on a hung tool
test_harness.py      Offline tests against the retry/timeout/isolation layer
```
