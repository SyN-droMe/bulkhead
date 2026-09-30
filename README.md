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
- **Timeout enforcement**: each tool call runs on a daemon thread with a hard timeout,
  so one hung tool can't hang the whole run (or the process, see below).
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
        no  -> run on a daemon thread with a timeout
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
| silent_empty (returns `""`) | completed | correct: model reasoned around the empty result; the harness can't tell "empty but technically successful" from real success, and doesn't try to |
| exception (raises mid-call) | completed | **recovered**: caught, retried, reported, model adapted |
| loop_inducing (always says "retry this") | **max_steps** | **loop blocked twice, then gave up**: an honest failure, not a hidden one |
| plausible_wrong_answer (silently off by +1) | completed | **SILENTLY WRONG**: final answer `19.14...` instead of `18`, no error anywhere |

Full machine-readable output in `results.json`.

## What this actually shows

- **The harness's job is narrower than "always succeed," and it did that job.** For
  every failure that produces *any* signal (a timeout, an exception, garbage output
  the model notices, a repeated call), the model either recovered or the harness
  surfaced the failure honestly. Nothing hung, nothing crashed uncaught, nothing
  silently returned a blank result to the caller.
- **`loop_inducing` is a genuine, useful negative result.** The harness correctly
  stopped a pointless duplicate call twice, but a tool that keeps saying "retry this
  exact same thing" with zero new information is genuinely hard to escape within a
  fixed step budget, and the model ran out of steps. That's reported as `max_steps`,
  not disguised as a success. A smarter version would detect "no forward progress
  after N interventions" and escalate faster instead of just capping total steps.
- **`plausible_wrong_answer` is the one this harness cannot catch, by design**, and
  it's worth being direct about why: the retry/timeout/error-surfacing layer only
  reacts to signals like exceptions, timeouts, and repeated calls. A tool that
  returns successfully with a wrong number produces none of those signals. In this
  run the wrongness even compounded silently across two chained calculator calls
  (`126 -> 127`, then `127/7 -> 18.14... -> 19.14...`) without a single error anywhere
  in the trace. Catching this class of failure needs answer verification or
  cross-checking, which is a different (and harder) problem than reliable tool
  orchestration.

## A bug this project actually caught

While building the timeout wrapper, an early version used
`concurrent.futures.ThreadPoolExecutor` and passed all of `test_harness.py`'s
assertions, but the process itself never exited. The leaked thread from a timed-out
call (Python can't force-kill a thread) was non-daemon, so the interpreter waited on
it at shutdown, indefinitely. `test_harness.py` was rerun with a wall-clock check and
caught it directly; the fix was switching to a raw `daemon=True` thread per call. The
tests passing wasn't proof of correctness on its own: proof was watching the process
actually exit.

## Limitations

- **Daemon threads don't kill the work, they just stop waiting for it.** A genuinely
  hung tool call keeps running in the background for the life of the process. A
  production version would isolate tool execution in separate processes so a hung
  call can actually be killed, not just abandoned.
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
- Process-level tool isolation for a real hard-kill on timeout.
- A second model/provider run, to see how much of the recovery behavior is
  harness-driven versus model-driven.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # free tier at console.groq.com, no card required
python test_harness.py        # offline: verifies retry/timeout/error logic directly
python run_bench.py           # live: runs the 7-variant benchmark, writes results.json
```

## Files

```
harness.py       Core loop: retry, timeout, error surfacing, loop handling
tools.py         calculator, lookup: the two real tools
broken_tools.py  6 broken calculator variants, one per failure mode
run_bench.py     Runs the fixed task through each variant, classifies the outcome
test_harness.py  Offline tests against the retry/timeout layer (no API key needed)
```
