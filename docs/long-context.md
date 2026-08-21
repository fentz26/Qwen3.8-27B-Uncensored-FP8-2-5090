# Long context

## The core rule

> The server starting with `-c 262144` is **capacity**, not capability.
> A context claim requires a **recall number**.

`scripts/validate-context.py` is the gate. It exits non-zero unless needle
recall is perfect at the claimed depth.

## Depths

Benchmark at **8K / 32K / 64K / 128K / 192K / 256K**. Record actual
`usage.prompt_tokens`, not the `-c` value — `runtime.context_depth` in the
result schema means real tokens in the prompt.

Per depth collect: prompt tok/s, decode tok/s, TTFT, VRAM, GPU power,
DFlash acceptance, stability, and output correctness.

Acceptance often degrades with depth. A DFlash configuration that is excellent
at 8K and useless at 128K is a finding worth publishing, not a config to ship.

## Suites

**Needle** (`bench/context/needle.py`) — unique facts at 5% / 25% / 50% / 75% /
95% depth, retrieved individually. Position matters: mid-context recall is
typically the weakest and is exactly what a single end-of-context probe misses.

**Long-code retrieval** (`bench/context/repo_retrieval.py`) — questions
requiring retrieval from distant parts of a large source tree. Closer to real
agent use than synthetic needles.

**Tool-schema / agent turns** — large system prompt plus realistic tool
definitions, then repeated turns. This is where prefix-cache reuse dominates
end-to-end latency; see `workloads/agent_trajectory.json`.

## KV precision vs context

Compare **Q4 / Q8 / F16 KV at identical depth**, on both memory *and* quality.
The tradeoff is the whole point: Q4 KV buys depth and costs fidelity, and the
cost is not uniform across depth.

Note the interaction with `multi-gpu.md`: tensor split forbids quantized KV
entirely, which is a real argument against it for this workload.

The goal is not "fits in VRAM". It is:

> fits **and** remains accurate **and** remains stable **and** stays fast enough.

## Reporting

A profile may claim a context length only when, at that depth: no OOM, needle
recall 1.0, no corrupted output, tool calling still works, and latency is
recorded. `validate-context.py --out` writes the artifact proving it.
