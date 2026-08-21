# Methodology

## Three different "throughput" numbers — never blur them

| Metric | Meaning | How |
|---|---|---|
| **A — single-request decode** | one user, one sequence. Headline latency number. | `run.py --concurrency 1` |
| **B — aggregate** | summed across concurrent independent requests | `run.py --concurrency N`, sets `aggregate: true` |
| **C — end-to-end agent** | TTFT, prefill, ITL, completion latency, tool-call latency, cache reuse | always collected |

For agent workloads **C often matters more than A**. A config with lower decode
tok/s but much better prefix-cache reuse can win on real end-to-end latency.

Reporting aggregate as if single-request is the most common way to overstate a
result. `generate-report.py` keeps them in separate tables.

## Measuring TTFT and ITL

Requires streaming. `run.py` uses `stream=True` and timestamps the first
content delta. Non-streaming timing cannot separate prefill from decode.

Decode rate is measured **first token → last token**, excluding prefill.
Dividing total tokens by total wall time silently mixes prefill into decode and
understates it.

## llama-bench vs llama-server — report both, never mix

`llama-bench` measures model execution and **deliberately excludes tokenization
and sampling**. `llama-server` shows what a client actually experiences.

Report them separately: `llama-bench` for model-engine throughput and
context-depth sweeps (`-d`), `llama-server` + `run.py` for real API latency.

## The KV-block-size trap

**A prefix-cache test whose shared prefix is shorter than one KV block reports
zero hits even when caching works perfectly.**

This model's hybrid mamba/GDN layers force an unusually large block —
**784 tokens** on the vLLM stack ("attention block size padded to match mamba
page size"). A ~300-token test prefix produced a confident, completely wrong
"prefix caching is broken" conclusion during the original vLLM work.

Rules:
* shared test prefixes **≥ 3000 tokens**
* read cache counters from `/metrics`, never from the periodic throughput log
  line (it samples on an interval and misses short, low-volume test traffic)

## Warmup and repeats

Cold start ≠ steady state, especially with torch.compile / CUDA-graph capture
and JIT'd kernels. First-call numbers can be 30%+ off.

Minimum: 1 warmup + **≥3 measured runs**, report the median. `run.py` defaults
to this and records `warmup_runs` / `measured_runs`.

## Provenance is part of the result

A number without its build is not reproducible. Every artifact records: GPU
model/count, driver, CUDA, P2P status, **exact engine commit SHA** (never
"latest"), model + draft ids and SHA256, quant, lineage, context size *and*
depth, KV types, split mode, batch/ubatch, concurrency, and speculation config.

llama.cpp performance changes materially between revisions, and DFlash2
currently lives on a rebasable PR branch — an unpinned SHA makes a result
worthless later.

## Staged gating — do not run the Cartesian product

The full matrix is thousands of runs. Gate it (Section 22):

1. fastest 1-GPU configuration
2. validate 128K and 256K context on that winner
3. dual replicas vs layer split
4. tensor split **only** if topology justifies it
5. scale the winning single-GPU profile to 4 replicas
6. only then NVFP4 / custom kernels

Each stage's winner is the only input to the next.

## Acceptance criteria (Section 23)

A configuration may be marked **recommended** only with: no OOM across the
suite; clean startup; no corrupted generations; greedy sanity checks passing
where expected; working tool calls; passing long-context retrieval; acceptable
quantization quality; no DFlash-induced correctness failures; reproducible
across ≥3 warm runs; exact versions recorded.

**Do not optimize for tok/s alone.**
