# Qwen3.8-27B RTX 5090 Inference Optimization Lab

Reproducible inference-optimization research for **Qwen3.8-27B on RTX 5090**
across 1 / 2 / 4 GPUs, two engines (vLLM FP8, llama.cpp GGUF + DFlash2), long
context, and speculative decoding.

**This is an empirical project.** A configuration is not "faster" because it
should be. Every performance claim here is either backed by a committed
benchmark artifact or explicitly labelled `UNTESTED`.

## Status — read this first

| Track | State |
|---|---|
| **vLLM FP8, 2x RTX 5090** | ✅ **MEASURED** — 76.3 tok/s, full results in [`docs/findings.md`](docs/findings.md) |
| **llama.cpp + DFlash2, all profiles** | ⬜ **UNTESTED** — harness written, zero numbers |
| 1x / 4x GPU profiles | ⬜ **UNTESTED** |
| Long context 128K / 256K | ⬜ **UNTESTED** (capacity ≠ capability) |
| NVFP4 | ⬜ **UNTESTED** |

The hardware that produced the vLLM baseline was rented and has been released.
**No RTX 5090 is currently available to this project**, so everything on the
llama.cpp track is scaffolding awaiting hardware. `results/` is empty by design
rather than by omission.

## Research objectives

**Primary.** How far past **100 generated tok/s** can Qwen3.8-27B be pushed on
RTX 5090 while keeping useful long-context capacity, correctness, tool calling,
and reproducibility?

**Secondary.** What is the best scaling architecture for 1, 2, and 4 GPUs —
distinguishing *single-request* performance from *aggregate system* throughput?

### Three different throughput numbers, never blurred

| | Meaning |
|---|---|
| **Metric A** — single-request decode | one user, one sequence. The headline latency number. |
| **Metric B** — aggregate | summed across concurrent independent requests. Two replicas at 110 tok/s = 220 **aggregate**, *not* "one request at 220". |
| **Metric C** — end-to-end agent | TTFT, prefill, ITL, tool-call latency, cache reuse. Often matters more than A for agents. |

Schema and report generator enforce the separation.

## Quick start

```sh
# what hardware is this, and which profiles are worth running?
./scripts/topology.sh

# --- vLLM track (validated) ---
engines/vllm/serve.sh

# --- llama.cpp track (experimental, needs an unmerged PR) ---
engines/llamacpp/build.sh          # fetches llama.cpp PR #27342 (DFlash2)
TRACK=A engines/llamacpp/download.sh
engines/llamacpp/serve-single.sh 1x5090-fast 0 9000
python3 bench/run.py --url http://127.0.0.1:9000 \
  --workload bench/workloads/python.json --profile 1x5090-fast --out results/
```

## Profiles

| Profile | Topology | Purpose |
|---|---|---|
| `1x5090-fast` | 1 GPU | the >100 tok/s candidate |
| `1x5090-quality` | 1 GPU | Q5 + Q8 KV, quality-leaning |
| `1x5090-256k` | 1 GPU | maximum context |
| `2x5090-replicas` | 2 GPU | **aggregate** throughput — favoured for agents |
| `2x5090-layer` | 2 GPU | single model, pipeline split |
| `2x5090-tensor-experimental` | 2 GPU | needs P2P; likely poor on consumer cards |
| `4x5090-replicas` | 4 GPU | expected best aggregate |
| `4x5090-2x2` | 4 GPU | two 2-GPU replicas, only if one card can't hold the config |

## Things this repo learned the hard way

**DFlash2 is not in llama.cpp master.** It is [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342),
open and unmerged as of 2026-08-20. Build master and every DFlash profile fails.

**A stock GGUF is not an uncensored model.** Track A (stock Qwen3.8) and
Track B (abliterated) are separate, and results carry a `lineage` field.
Usefully, `orcarouter` already publishes a GGUF of the abliterated model, so
Track B needs no FP8→GGUF requantization. See [`docs/model-lineage.md`](docs/model-lineage.md).

**A short prefix silently breaks prefix-cache testing.** This model's KV block
size is 784 tokens; a ~300-token shared prefix reports zero cache hits even
when caching works perfectly. That artifact produced a confidently wrong
conclusion during the vLLM work. Test prefixes are now ≥3000 tokens.

**Capacity is not capability.** `-c 262144` starting successfully proves
nothing about 256K usability. Context claims require recall numbers from
`scripts/validate-context.py`.

## Layout

```
docs/       methodology, findings, per-topic reasoning
engines/    vllm/ (validated baseline) · llamacpp/ (experimental)
profiles/   *.env per hardware/precision/topology combination
router/     session-affinity router for replica profiles
bench/      engine-neutral harness, workloads, long-context suites, parsers
results/    machine-readable artifacts (community-submittable)
scripts/    hardware detection, topology, context validation, reporting
logs/       historical evidence from the original vLLM run
```

Start with [`docs/architecture.md`](docs/architecture.md), then
[`docs/methodology.md`](docs/methodology.md).

## Contributing

Benchmark results from real RTX 5090 hardware are the main thing this project
needs. Negative results ("layer split lost to one GPU on my box") are welcome
and useful.

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Raw JSON artifacts required —
screenshots may supplement but never replace measurements.

## License

Scripts and docs: MIT. Models carry their own licenses (Qwen3.8 and DFlash2 are
Apache-2.0; the abliterated checkpoints have their own terms and some are gated).
