# Multi-GPU topologies

Central fact shaping every recommendation here: **Qwen3.8-27B at Q4 (~17.6 GB)
plus a 1.1 GB draft fits on one 32 GB RTX 5090.** Nothing forces model
parallelism. So multi-GPU is a *choice*, and it must beat one GPU on a measured
number to be recommended.

## Profile B/E — independent replicas (default for >1 GPU)

One complete model + draft per GPU. No NCCL, no TP, no inter-GPU sync,
therefore no interconnect sensitivity at all. Front with `router/router.py`.

Best for: agents, multi-user, subagent fan-out — anything with concurrent
independent requests.

**Reporting rule.** Two replicas each doing ~110 tok/s is
`aggregate: true, concurrency: 2, decode_tps: ~218`. It is **not** "one request
at 218 tok/s". `bench/run.py` sets this automatically and
`scripts/generate-report.py` refuses to put aggregate and single-stream in the
same table.

Routing matters: each replica has its own prefix cache, so a session that
bounces between replicas re-prefills its system prompt every hop. The router
defaults to sticky affinity keyed on session id, `user`, or a hash of the
system prompt.

## Profile C — layer split (single model across GPUs)

`--split-mode layer` — pipeline-style. Lower interconnect demand than tensor
split and **keeps quantized KV**, which matters a lot for long context.

Expected to be *questionable* for this model: since the weights already fit one
card, splitting can add pipeline overhead without improving decode. Measure
against Profile A; a loss is a publishable result.

**Split bias.** If the draft is pinned to GPU0 (`--spec-draft-device CUDA0`),
GPU0 also holds draft weights, draft KV, and feature-injection buffers — so an
even `-ts 1,1` target split is *not* an even runtime load. Sweep
`1,1` → `0.9,1.1` → `0.8,1.2`, biasing target layers toward GPU1.

## Profile D — tensor split (experimental)

`--split-mode tensor`. Splits weights and KV across GPUs with cross-GPU
reductions.

Constraints (verify at runtime, they change):

* Flash Attention **required**
* **Quantized KV unsupported** → f16/bf16/f32 only, which is expensive at long
  context and directly fights this project's long-context goal
* NCCL should be built in (`WITH_NCCL=1 ./build.sh`)
* auto-fit does not work in this mode
* architecture support must be confirmed at runtime

**Topology reality on consumer 5090s:** no NVLink, and CUDA P2P is frequently
unavailable. On the dual-5090 node this repo was originally built on, P2P
reported **unavailable**. Without P2P, cross-GPU reduction cost can dominate
and tensor split will likely lose to a single GPU.

Run `scripts/topology.sh` first. If it reports no P2P, benchmark this only to
document the negative result. **Do not force `GGML_CUDA_P2P` when the platform
reports it unsupported.**

## Profile E-alt — 2 x (2-GPU) on a 4-GPU box

Only justified when one card genuinely cannot hold the needed configuration:
Q6 weights, F16 KV, or max context at high-precision KV. Otherwise four
independent replicas should win on aggregate.

## Recommendation order

`scripts/topology.sh` prints this automatically:

| GPUs | Order |
|---|---|
| 1 | `1x5090-fast` → `-quality` → `-256k` |
| 2, no P2P | replicas → layer → (tensor only to document the negative) |
| 2, with P2P | replicas → layer → tensor |
| 4 | replicas → 2x2 only if a single card can't hold the config |

**Never** enable P2P or recommend tensor split from GPU *count* alone.

## Scaling caveat

Aggregate throughput scales with independent streams only until CPU, host
memory bandwidth, request scheduling, the router, or the client becomes the
bottleneck. Benchmark 1/2/4/8/16 concurrency and find that knee — do not assume
4 GPUs means 4x.
