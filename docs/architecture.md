# Repository architecture

## What this repository is

An **empirical inference-optimization lab** for Qwen3.8-27B on RTX 5090, across
1 / 2 / 4 GPUs and two engines (vLLM FP8 serving, llama.cpp GGUF + DFlash2).

It is not a "best settings" list. It is a harness plus a record of what was
actually measured — including negative results.

## Layout

```
docs/          reasoning, methodology, findings
engines/
  vllm/        PRESERVED validated FP8 baseline (the one measured configuration)
  llamacpp/    build + serve scripts for GGUF/DFlash2 profiles
profiles/      *.env — one per hardware/precision/topology combination
router/        session-affinity router for independent replicas
bench/         engine-neutral harness, workloads, long-context suites, parsers
results/       machine-readable benchmark artifacts (community-submittable)
scripts/       hardware detection, topology, context validation, reporting
logs/          historical evidence from the original vLLM run
```

## Two engine tracks, deliberately parallel

**vLLM** (`engines/vllm/`) — the production-style serving baseline. FP8 weights,
FP8 KV, prefix caching, MTP. Already validated; see `findings.md`. **Do not
rewrite this to chase llama.cpp numbers** — it is the higher-precision
reference point and the only measured configuration in the repo.

**llama.cpp** (`engines/llamacpp/`) — the experimental track: GGUF quants,
DFlash2 block-diffusion speculative decoding, quantized KV, flexible multi-GPU
topologies. Currently entirely UNTESTED.

They answer different questions. Keep both.

## Configuration flow

```
profiles/<name>.env  ──sourced by──>  engines/llamacpp/serve-*.sh  ──>  llama-server
        │                                                                    │
        └────────────────> env exported into bench/run.py ──> results/*.json ─┘
                                                                   │
                                              scripts/generate-report.py
                                                                   │
                                                        docs/findings-generated.md
```

Profiles are data, scripts are mechanism, results are evidence. A profile
change that is not accompanied by a result is a hypothesis, not a finding.

## Design rules

1. **Measured and untested never mix.** Schema has an explicit `status`;
   `findings.md` has two separate sections.
2. **Aggregate and single-request never mix.** Enforced in schema
   (`aggregate`) and in the report generator's table separation.
3. **Provenance travels with the number.** Engine SHA, model SHA256, lineage,
   topology, context depth.
4. **Missing measurements are null, never zero.** A zero is a claim.
5. **Negative results are results.** "Layer split lost to one GPU" is worth
   committing.
