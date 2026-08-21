# DFlash / DFlash2 — speculative decoding

## Blocking constraint: DFlash2 is not in llama.cpp master

Verified 2026-08-20:

* PR [ggml-org/llama.cpp#27342](https://github.com/ggml-org/llama.cpp/pull/27342)
  — *"spec : add DFlash2 support (local convolution + candidate selector)"*,
  branch `dflash2` — **state: OPEN, not merged.**

Consequences:

* `--spec-type draft-dflash` **does not exist on master.** Building master and
  running any DFlash profile here fails at startup (`_common.sh` checks for the
  flag and exits with a clear message rather than failing mid-benchmark).
* `engines/llamacpp/build.sh` fetches the PR branch by default.
* Every DFlash result **must** record the PR number and the exact build SHA.
  A future rebase of that branch can change performance.
* Re-check before each session:
  `gh pr view 27342 --repo ggml-org/llama.cpp --json state,mergedAt`
  If merged, switch `LLAMACPP_REF` to a pinned master SHA and note the change.

## What DFlash2 actually is

A **block-diffusion drafter**: it predicts a whole block of tokens in one
forward pass, keeps top candidates at every position, and a lightweight
selector traces one coherent path. Two-tap dynamic convolutions keep draft
quality from decaying toward the end of the block. It is **not** a generic
small LLM used as a draft model, and it consumes target-model hidden features.

Upstream states decoding is **lossless**: greedy output matches the target
exactly, and sampling preserves the distribution. That is a testable claim,
and this repo tests it (`correctness.greedy_match_vs_no_spec`).

**DFlash ≠ MTP.** MTP is the multi-token-prediction head baked into the
Qwen3.8 checkpoint itself (`mtp.layers.0.*`) — that is what the vLLM baseline
in `findings.md` used. DFlash2 is a separate draft model. Do not conflate them
in results; `runtime.speculation.type` distinguishes `mtp` from `dflash2`.

## Upstream's own numbers (theirs, not ours)

Target `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`, first eight GSM8K test examples,
Qwen3.8 recommended sampling (temp 1.0, top-p 0.95, top-k 20), xhigh reasoning,
max 2048 new tokens:

| Draft quant | Acceptance length |
|---|---|
| BF16 | 5.28 |
| Q8_0 | 5.13 |
| Q4_K_M | 5.39 |

Two things worth noting: acceptance is high (~5 tokens per verification step),
and the **cheapest draft (Q4_K_M) scored highest** — within noise, but it means
there is no obvious reason to pay for a BF16 draft. Start at Q4_K_M.

`--spec-draft-n-max 7` is the value upstream's own quick-start uses.

## Open question: acceptance on an abliterated target

The draft is trained against **stock** Qwen3.8. This repo's Track B target is
**abliterated**. Abliteration modifies the model's activations/weights, so
draft/target agreement could shift.

**This is untested and must not be guessed at.** The experiment:

1. Same DFlash2 draft, same quant, same workloads, same n-max.
2. Target = stock Qwen3.8 (Track A) → record acceptance.
3. Target = abliterated Qwen3.8 (Track B) → record acceptance.
4. Compare: accepted, drafted, acceptance rate, mean acceptance length,
   draft time, verification time, final tok/s, greedy agreement, and behaviour
   at depth.

Interpretation, decided in advance:

* Within a few points of stock → cross-lineage reuse works; document it (this
  would be genuinely useful upstream).
* Materially lower → a draft adapted to the abliterated target is needed.
  Say so plainly rather than shipping a degraded default.

## Sweeping n-max

`n-max` is **clamped by the draft's trained block size** — asking for more than
that is silently capped. More speculative tokens is not automatically faster:
deeper blocks cost more draft compute and more wasted work on rejection.

Sweep 2 / 4 / 7 / max-supported per workload. Expect the optimum to differ
by output entropy: high for HTML/JSON/code, lower for creative prose.

## Telemetry gap

llama-server's `/metrics` exposes prompt/generation throughput but **not**
draft acceptance counters on this PR branch. `bench/parsers/llamacpp.py`
scrapes stderr as a stopgap and reports **null, never 0**, when it finds
nothing — a missing measurement must never look like a measured zero.

If the gap persists after validation, the right fix is a metrics PR upstream
rather than a permanent private patch.

## Known upstream caveats to check before trusting a config

Reported elsewhere; **not** confirmed for llama.cpp + RTX 5090 here. They are
the reason the validation suite exists:

* quantization-dependent acceptance collapse on some backends
* Qwen3.8/DFlash2 greedy-divergence reports on MLX
* hybrid-model (GDN) cache/speculation interactions
* cold-start vs steady-state differences — always warm up, ≥3 measured runs
* strong sensitivity to block size
