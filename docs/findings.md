# Findings

The empirical record. Two tiers, never mixed:

* **MEASURED** — reproduced on real hardware, with the config that produced it.
* **UNTESTED** — not yet run. Hypotheses live here, clearly labelled.

A configuration is not "faster" because it should be. If it has no number, it says UNTESTED.

---

## MEASURED — vLLM on 2x RTX 5090

The original validated baseline. Preserved in `engines/vllm/`.

Setup: `orcarouter/Qwen3.8-27B-Uncensored-FP8`, vLLM 0.27.1, TP=2,
`--max-model-len 131072`, 2x RTX 5090 32GB.
Method: single request, `temperature=0`, 220 completion tokens, same prompt, steady state.

| Config | Decode | Note |
|---|---|---|
| Baseline (bf16 KV, no prefix cache, no MTP) | **56.7 tok/s** | starting point |
| + prefix caching + xxhash (bf16 KV) | ~56.7 tok/s | no decode change; wins TTFT on repeat prefixes |
| + `--kv-cache-dtype fp8` | **76.3 tok/s** | +34%; decode is memory-bandwidth bound |
| + `qwen3_5_mtp` speculative decoding (bf16 KV, no fp8) | **88–91 tok/s** | +56–60%; 73.8% draft acceptance, mean acceptance length 1.74 |
| MTP + fp8 KV **together** | **49.3 tok/s** | regression vs either alone |

KV cache budget at `--max-model-len 131072`: 336,719 tokens (bf16) →
**634,971 tokens** (fp8), i.e. 4.84x concurrency headroom.

### On the MTP + FP8 regression — scope this claim correctly

This is **a specific observed regression on that exact stack**
(vLLM 0.27.1 / 2x RTX 5090 / TP=2 / that checkpoint), **not** a universal
incompatibility between MTP and FP8 KV. The OrcaRouter model card reports
successful FP8-KV + MTP serving elsewhere. Plausible causes not yet
separated: vLLM version, TP topology, scheduler interaction, kernel
selection, hardware. Root cause was **not** diagnosed — the rented hardware
was released first.

Anyone reproducing this: bisect vLLM version first, then TP=1 vs TP=2.

### Correctness observations (same stack)

* MTP produced **byte-identical greedy output** vs no-MTP on the tested prompt.
* fp8 KV produced **byte-identical greedy output** vs bf16 KV on the tested prompt.
* Prefix caching verified hitting via `vllm:prefix_cache_hits_total`, ~40–45% on a
  3,486-token shared prefix.

### Methodology correction worth keeping

An earlier pass concluded "MTP breaks prefix caching" from a **~300-token**
shared prefix. That model's KV block size is **784 tokens**, so no prefix
shorter than one block can ever register a hit. The conclusion was a
measurement artifact. Re-tested at 3,486 tokens, prefix caching worked under
every config tried — including MTP, at a reduced hit rate.

Lesson now encoded in `bench/` and `docs/methodology.md`: **a short test
prefix silently produces a false negative.**

---

## UNTESTED — everything below

No llama.cpp, DFlash, 1-GPU, 4-GPU, NVFP4, or long-context number in this
repository has been measured. The hardware used for the vLLM baseline was
released and no RTX 5090 is currently available to this project.

| Question | Status |
|---|---|
| Can 1x RTX 5090 exceed 100 tok/s single-request on structured output? | UNTESTED — primary objective |
| DFlash2 acceptance on **stock** Qwen3.8 target, RTX 5090 | UNTESTED (upstream reports acceptance length 5.13–5.39 on GSM8K, Q4_K_M draft, `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` target — *their* measurement, not ours) |
| DFlash2 acceptance on **abliterated** target | UNTESTED — the key cross-lineage question, see `dflash.md` |
| 2x replicas aggregate throughput | UNTESTED |
| 2x layer split vs 1 GPU | UNTESTED — may well lose; the model fits one card |
| 2x tensor split | UNTESTED — likely poor without P2P, see `multi-gpu.md` |
| 4x replicas scaling / knee | UNTESTED |
| Usable context at 128K / 256K (recall, not capacity) | UNTESTED |
| Q4 vs Q5 vs Q6 quality/speed tradeoff | UNTESTED |
| Q4 vs Q8 vs F16 KV quality at depth | UNTESTED |
| NVFP4 on Blackwell | UNTESTED |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` effect | UNTESTED |
| Adaptive DFlash block sizing | UNTESTED — do not implement before static baselines exist |

### Hypotheses (explicitly not results)

1. **>100 tok/s on 1 GPU is plausible but unproven.** A reported Qwen3.8-27B +
   DFlash2 + llama.cpp configuration reaches near 100 tok/s on RTX PRO 4500
   (896 GB/s). RTX 5090 has ~1.792 TB/s — roughly 2x bandwidth. **Do not
   extrapolate 2x throughput.** Decode is only partly bandwidth-bound;
   verification compute, draft compute, acceptance length, GDN kernels,
   sampling, and launch overhead all cap the gain.
2. **Independent replicas beat tensor parallelism on consumer 5090s** for
   aggregate/agent workloads, because the 27B Q4 target fits one 32GB card and
   consumer cards lack NVLink.
3. **Layer split may be strictly worse than 1 GPU** for this model size.

Each needs a number in `results/` before it graduates out of this section.
