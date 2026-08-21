# NVFP4 on Blackwell — experimental track

**Status: UNTESTED. Experimental branch, never the default.**

## Why it is technically grounded

Blackwell SM120 has native FP4 tensor-core support, and current llama.cpp CUDA
sources contain native MXFP4/NVFP4 MMQ paths. So this is not wishful thinking —
but "the kernels exist" is a long way from "this checkpoint converts cleanly and
is faster and is still accurate".

Ecosystem checkpoints exist (verified present 2026-08-20):

* `unsloth/Qwen3.8-27B-NVFP4` — referenced as a base model by
  [`esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`](https://hf.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF),
  which is tagged `nvfp4` / `blackwell` / `mtp`.

Unsloth describes its Blackwell-oriented NVFP4 checkpoint as substantially
faster than conventional 4-bit GGUF **under suitable serving stacks** — note the
qualifier. Treat it as a separate vLLM/SGLang/llama.cpp research path, not a
silent replacement for the GGUF track.

## Precision policy to investigate

Mixed precision is likely to matter more than blanket NVFP4, given this model's
hybrid GDN architecture:

```
large linear matrices       -> NVFP4
some attention projections  -> NVFP4 / FP8
GDN-sensitive tensors       -> FP8 / BF16
norms                       -> BF16 / F32
embeddings / lm_head        -> higher precision
```

The GDN (Gated DeltaNet) linear-attention layers are the ones to watch: they
carry recurrent state, so precision loss there can compound along the sequence
in a way it does not in plain attention. Test quality **at depth**, not just on
short prompts — a 4-bit state error that is invisible at 2K may be obvious at
128K.

## Required checks before any recommendation

1. Does the conversion produce a loadable model at all?
2. Greedy output vs the Q4/Q5 baseline — where does it diverge?
3. Quality at 8K vs 128K vs 256K (per above).
4. Actual speedup on RTX 5090, measured, not inherited from a vendor claim.
5. Does DFlash2 still work with an NVFP4 target, and at what acceptance?
   (Quantization-dependent acceptance collapse is a reported failure mode —
   see `dflash.md`.)

## Gate

Per `methodology.md`, this is **Stage 6** — attempted only after a validated
single-GPU baseline, validated long context, and settled multi-GPU topology.
Doing it earlier means not being able to attribute any observed change.
