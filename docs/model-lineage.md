# Model lineage — do not substitute one track for another

The single easiest way to publish a wrong result in this project is to
benchmark a **stock** Qwen3.8 GGUF and label it *uncensored*. They are
different models. Keep the tracks separate and label every result with
`model.lineage` in the result JSON.

## Track A — reference / performance ceiling (stock Qwen3.8)

What stock Qwen3.8 plus a matching DFlash draft can do. This is the honest
apples-to-apples target for upstream comparison.

* Target: [`unsloth/Qwen3.8-27B-GGUF`](https://hf.co/unsloth/Qwen3.8-27B-GGUF) — **verified present 2026-08-20**
  * `Qwen3.8-27B-UD-Q4_K_XL.gguf` — 17,559,178,144 B (~17.6 GB), confirmed
  * Q5/Q6 variants referenced in the spec: verify filenames before scripting them
  * Alternative used by DFlash upstream's own eval: [`ggml-org/Qwen3.8-27B-GGUF`](https://hf.co/ggml-org/Qwen3.8-27B-GGUF)
* Draft: [`incoai/Qwen3.8-27B-DFlash2-GGUF`](https://hf.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
  (mirror: [`z-lab/`](https://hf.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) — **verified present**
  * `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` 1.14 GB · `Q8_0` 2.06 GB · `BF16` 3.86 GB

`lineage: "stock"`

## Track B — this repository's actual target (uncensored / abliterated)

The existing vLLM baseline serves `orcarouter/Qwen3.8-27B-Uncensored-FP8`,
produced by abliteration of BF16 followed by block-FP8 quantization.

**Finding that simplifies the spec's plan:** the spec proposed a four-step
fallback chain (find BF16 abliterated source → reproduce abliteration →
quantize to GGUF → last resort, requantize from FP8). That chain is mostly
unnecessary — **the same publisher already ships a GGUF**:

* [`orcarouter/Qwen3.8-27B-Uncensored-GGUF`](https://hf.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF)
  — verified present 2026-08-20, same org as the FP8 checkpoint, tagged
  `abliterated` / `mtp` / `function-calling`, includes `mmproj` for vision.
  **Gated** — accept its terms with the account behind `$HF_TOKEN` first
  (same gating friction as the FP8 repo).

So Track B is a direct download, not a conversion project.

`lineage: "abliterated"`

### Other abliterated GGUFs, if a cross-check is wanted

Independent abliterations of the same base — useful for checking whether a
DFlash acceptance result is specific to OrcaRouter's abliteration or general
to abliterated Qwen3.8:

* [`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`](https://hf.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) — notably lists a BF16 source (`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16`)
* [`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`](https://hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)
* [`chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF`](https://hf.co/chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF)
  — explicitly derived from the FP8 checkpoint, ships both safetensors and GGUF,
  tagged `bf16`. If a BF16 intermediate of *this specific* abliteration is
  needed, start here.

## Requantization from FP8 — last resort only

If FP8 → GGUF requantization is ever used, label it `lineage:
"requantized-from-fp8"` and **measure the quality loss** rather than assuming
it is small. FP8 → Q4 is a lossy step applied on top of an already-quantized
checkpoint. Given Track B is directly available, this path should not be
needed; if you take it, say why in the result's `correctness.notes`.

## Rule

> A stock GGUF is never labelled uncensored, and an abliterated model's
> numbers are never presented as stock Qwen3.8 performance.

Cross-track comparison is legitimate and interesting — see `dflash.md` — but
only when both sides are labelled.
