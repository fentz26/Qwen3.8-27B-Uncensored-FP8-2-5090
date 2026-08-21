# vLLM — validated FP8 serving baseline

**This is the only measured configuration in the repository. Do not rewrite it
to chase llama.cpp numbers.** It is the higher-precision, production-style
reference point, and the exact config that produced the numbers in
`docs/findings.md`.

## Files

| File | What |
|---|---|
| `serve.sh` | the exact validated launch command |
| `download-model.sh` | gated-model download |
| `systemd/` | run it as a managed service instead of a bare background process |
| `dsh-settings.snippet.yaml` | deepseek-harness provider config |
| `bench-legacy.sh` | the original vLLM-specific bench; superseded by `bench/run.py` but kept — it produced the recorded results |

## Configuration

`orcarouter/Qwen3.8-27B-Uncensored-FP8` · vLLM 0.27.1 · 2x RTX 5090 · TP=2 ·
131072 context · FP8 KV cache · prefix caching + xxhash · `gpu-memory-utilization
0.96` · `max-num-batched-tokens 8192` · Qwen tool calling via `qwen3_xml`.

Measured: **76.3 tok/s** single-request decode; KV budget 634,971 tokens
(4.84x concurrency headroom at full context). Full table and caveats:
`docs/findings.md`.

## Flags that are load-bearing

* `--enable-auto-tool-choice --tool-call-parser qwen3_xml` — without these
  vLLM 400s **every** request from a tool-calling client.
* `--kv-cache-dtype fp8` — +34% decode and ~2x KV budget here.
* `--enable-prefix-caching` — no decode change, large TTFT win on repeated
  prefixes (agent turns).
* `--max-model-len 131072` — model supports 256K; this trades ceiling for
  concurrency headroom.

## MTP

The checkpoint ships real MTP weights (`mtp.layers.0.*`). `--speculative-config
'{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'` measured **88–91
tok/s** (+56–60%) with 73.8% acceptance — but regressed to 49.3 tok/s when
combined with fp8 KV **on this stack**. See `docs/findings.md` for why that is
a scoped observation and not a universal rule.

MTP here is the checkpoint's own head — not DFlash2, which is a separate draft
model on the llama.cpp track. Don't conflate them in results.
