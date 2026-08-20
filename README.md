# Qwen3.8-27B-Uncensored-FP8 on 2x RTX 5090

Config and setup notes for serving [`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
(Qwen3.5-family, FP8, hybrid linear-attention, 256K native context) via vLLM
on **2x RTX 5090 (32GB each)**, wired into
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
as a custom model provider.

No secrets are stored in this repo — see "Credentials" below.

## Hardware / software

- 2x RTX 5090 (32GB each)
- torch 2.13.0+cu130
- vLLM 0.27.1
- Model: FP8 checkpoint, ~29GB on disk, 256K max context (served at 131072)

## Setup order

1. `scripts/download-model.sh` — downloads the model. The repo is gated;
   the account behind `HF_TOKEN` must accept its terms on the HF model page
   first. If `huggingface.co` isn't reachable from your network, set
   `HF_ENDPOINT=https://hf-mirror.com` before running it — a common mirror
   fallback.
2. `scripts/serve.sh` — launches vLLM, bound to `0.0.0.0:8000` (so it's
   reachable at `127.0.0.1:8000` locally by default).
3. (optional) `supervisor/vllm-qwen.service` — runs it as a proper systemd
   service instead of a bare background process, so it survives crashes and
   logs go through `journalctl`. See `supervisor/README.md`.
4. `dsh/settings.snippet.yaml` — the `dsh` custom-provider config to point
   the harness at the running server.

## vLLM flags, and why

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | Split across both GPUs. The GDN linear-attention layer and MLA/DSA-style layers both shard their heads by `tp_size`, so TP=2 is fully supported for this model family. |
| `--max-model-len` | `131072` | Model natively supports up to 256K. 131072 was chosen to leave real concurrency headroom rather than maxing out at 1x; see benchmarks below for the actual final KV budget. |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_xml` | | Required for any tool-calling client (`dsh` always sends tool defs). Without this vLLM 400s every request with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. `qwen3_xml` is vLLM's parser for this Qwen3.x family. |
| `--enable-prefix-caching` + `--prefix-caching-hash-algo xxhash` | | Off by default for this model. `dsh` resends the same system prompt + tool schema every turn; this reuses cached KV state across turns instead of recomputing it. `xxhash` (needs `pip install xxhash` in the vLLM venv) is faster block-hashing than the sha256 default. Confirmed real hits via `curl :8000/metrics | grep prefix_cache` — see "Benchmarking gotcha" below before trusting any prefix-cache test you write yourself. |
| `--kv-cache-dtype fp8` | | Halves per-token KV cache memory. On this hardware it nearly doubled the KV budget (302K → 634,971 tokens) *and* measurably sped up decode (56.7 → 76.3 tok/s) since decode is memory-bandwidth bound. Confirmed correct (byte-identical greedy output vs bf16) and confirmed compatible with prefix caching. |
| `--gpu-memory-utilization` | `0.96` | Default (0.92) left ~3GB/GPU idle in practice; bumping it grew the KV cache pool further. |
| `--max-num-batched-tokens` | `8192` | Default is 2048 — small chunks force a long prompt (like dsh's ~10K-token system prompt) through many scheduler steps just to prefill. 8192 matches better with the large context/KV budget here. |

### MTP (multi-token prediction) — investigated, not used in the final config

This checkpoint genuinely ships MTP weights (`mtp.layers.0.*`, 1606 tensors
total incl. MTP — check `model.safetensors.index.json`), and vLLM 0.27.1 has
a matching `--spec-method qwen3_5_mtp`. It works: `--speculative-config
'{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'` gives **+56-60%
decode throughput** (56.7 → 88-91 tok/s) with byte-identical output under
greedy decoding (73.8% draft-token acceptance rate, mean acceptance length
1.74).

**It's not combined with `--kv-cache-dtype fp8` here** — that specific pair
regresses to 49 tok/s, worse than either flag alone. Root cause not
diagnosed (ran out of time on the target hardware before a deeper
investigation); if revisiting, bisect there first. MTP alone (bf16 KV
cache) is a legitimate choice if raw decode speed matters more than the
fp8 KV budget/speed bump — see `logs/vllm_serve_final.log` and the git
history of this file for exact numbers from that config.

### Benchmarking gotcha: KV cache block size

This model's hybrid mamba/attention layers force an unusually large KV
cache block size (784 tokens — "attention block size padded to match mamba
page size" in the boot log). **A prefix-cache test with a shared prefix
under ~784 tokens will always show zero hits, regardless of whether caching
actually works.** An early pass here concluded MTP "breaks" prefix caching
based on exactly this mistake (a ~300-token test prefix); a corrected test
(3486 tokens, several full blocks) showed prefix caching working under
every config tested (bf16, fp8, xxhash, and even MTP — just at a lower hit
rate under MTP). Use `curl :8000/metrics | grep prefix_cache_hits_total`
for ground truth, not the periodic log line (`Prefix cache hit rate` in the
10s-interval throughput log resets/misses fast, low-volume test traffic).
`scripts/bench.sh` does this correctly.

### Numbers, for reference

Single-request, temperature=0, 220 completion tokens, same prompt each time:

| Config | Decode speed |
|---|---|
| Baseline (bf16, no prefix cache, no MTP) | 56.7 tok/s |
| + prefix caching + xxhash (bf16) | ~56.7 tok/s (unaffected; wins on TTFT for repeat prefixes, not raw decode) |
| + `kv-cache-dtype fp8` (this repo's final config) | **76.3 tok/s** |
| + `qwen3_5_mtp` speculative decoding instead of fp8 | 88-91 tok/s |
| MTP + fp8 combined | 49.3 tok/s (regression — don't combine) |

KV cache budget (`GPU KV cache size` in the boot log) at `--max-model-len
131072`: 336,719 tokens without fp8 → **634,971 tokens with fp8** (4.84x
concurrency headroom at full context length).

## `dsh` integration

`dsh/settings.snippet.yaml` is the provider block to add under
`llm-pi-ai.providers` in `$DSH_HOME/settings.yaml`. It does not make the
model the default — it just makes it selectable. The model needed explicit
`contextWindow`/`maxTokens` set because, absent a catalog entry, `dsh`
defaulted the output cap (`max_tokens`) to the full context window, which
left no room for the input prompt and every request 400'd with
`CONTEXT_WINDOW_EXCEEDED`.

By default it points at `http://127.0.0.1:8000/v1` (matches `scripts/serve.sh`
run on the same host). For a remote deployment, point `baseURL` at your own
`http://$HOST:$PORT/v1` and put whatever reverse proxy/tunnel/auth you use in
front — that part is entirely up to your environment, nothing here assumes
a specific one.

## Credentials

Nothing here is a secret by itself, but two things this setup can depend on
are, and neither is committed:

- **`HF_TOKEN`** — needed to accept the gated model's terms; never hardcode
  it, `download-model.sh` reads it from the environment.
- **`QWEN_API_KEY`** (optional) — only relevant if you put an authenticated
  proxy in front of vLLM for remote access. Reference it by env-var name in
  `settings.snippet.yaml`'s `apiKeyEnv`, store the actual value in
  `$DSH_HOME/.credentials.yaml`. Not needed for a plain local deployment.

If either of these tokens ever ends up in a shared transcript, chat log, or
anywhere outside `.credentials.yaml`, rotate it.

## Logs

`logs/vllm_serve_final.log` and `logs/model_download.log` are boot/serve and
model-download logs from a validated run on the target 2x RTX 5090
configuration, kept for reference (KV cache sizing, compile timings, route
list, etc.).
