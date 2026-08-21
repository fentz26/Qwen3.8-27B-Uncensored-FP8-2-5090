# Benchmark harness

Engine-neutral: works against any OpenAI-compatible endpoint (llama-server,
vLLM, SGLang).

```sh
python3 run.py --url http://127.0.0.1:9000 \
  --workload workloads/python.json --profile 1x5090-fast \
  --warmup 1 --runs 3 --out ../results/
```

## What it measures

* **Metric A** single-stream decode (`--concurrency 1`)
* **Metric B** aggregate (`--concurrency N` or `--urls ...`) → sets `aggregate: true`
* **Metric C** TTFT / ITL / E2E — always, via streaming

Decode rate is measured first-token→last-token, excluding prefill. TTFT needs
streaming, so `stream=True` is not optional.

## Workloads

Chosen to span output entropy, because that is what DFlash acceptance actually
tracks:

| Workload | Expected acceptance |
|---|---|
| `json_tool_call` | highest — rigid schema |
| `html`, `python` | high — structured |
| `markdown`, `technical_prose` | medium |
| `creative` | low — expected worst case |
| `agent_trajectory` | big system prompt, short answer — stresses TTFT and prefix cache, not decode |

`long_context.json` is a placeholder: generate real depth with
`context/needle.py --depth`, since `-c` is capacity, not depth.

## Long context

```sh
python3 context/needle.py --url ... --sweep 8192,32768,131072,262144
python3 context/repo_retrieval.py --url ... --depth 131072
python3 ../scripts/validate-context.py --url ... --depths 8192,131072,262144
```

`validate-context.py` exits non-zero unless recall is perfect — that is the
gate for claiming a context length.

## GPU metrics and device selection

`nvidia-smi` reports **physical** indices and ignores `CUDA_VISIBLE_DEVICES`, so
GPU metrics are filtered to the devices actually participating in the run.
Summing every GPU on the host would over-report a 1-GPU profile on a multi-GPU
box.

Selection precedence: `--gpus` (or `BENCH_GPUS` / `GPUS` / `GPU` env) →
`CUDA_VISIBLE_DEVICES` → all host GPUs.

Both are recorded: `metrics.per_gpu` (per-device detail) and
`metrics.vram_mib` / `gpu_util_pct` / `power_w` (aggregated over the selection
alone). `metrics.gpu_selection` records the source, the physical indices, and
the host GPU count — so `host_gpu_count > len(indices)` visibly confirms that
unrelated GPUs were excluded.

Indices are physical, not CUDA-remapped: with `CUDA_VISIBLE_DEVICES=2,3` the
artifact records `[2, 3]`, not `[0, 1]`.

These read the **local** host's nvidia-smi. Benchmarking a remote endpoint makes
them describe the wrong machine; `gpu_selection.source` keeps that auditable.

## Compatibility assumption: `stream_options`

`run.py` sends `stream_options.include_usage` to get server-reported token
counts. OpenAI, vLLM and llama-server support it; servers that ignore unknown
fields degrade safely to counting stream chunks (which miscounts when a chunk
carries several tokens). A server that **rejects** the field fails with an
explicit message rather than silently falling back — wrong token metrics are
worse than a failed run.

## Tests

```sh
python3 -m unittest discover -s bench -p 'test_*.py'
```

`test_gpu_snapshot.py` covers device selection with synthetic `nvidia-smi`
output — no GPU needed.

## Provenance

`run.py` reads these env vars into the artifact. Set them or the result is not
reproducible:

```
ENGINE_COMMIT  DFLASH_PR  BUILD_FLAGS
MODEL_QUANT  MODEL_LINEAGE  MODEL_SHA256
DRAFT_MODEL_ID  DRAFT_QUANT  DRAFT_SHA256
CTX  KV_K  KV_V  SPLIT_MODE  TENSOR_SPLIT  SPEC_KIND  SPEC_N_MAX
```

Anything unmeasurable is written as `null`. Acceptance metrics currently come
from `parsers/llamacpp.py` (stderr scrape) because llama-server does not export
them — see `docs/dflash.md`.
