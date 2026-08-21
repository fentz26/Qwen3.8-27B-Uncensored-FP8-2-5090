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
