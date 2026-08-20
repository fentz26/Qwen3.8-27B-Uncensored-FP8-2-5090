# Qwen3.8-27B-Uncensored-FP8 on 2x RTX 5090

Config and setup notes for serving [`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
(Qwen3.5-family, FP8, hybrid linear-attention, 256K native context) via vLLM
on a rented dual-RTX-5090 the target hardware instance, wired into
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
as a custom model provider.

No secrets are stored in this repo — see "Credentials" below.

## Hardware / software

- 2x RTX 5090 (32GB each), the target hardware deployment
- torch 2.13.0+cu130
- vLLM 0.27.1
- Model: FP8 checkpoint, ~29GB on disk, 256K max context (served at 131072)

## Setup order

1. `scripts/download-model.sh` — downloads the model via `hf-mirror.com`
   (huggingface.co was blocked on this instance's network route). The repo
   is gated; the account behind `HF_TOKEN` must accept its terms on the HF
   model page first.
2. `scripts/serve.sh` — launches vLLM.
3. `scripts/expose-port.sh` — wires it behind the instance's authed a reverse proxy
   edge so it's reachable from outside the container.
4. `dsh/settings.snippet.yaml` — the `dsh` custom-provider config to point
   the harness at the running server.

## vLLM flags, and why

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | Split across both GPUs. The GDN linear-attention layer and MLA/DSA-style layers both shard their heads by `tp_size`, so TP=2 is fully supported for this model family. |
| `--max-model-len` | `131072` | KV cache budget on this box is ~300K+ tokens total; 131072 leaves ~2.5x concurrency headroom while covering long agentic sessions. Model natively supports up to 256K if you want to push further (leaves ~1x concurrency). |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_xml` | | Required for any tool-calling client (`dsh` always sends tool defs). Without this vLLM 400s every request with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. `qwen3_xml` is vLLM's parser for this Qwen3.x family. |
| `--enable-prefix-caching` | | Off by default for this model. `dsh` resends the same system prompt + tool schema every turn, so this reuses cached KV state across turns instead of recomputing it — cuts time-to-first-token on every follow-up message. Confirmed hitting (`Prefix cache hit rate` > 0 in logs) after enabling. |
| `--gpu-memory-utilization` | `0.96` | Default (0.92) left ~3GB/GPU idle in practice; bumping it grew the KV cache pool from ~302K to ~337K tokens. |

## Exposing it externally

the target hardware external ports are fixed at instance creation — vLLM's own port
(8000) was never one of them, so it's only reachable inside the container by
default. `scripts/expose-port.sh` adds a `portal.yaml` entry that puts it
behind the instance's a reverse proxy edge on the first free "normal" port
(`container_port=10100` → `public_port=8000` on this particular deployment —
**check `platform-capabilities | jq '.instance.open_ports'` on a fresh deployment,
these are not stable across instances**).

Requests need the instance's portal token:
```
curl -H "Authorization: Bearer $QWEN_API_KEY" http://$HOST:$PORT/v1/models
```

## `dsh` integration

`dsh/settings.snippet.yaml` is the provider block to add under
`llm-pi-ai.providers` in `$DSH_HOME/settings.yaml`. It does not make the
model the default — it just makes it selectable. The model needed explicit
`contextWindow`/`maxTokens` set because, absent a catalog entry, `dsh`
defaulted the output cap (`max_tokens`) to the full context window, which
left no room for the input prompt and every request 400'd with
`CONTEXT_WINDOW_EXCEEDED`.

## Credentials

Nothing here is a secret by itself, but two things this setup depends on
are, and neither is committed:

- **the target hardware instance's `$QWEN_API_KEY`** (the a reverse proxy edge auth token) —
  goes in `$DSH_HOME/.credentials.yaml` as `QWEN_API_KEY`, referenced
  by env-var name only in `settings.snippet.yaml`.
- **HF token** used to accept the gated model's terms — never hardcode it;
  `download-model.sh` reads it from `$HF_TOKEN`.

If either of these tokens ever ends up in a shared transcript, chat log, or
anywhere outside `.credentials.yaml`, rotate it.

## Current endpoint (this deployment)

- `http://127.0.0.1:8000/v1` (via a reverse proxy edge, token-authed)
- Instance IP/port allocation changes if the the target hardware deployment is recreated —
  update `dsh/settings.snippet.yaml` and the curl example above when it does.
