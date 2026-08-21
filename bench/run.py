#!/usr/bin/env python3
"""
Engine-neutral benchmark runner for OpenAI-compatible endpoints
(llama-server, vLLM, SGLang).

Measures, per Section 4's three metrics:
  Metric A  single-stream decode throughput   (--concurrency 1)
  Metric B  aggregate throughput              (--concurrency N, aggregate=true)
  Metric C  end-to-end agent-facing latency   (TTFT / ITL / E2E, always collected)

TTFT and ITL require streaming; this uses stream=True and timestamps the first
content token. Non-streaming timing cannot separate prefill from decode.

Emits schema.json-conformant JSON. Never invents numbers: anything it could not
measure is null.

  ./run.py --url http://127.0.0.1:9000 --workload workloads/python.json \
           --profile 1x5090-fast --out ../results/
"""
import argparse, json, os, statistics, sys, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone

def post_stream(url, payload, timeout=600):
    """Yield (timestamp, delta_text, usage_or_None) for an OpenAI stream."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    key = os.environ.get("QWEN_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            choices = obj.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content") or ""
            yield time.perf_counter(), delta, usage


def one_request(url, model, messages, max_tokens, temperature):
    """Run a single streaming request; return timing dict."""
    # COMPATIBILITY ASSUMPTION: the endpoint accepts `stream_options.include_usage`
    # (OpenAI, vLLM and llama-server all do). It is what yields server-reported
    # token counts; without it we fall back to counting stream chunks, which
    # miscounts whenever a chunk carries more than one token.
    #
    # Servers that ignore unknown fields degrade safely to that fallback. A server
    # that *rejects* the field is detected below and fails loudly, because silently
    # switching to chunk-counting would publish wrong token metrics.
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    text = []
    usage = None
    try:
        for ts, delta, u in post_stream(url, payload):
            if u:
                usage = u
            if delta:
                if ttft is None:
                    ttft = ts - t0
                token_times.append(ts)
                text.append(delta)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        if "stream_options" in detail or "include_usage" in detail:
            return {"error":
                    "endpoint rejected `stream_options.include_usage` "
                    f"(HTTP {e.code}): {detail.strip()}\n"
                    "This endpoint cannot report server-side token counts. Refusing to "
                    "fall back to chunk-counting, which would publish wrong token "
                    "metrics. Use an endpoint that supports stream_options, or record "
                    "the result as status=failed."}
        return {"error": f"HTTPError {e.code}: {detail.strip() or e.reason}"}
    except Exception as e:  # noqa: BLE001 - surface, never silently zero-fill
        return {"error": f"{type(e).__name__}: {e}"}
    t_end = time.perf_counter()

    # Prefer server-reported token counts; fall back to stream chunk count,
    # which over/under-counts when a chunk carries multiple tokens.
    completion_tokens = (usage or {}).get("completion_tokens")
    counted = completion_tokens if completion_tokens else len(token_times)
    decode_span = (token_times[-1] - token_times[0]) if len(token_times) > 1 else None

    return {
        "ttft_s": ttft,
        "e2e_s": t_end - t0,
        "completion_tokens": counted,
        "completion_tokens_reported": completion_tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        # Decode rate excludes prefill: measured from first to last token.
        "decode_tps": ((counted - 1) / decode_span) if (decode_span and counted and counted > 1) else None,
        "itl_ms": ((decode_span / (counted - 1)) * 1000) if (decode_span and counted and counted > 1) else None,
        "text": "".join(text),
    }


# --- GPU measurement -------------------------------------------------------
#
# nvidia-smi always reports PHYSICAL device indices and ignores
# CUDA_VISIBLE_DEVICES. The inference server, however, only ever touches the
# devices it was given. Summing every GPU on the host therefore produces wrong
# numbers whenever a 1-GPU profile runs on a 2- or 4-GPU box.
#
# So: query all GPUs, then keep only the participating ones, and report both
# per-GPU detail and an aggregate over that subset alone.

GPU_QUERY_FIELDS = ["index", "name", "memory.used", "utilization.gpu", "power.draw", "uuid"]


def _num(value):
    """nvidia-smi emits '[N/A]' / '[Not Supported]' for unsupported sensors."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_nvidia_smi_csv(text):
    """Parse `--query-gpu=<GPU_QUERY_FIELDS> --format=csv,noheader,nounits`."""
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(GPU_QUERY_FIELDS):
            continue
        idx = _num(parts[0])
        if idx is None:
            continue
        rows.append({
            "index": int(idx),
            "name": parts[1],
            "vram_mib": _num(parts[2]),
            "gpu_util_pct": _num(parts[3]),
            "power_w": _num(parts[4]),
            "uuid": parts[5],
        })
    return rows


def resolve_gpu_selection(rows, cuda_visible_devices=None, explicit=None):
    """
    Decide which physical GPUs are participating.

    Precedence: explicit selection (--gpus / profile GPU(S)) > CUDA_VISIBLE_DEVICES
    > all GPUs on the host.

    Accepts physical indices ("0,1") or UUIDs ("GPU-abc,GPU-def"). Mirrors CUDA's
    rule that an invalid entry truncates the list from that point onward, so the
    selection here matches what the server actually saw.

    Returns (selected_rows, source_label).
    """
    by_index = {r["index"]: r for r in rows}
    by_uuid = {r["uuid"]: r for r in rows if r.get("uuid")}

    def resolve(spec):
        picked = []
        for tok in [t.strip() for t in spec.split(",")]:
            if not tok:
                continue
            if tok in by_uuid:
                row = by_uuid[tok]
            else:
                try:
                    row = by_index.get(int(tok))
                except ValueError:
                    row = None
            if row is None:
                break  # CUDA truncates at the first invalid entry
            if row not in picked:
                picked.append(row)
        return picked

    if explicit is not None and str(explicit).strip() != "":
        return resolve(str(explicit)), "explicit"
    if cuda_visible_devices is not None:
        # An empty CUDA_VISIBLE_DEVICES means "no GPUs visible", which is
        # different from the variable being unset.
        if cuda_visible_devices.strip() == "":
            return [], "CUDA_VISIBLE_DEVICES(empty)"
        return resolve(cuda_visible_devices), "CUDA_VISIBLE_DEVICES"
    return list(rows), "all-host-gpus"


def summarize_gpus(selected):
    """Aggregate over participating GPUs only. Missing sensors stay None."""
    if not selected:
        return {}
    vram = [g["vram_mib"] for g in selected if g["vram_mib"] is not None]
    util = [g["gpu_util_pct"] for g in selected if g["gpu_util_pct"] is not None]
    power = [g["power_w"] for g in selected if g["power_w"] is not None]
    return {
        # Backward-compatible aggregate fields, now scoped to the selection.
        "vram_mib": sum(vram) if vram else None,
        "gpu_util_pct": (sum(util) / len(util)) if util else None,
        "power_w": sum(power) if power else None,
        "per_gpu": [{k: g[k] for k in ("index", "name", "vram_mib", "gpu_util_pct", "power_w")}
                    for g in selected],
    }


def gpu_snapshot(explicit=None):
    """
    Snapshot the participating GPUs.

    NOTE: this reads the LOCAL host's nvidia-smi. If run.py is pointed at a
    remote endpoint, these numbers describe the wrong machine —
    `gpu_selection.source` is recorded so that is auditable in the artifact.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    rows = parse_nvidia_smi_csv(out)
    if not rows:
        return {}
    selected, source = resolve_gpu_selection(
        rows, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), explicit=explicit)
    snap = summarize_gpus(selected)
    snap["gpu_selection"] = {
        "source": source,
        "indices": [g["index"] for g in selected],
        "host_gpu_count": len(rows),
    }
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:9000")
    ap.add_argument("--urls", nargs="*", default=None,
                    help="Multiple replica URLs (Profiles B/E). Sets aggregate=true.")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "default"))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--profile", default="unknown")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3, help="Section 23 requires >=3 warm runs.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default=os.environ.get("BENCH_GPUS") or os.environ.get("GPUS")
                    or os.environ.get("GPU"),
                    help="Physical GPU indices or UUIDs participating in this run "
                         "(e.g. '0' or '2,3'). Defaults to BENCH_GPUS/GPUS/GPU env, "
                         "then CUDA_VISIBLE_DEVICES, then all host GPUs.")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    wl = json.load(open(args.workload))
    messages = wl["messages"]
    targets = args.urls or [args.url]
    concurrency = max(args.concurrency, len(targets))
    aggregate = concurrency > 1

    def call(i):
        return one_request(targets[i % len(targets)], args.model, messages,
                           args.max_tokens, args.temperature)

    for _ in range(args.warmup):
        call(0)

    per_run = []
    for _ in range(args.runs):
        results, threads = [None] * concurrency, []
        wall0 = time.perf_counter()
        for i in range(concurrency):
            def worker(idx=i):
                results[idx] = call(idx)
            t = threading.Thread(target=worker); t.start(); threads.append(t)
        for t in threads:
            t.join()
        wall = time.perf_counter() - wall0
        ok = [r for r in results if r and "error" not in r]
        per_run.append({"wall_s": wall, "results": results, "ok": len(ok),
                        "errors": len(results) - len(ok)})

    gpu = gpu_snapshot(explicit=args.gpus)
    flat = [r for run in per_run for r in run["results"] if r and "error" not in r]
    errors = sum(run["errors"] for run in per_run)

    def med(key):
        vals = [r[key] for r in flat if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    if aggregate:
        # Aggregate = total completion tokens / wall-clock of the whole batch.
        rates = []
        for run in per_run:
            toks = sum((r.get("completion_tokens") or 0)
                       for r in run["results"] if r and "error" not in r)
            if run["wall_s"] > 0 and toks:
                rates.append(toks / run["wall_s"])
        decode_tps = statistics.median(rates) if rates else None
    else:
        decode_tps = med("decode_tps")

    depth = med("prompt_tokens")
    result = {
        "schema_version": 1,
        "status": "measured" if flat else "failed",
        "aggregate": aggregate,
        "hardware": {"gpu": None, "gpu_count": None, "driver": None, "cuda": None,
                     "p2p_supported": None,
                     "_note": "merge scripts/detect-hardware.sh output here"},
        "engine": {"name": "llama.cpp", "version_or_commit": os.environ.get("ENGINE_COMMIT", "UNRECORDED"),
                   "dflash_pr": int(os.environ["DFLASH_PR"]) if os.environ.get("DFLASH_PR") else None,
                   "build_flags": os.environ.get("BUILD_FLAGS", "")},
        "model": {"id": args.model, "sha256": os.environ.get("MODEL_SHA256"),
                  "quant": os.environ.get("MODEL_QUANT", "unknown"),
                  "lineage": os.environ.get("MODEL_LINEAGE", "unknown"),
                  "draft_model": os.environ.get("DRAFT_MODEL_ID"),
                  "draft_sha256": os.environ.get("DRAFT_SHA256"),
                  "draft_quant": os.environ.get("DRAFT_QUANT")},
        "runtime": {"profile": args.profile,
                    "context_size": int(os.environ.get("CTX", 0)) or None,
                    "context_depth": int(depth) if depth else None,
                    "kv_k": os.environ.get("KV_K", "unknown"),
                    "kv_v": os.environ.get("KV_V", "unknown"),
                    "split_mode": os.environ.get("SPLIT_MODE", "none"),
                    "tensor_split": os.environ.get("TENSOR_SPLIT"),
                    "concurrency": concurrency,
                    "speculation": {"type": os.environ.get("SPEC_KIND", "none"),
                                    "n_max": int(os.environ["SPEC_N_MAX"]) if os.environ.get("SPEC_N_MAX") else None}},
        "metrics": {
            "decode_tps": round(decode_tps, 2) if decode_tps else None,
            "prompt_tps": None,  # from llama-bench / server metrics, not inferable here
            "ttft_ms": round(med("ttft_s") * 1000, 2) if med("ttft_s") else None,
            "itl_ms": round(med("itl_ms"), 3) if med("itl_ms") else None,
            "e2e_ms": round(med("e2e_s") * 1000, 2) if med("e2e_s") else None,
            "vram_mib": gpu.get("vram_mib"), "gpu_util_pct": gpu.get("gpu_util_pct"),
            "power_w": gpu.get("power_w"),
            "per_gpu": gpu.get("per_gpu"),
            "gpu_selection": gpu.get("gpu_selection"),
            "draft_accepted": None, "draft_drafted": None,
            "draft_acceptance_rate": None, "mean_acceptance_length": None,
            "_spec_note": "acceptance metrics need parsers/llamacpp.py — see docs/dflash.md",
            "errors": errors,
        },
        "correctness": {"greedy_match_vs_no_spec": None, "tool_call_ok": None,
                        "needle_recall": None, "notes": args.note},
        "workload": os.path.basename(args.workload),
        "warmup_runs": args.warmup, "measured_runs": args.runs,
        "raw": {"per_run_wall_s": [r["wall_s"] for r in per_run]},
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    out = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = f"{args.profile}-{os.path.splitext(os.path.basename(args.workload))[0]}-c{concurrency}.json"
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            f.write(out + "\n")
        print(f"wrote {path}", file=sys.stderr)
    print(out)
    if aggregate:
        print("\nNOTE: aggregate=true — this is summed across "
              f"{concurrency} concurrent requests, NOT single-request speed.", file=sys.stderr)

if __name__ == "__main__":
    main()
