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


def gpu_snapshot():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        rows = [[float(x) for x in ln.split(",")] for ln in out.splitlines() if ln.strip()]
        if not rows:
            return {}
        return {
            "vram_mib": sum(r[0] for r in rows),
            "gpu_util_pct": sum(r[1] for r in rows) / len(rows),
            "power_w": sum(r[2] for r in rows),
        }
    except Exception:
        return {}


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

    gpu = gpu_snapshot()
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
