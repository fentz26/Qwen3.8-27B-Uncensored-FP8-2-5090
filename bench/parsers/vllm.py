#!/usr/bin/env python3
"""
Parse vLLM /metrics, including spec-decode counters.

Unlike llama.cpp, vLLM DOES expose speculative-decoding metrics, and its boot
log prints acceptance directly, e.g.:
  SpecDecoding metrics: Mean acceptance length: 1.74, ... Avg Draft acceptance rate: 73.8%

Also exposes prefix cache counters, which are the ground truth for cache hits.
NOTE: a prefix-cache test whose shared prefix is shorter than the KV block size
will report zero hits regardless of whether caching works — see
docs/methodology.md.
"""
import argparse, json, urllib.request

KEYS = ("prefix_cache", "spec_decode", "draft", "acceptance",
        "num_preemptions", "gpu_cache_usage", "time_to_first_token",
        "time_per_output_token", "generation_tokens", "prompt_tokens")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    a = ap.parse_args()
    with urllib.request.urlopen(a.url, timeout=10) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(k in name for k in KEYS):
            try:
                out.setdefault(name, []).append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    agg = {k: (sum(v) if len(v) > 1 else v[0]) for k, v in out.items()}
    q = agg.get("vllm:prefix_cache_queries_total")
    h = agg.get("vllm:prefix_cache_hits_total")
    if q:
        agg["_prefix_cache_hit_rate"] = (h or 0) / q
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
