#!/usr/bin/env python3
"""
Parse llama-server output for speculative-decoding telemetry (Section 17).

llama-server's Prometheus /metrics exposes prompt/generation throughput but,
as of the DFlash2 PR branch, does NOT expose draft acceptance counters. Until
it does, acceptance has to be scraped from stderr.

If upstream still lacks these counters after validation, the clean fix is a
metrics PR upstream, not a permanent private patch (Section 28).

  ./llamacpp.py --log /tmp/qwen-replicas/replica-gpu0-9000.log
  ./llamacpp.py --metrics http://127.0.0.1:9000/metrics
"""
import argparse, json, re, sys, urllib.request

# Log line shapes vary across builds; keep patterns permissive and report
# which ones actually matched rather than silently returning zeros.
PATTERNS = {
    "drafted":  [r"n_draft\s*=\s*(\d+)", r"drafted\s*[:=]\s*(\d+)"],
    "accepted": [r"n_accept\s*=\s*(\d+)", r"accepted\s*[:=]\s*(\d+)"],
    "acceptance_rate": [r"accept(?:ance)?[_ ]rate\s*[:=]\s*([0-9.]+)"],
    "acceptance_length": [r"accept(?:ance)?[_ ]length\s*[:=]\s*([0-9.]+)",
                          r"n_accept_avg\s*[:=]\s*([0-9.]+)"],
}

def scrape_log(path):
    text = open(path, "r", errors="replace").read()
    out, matched = {}, []
    for key, pats in PATTERNS.items():
        for p in pats:
            m = re.findall(p, text, re.IGNORECASE)
            if m:
                out[key] = float(m[-1]); matched.append(p); break
    if out.get("drafted") and out.get("accepted") and "acceptance_rate" not in out:
        out["acceptance_rate"] = out["accepted"] / out["drafted"]
    return out, matched

def scrape_metrics(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(k in name for k in ("draft", "spec", "accept", "tokens_predicted", "prompt_tokens")):
            try:
                out[name] = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log"); ap.add_argument("--metrics")
    a = ap.parse_args()
    if not (a.log or a.metrics):
        ap.error("need --log or --metrics")
    res = {}
    if a.log:
        vals, matched = scrape_log(a.log)
        res["from_log"] = vals
        res["patterns_matched"] = matched
        if not vals:
            res["warning"] = ("No speculative counters found. Either the build emits none, "
                              "or the log format changed. Do NOT record acceptance as 0 — "
                              "record it as null/unmeasured.")
    if a.metrics:
        res["from_metrics"] = scrape_metrics(a.metrics)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
