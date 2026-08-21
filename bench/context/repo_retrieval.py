#!/usr/bin/env python3
"""
Long-code retrieval test (Section 7).

More realistic than synthetic needles: builds a synthetic source tree, fills
the context with it, then asks questions that require retrieving a specific
definition from a distant file. Closer to how an agent actually uses long
context.

  ./repo_retrieval.py --url http://127.0.0.1:9000 --depth 131072
"""
import argparse, json, os, random, sys, time, urllib.request

TEMPLATE = '''
# ---- file: {path} ----
import dataclasses
from typing import Optional

@dataclasses.dataclass
class {cls}:
    """Domain object for the {mod} subsystem."""
    identifier: str
    threshold: int = {thresh}
    enabled: bool = True

    def evaluate(self, value: int) -> bool:
        return self.enabled and value > self.threshold

def {fn}(records: list) -> int:
    """Aggregate {mod} records above threshold."""
    return sum(1 for r in records if r > {thresh})
'''

def build(depth_tokens, seed=0):
    rnd = random.Random(seed)
    mods = ["billing", "ingest", "scheduler", "telemetry", "auth", "cache",
            "router", "planner", "indexer", "replicator"]
    target_chars = depth_tokens * 4
    files, facts, total = [], [], 0
    i = 0
    while total < target_chars:
        mod = mods[i % len(mods)] + str(i)
        thresh = rnd.randint(1000, 9999)
        blob = TEMPLATE.format(path=f"src/{mod}/core.py", cls=mod.capitalize() + "Engine",
                               mod=mod, thresh=thresh, fn=f"count_{mod}")
        files.append(blob); total += len(blob)
        facts.append((mod, thresh))
        i += 1
    return "".join(files), facts

def ask(url, tree, mod, timeout=900):
    msgs = [{"role": "system", "content": "Answer using only the provided source tree."},
            {"role": "user", "content": tree +
             f"\n\nQuestion: what is the default `threshold` value of "
             f"{mod.capitalize()}Engine in src/{mod}/core.py? Reply with the integer only."}]
    payload = {"model": os.environ.get("BENCH_MODEL", "default"), "messages": msgs,
               "max_tokens": 32, "temperature": 0.0}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if os.environ.get("QWEN_API_KEY"):
        req.add_header("Authorization", f"Bearer {os.environ['QWEN_API_KEY']}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return (obj["choices"][0]["message"]["content"],
            obj.get("usage", {}).get("prompt_tokens"), time.perf_counter() - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--depth", type=int, default=131072)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tree, facts = build(a.depth)
    # Probe across the tree: early, middle, late definitions.
    idxs = [int(len(facts) * f) for f in (0.05, 0.25, 0.5, 0.75, 0.95)][:a.probes]
    hits, results = 0, []
    for i in idxs:
        mod, thresh = facts[min(i, len(facts) - 1)]
        try:
            ans, ptok, dt = ask(a.url, tree, mod)
        except Exception as e:
            results.append({"module": mod, "ok": False, "error": str(e)}); continue
        ok = str(thresh) in (ans or "")
        hits += ok
        results.append({"module": mod, "position": round(i / len(facts), 2), "ok": ok,
                        "expected": thresh, "got": (ans or "").strip()[:40],
                        "prompt_tokens": ptok, "latency_s": round(dt, 2)})
        print(f"  {mod:<16} pos {i/len(facts):>5.0%} {'PASS' if ok else 'FAIL'} "
              f"({dt:.1f}s, prompt_tokens={ptok})", file=sys.stderr)

    blob = {"requested_depth": a.depth, "recall": hits / max(len(results), 1),
            "probes": results}
    s = json.dumps(blob, indent=2)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)

if __name__ == "__main__":
    main()
