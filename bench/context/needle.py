#!/usr/bin/env python3
"""
Long-context retrieval test (Section 7).

"The server started with -c 262144" is NOT evidence that 256K context works.
This inserts unique facts at controlled depths and checks retrieval, so a
context claim is backed by recall numbers.

Probes at 5% / 25% / 50% / 75% / 95% depth.

  ./needle.py --url http://127.0.0.1:9000 --depth 131072
  ./needle.py --url http://127.0.0.1:9000 --sweep 8192,32768,65536,131072,196608,262144
"""
import argparse, json, os, random, sys, time, urllib.request

FILLER = ("The distributed ledger reconciliation subsystem processes batched entries "
          "according to the regional compliance schedule defined in the operations manual. ")
POSITIONS = [0.05, 0.25, 0.50, 0.75, 0.95]

def chat(url, messages, max_tokens=64, temperature=0.0, timeout=900):
    payload = {"model": os.environ.get("BENCH_MODEL", "default"), "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if os.environ.get("QWEN_API_KEY"):
        req.add_header("Authorization", f"Bearer {os.environ['QWEN_API_KEY']}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return (obj["choices"][0]["message"]["content"],
            obj.get("usage", {}).get("prompt_tokens"),
            time.perf_counter() - t0)

def build(depth_tokens, needles):
    # ~4 chars/token is a rough heuristic; actual depth is read back from
    # usage.prompt_tokens and that reported value is what gets recorded.
    target_chars = depth_tokens * 4
    unit = len(FILLER)
    body = []
    placed = {}
    total = 0
    idx = 0
    while total < target_chars:
        for frac, (key, secret) in needles.items():
            at = int(target_chars * frac)
            if key not in placed and total >= at:
                s = f"\n\nIMPORTANT FACT: the {key} access code is {secret}.\n\n"
                body.append(s); total += len(s); placed[key] = frac
        body.append(FILLER); total += unit; idx += 1
    return "".join(body), placed

def run_depth(url, depth, seed=0):
    rnd = random.Random(seed)
    needles = {f: (k, f"{rnd.randint(10**7, 10**8-1)}")
               for f, k in zip(POSITIONS, ["alpha", "bravo", "charlie", "delta", "echo"])}
    haystack, placed = build(depth, needles)
    hits, results = 0, []
    reported_depth = None
    for frac, (key, secret) in needles.items():
        msgs = [{"role": "system", "content": "Answer using only the provided document."},
                {"role": "user", "content": haystack +
                 f"\n\nQuestion: what is the {key} access code? Reply with digits only."}]
        try:
            ans, ptok, dt = chat(url, msgs)
        except Exception as e:
            results.append({"position": frac, "key": key, "ok": False, "error": str(e)})
            continue
        reported_depth = ptok or reported_depth
        ok = secret in (ans or "")
        hits += ok
        results.append({"position": frac, "key": key, "ok": ok,
                        "expected": secret, "got": (ans or "").strip()[:60],
                        "latency_s": round(dt, 2), "prompt_tokens": ptok})
        print(f"  depth~{depth:>7} pos {frac:>5.0%} {key:<8} {'PASS' if ok else 'FAIL'} "
              f"({dt:.1f}s, prompt_tokens={ptok})", file=sys.stderr)
    return {"requested_depth": depth, "reported_prompt_tokens": reported_depth,
            "recall": hits / len(needles), "probes": results}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--depth", type=int, default=131072)
    ap.add_argument("--sweep", default=None, help="comma-separated depths")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    depths = [int(x) for x in a.sweep.split(",")] if a.sweep else [a.depth]
    out = []
    for d in depths:
        print(f"depth {d}:", file=sys.stderr)
        out.append(run_depth(a.url, d))
    blob = {"needle_results": out,
            "note": "recall < 1.0 at a depth means that context length is NOT usable, "
                    "even if the server accepted -c at that size."}
    s = json.dumps(blob, indent=2)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)

if __name__ == "__main__":
    main()
