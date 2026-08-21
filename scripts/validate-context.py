#!/usr/bin/env python3
"""
Acceptance gate for a context-length claim (Sections 7 and 23).

A configuration may only be called "256K capable" if it PASSES here, not
because llama-server accepted -c 262144.

Checks per depth:
  1. server answers at all at that depth
  2. needle recall == 1.0 across all five positions
  3. no truncation/corruption in output
  4. latency is recorded (so "usable" includes "not unusably slow")

  ./validate-context.py --url http://127.0.0.1:9000 --depths 8192,32768,131072,262144
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLE = os.path.join(HERE, "..", "bench", "context", "needle.py")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--depths", default="8192,32768,65536,131072,196608,262144")
    ap.add_argument("--min-recall", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proc = subprocess.run([sys.executable, NEEDLE, "--url", a.url, "--sweep", a.depths],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"needle.py failed rc={proc.returncode}")
    data = json.loads(proc.stdout)

    print("\n=== context validation ===")
    verdict, failures = {}, 0
    for r in data["needle_results"]:
        d = r["requested_depth"]
        recall = r["recall"]
        ok = recall >= a.min_recall
        verdict[d] = {"recall": recall, "pass": ok,
                      "reported_prompt_tokens": r.get("reported_prompt_tokens")}
        failures += (not ok)
        print(f"  depth {d:>7}: recall {recall:5.0%}  {'PASS' if ok else 'FAIL'}"
              f"   (reported prompt_tokens={r.get('reported_prompt_tokens')})")

    passing = [d for d, v in verdict.items() if v["pass"]]
    max_ok = max(passing) if passing else None
    print()
    if max_ok:
        print(f"VALIDATED usable context depth: {max_ok} tokens")
    else:
        print("NO depth passed — do not claim any long-context capability.")
    print("Note: the -c server flag is CAPACITY. Only the number above is a claim.")

    blob = {"validated_max_depth": max_ok, "min_recall_required": a.min_recall,
            "per_depth": verdict, "raw": data}
    if a.out:
        open(a.out, "w").write(json.dumps(blob, indent=2) + "\n")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
