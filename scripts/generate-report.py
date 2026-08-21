#!/usr/bin/env python3
"""
Build Markdown tables from results/**.json (Section 26).

Refuses to mix single-stream and aggregate numbers in one table, because that
is the single easiest way to publish a misleading throughput claim.

  ./generate-report.py --results ../results --out ../docs/findings-generated.md
"""
import argparse, glob, json, os, sys

def load(root):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if os.path.basename(p).startswith("_") or p.endswith("hardware.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr); continue
        if d.get("schema_version") != 1:
            continue
        d["_path"] = os.path.relpath(p, root)
        rows.append(d)
    return rows

def fmt(v, suffix="", nd=1):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{suffix}"
    return str(v)

def table(rows, aggregate):
    sel = [r for r in rows if bool(r.get("aggregate")) == aggregate
           and r.get("status") == "measured"]
    if not sel:
        return "_No measured results yet._\n"
    sel.sort(key=lambda r: (r["metrics"].get("decode_tps") or 0), reverse=True)
    head = ("| Profile | GPUs | Quant | KV | Ctx depth | Spec | Conc | "
            f"{'Aggregate tok/s' if aggregate else 'Decode tok/s'} | TTFT ms | ITL ms | Accept | Result |\n")
    head += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    out = []
    for r in sel:
        m, rt, mo, hw = r["metrics"], r["runtime"], r["model"], r["hardware"]
        spec = rt.get("speculation") or {}
        spec_s = spec.get("type", "none")
        if spec.get("n_max"):
            spec_s += f"/{spec['n_max']}"
        out.append("| {} | {} | {} | {}/{} | {} | {} | {} | **{}** | {} | {} | {} | [{}]({}) |".format(
            rt.get("profile", "?"), hw.get("gpu_count", "?"), mo.get("quant", "?"),
            rt.get("kv_k", "?"), rt.get("kv_v", "?"), fmt(rt.get("context_depth"), nd=0),
            spec_s, rt.get("concurrency", 1),
            fmt(m.get("decode_tps"), nd=1), fmt(m.get("ttft_ms"), nd=0),
            fmt(m.get("itl_ms"), nd=2),
            fmt((m.get("draft_acceptance_rate") or 0) * 100, "%", 0) if m.get("draft_acceptance_rate") else "—",
            os.path.basename(r["_path"]), r["_path"]))
    return head + "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.results)
    measured = [r for r in rows if r.get("status") == "measured"]
    md = ["# Generated benchmark report", "",
          f"_Auto-generated from `{a.results}/` — do not edit by hand._", "",
          f"Result files: {len(rows)} | measured: {len(measured)} | "
          f"untested/failed: {len(rows)-len(measured)}", ""]
    if not measured:
        md += ["> **No measured results yet.** Every configuration in this repository is",
               "> currently **UNTESTED**. Populate `results/` by running `bench/run.py`", ""]
    md += ["## Metric A — single-request decode throughput", "",
           "One request, one sequence. This is the headline latency number.", "",
           table(rows, aggregate=False), "",
           "## Metric B — aggregate throughput (concurrent independent requests)", "",
           "**Summed across concurrent requests. NOT single-request speed.**", "",
           table(rows, aggregate=True), ""]
    hw = {r["hardware"].get("gpu") for r in measured if r["hardware"].get("gpu")}
    if hw:
        md += ["## Hardware seen in results", ""] + [f"- {h}" for h in sorted(hw)] + [""]
    text = "\n".join(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(text)
        print(f"wrote {a.out}", file=sys.stderr)
    print(text)

if __name__ == "__main__":
    main()
