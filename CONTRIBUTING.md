# Contributing

Two kinds of contribution: **benchmark results** and **optimization changes**.
Both need evidence.

## Ground rules

1. **No unmeasured performance claims.** "Should be faster" is a hypothesis;
   file it as an issue, not a README edit.
2. **Never fabricate numbers.** Untested configurations say `UNTESTED`.
3. **Missing measurement is `null`, not `0`.**
4. **Aggregate ≠ single-request.** Set `aggregate` correctly.
5. **Negative results are welcome.** "Tensor split lost to one GPU on my box"
   is genuinely useful — it saves the next person the hardware time.

## Submitting a benchmark

1. `scripts/detect-hardware.sh > results/<host>-hardware.json`
2. `scripts/topology.sh` (paste output in the PR)
3. Run it:
   ```sh
   ENGINE_COMMIT=$(git -C ~/llama.cpp rev-parse HEAD) \
   MODEL_QUANT=UD-Q4_K_XL MODEL_LINEAGE=stock SPEC_KIND=dflash2 SPEC_N_MAX=7 \
   CTX=131072 KV_K=q4_0 KV_V=q4_0 \
   python3 bench/run.py --url http://127.0.0.1:9000 \
     --workload bench/workloads/python.json \
     --profile 1x5090-fast --runs 3 --out results/rtx5090/1gpu/
   ```
4. Validate context claims: `scripts/validate-context.py --url ... --out ...`
5. Regenerate: `scripts/generate-report.py --results results --out docs/findings-generated.md`

### Layout

```
results/rtx5090/<gpu-count>gpu/<engine>-<commit-short>/<profile>-<workload>-c<N>.json
```

### Required in every artifact

GPU model/count · driver · CUDA · **P2P status** · engine name + **exact commit
SHA** (never "latest") · model id + SHA256 + quant + **lineage** · draft id +
SHA256 + quant · context size **and** measured depth · KV types · split mode ·
tensor split · batch/ubatch · parallel slots · concurrency · speculation config
· warmup + measured run counts · whether results are single-stream or aggregate
· correctness notes · which GPUs participated (`metrics.gpu_selection`, filled
automatically — check `indices` matches the profile you ran).

Validate before submitting:
```sh
python3 -c "import json,jsonschema;jsonschema.validate(json.load(open('YOUR.json')),json.load(open('bench/schema.json')))"
```

### What gets rejected

* screenshots **instead of** raw artifacts (as a supplement: fine)
* `"latest"` as an engine version
* aggregate numbers presented as single-request
* a stock GGUF labelled uncensored (see `docs/model-lineage.md`)
* context claims without recall data
* single-run results with no warmup

## Submitting an optimization

Include a before/after pair of result artifacts from the **same machine**,
same commit except your change, ≥3 warm runs each. State the workload — a
speedup on JSON that regresses creative prose is a tradeoff, not a win, and
should be documented as one.

## Upstream first

If a finding is really about llama.cpp or DFlash rather than this repo
(kernel behaviour, missing metrics, split-mode guidance), consider an upstream
issue/PR with a minimal reproduction. See `docs/dflash.md`. Don't upstream
speculative claims.
