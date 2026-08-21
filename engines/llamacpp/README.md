# llama.cpp — GGUF + DFlash2 experimental track

**Status: entirely UNTESTED.** Scripts are written and reviewed; no number here
has been produced on hardware.

## Blocking prerequisite

**DFlash2** is not in llama.cpp master — though the *flags* are. `--spec-type
draft-dflash` is valid on master (that's DFlash v1), as are
`--spec-draft-n-max`, `--spec-draft-device`, `--spec-draft-ngl`.

What master lacks is DFlash2 itself: [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)
(branch `dflash2`), **OPEN / unmerged as of 2026-08-21**, which adds the
convolution and selector tensors the DFlash2 checkpoints need.

Because the flags parse either way, a DFlash2 draft on master fails at **model
load**, not with a clean argument error. `build.sh` fetches the PR branch by
default; `_common.sh` warns but cannot fully verify DFlash2 — the build SHA is
the real evidence.

Re-check: `gh pr view 27342 --repo ggml-org/llama.cpp --json state,mergedAt`

## Quick start

```sh
# 1. build for Blackwell (SM120). WITH_NCCL=1 only if you plan Profile D.
./build.sh

# 2. models — TRACK=A stock (reference), TRACK=B abliterated (this repo's target)
#    Read docs/model-lineage.md first; they are NOT interchangeable.
TRACK=A ./download.sh

# 3. serve
./serve-single.sh 1x5090-fast 0 9000

# 4. benchmark
python3 ../../bench/run.py --url http://127.0.0.1:9000 \
  --workload ../../bench/workloads/python.json --profile 1x5090-fast \
  --out ../../results/
```

## Scripts

| Script | Profile | Notes |
|---|---|---|
| `serve-single.sh` | A | 1 GPU. The >100 tok/s candidate. |
| `serve-replicas.sh` | B, E | N independent replicas, one per GPU. **Aggregate** throughput. |
| `serve-layer.sh` | C | 2 GPUs, pipeline split. May lose to 1 GPU — measure. |
| `serve-tensor.sh` | D | Experimental. Needs FA + NCCL + **non-quantized KV**. Refuses quantized KV rather than failing obscurely. |

All take a profile name from `profiles/`. See `docs/multi-gpu.md` for which to
run in what order — and run `scripts/topology.sh` first.

## Recording provenance

`build.sh` writes `build-info.json` with the exact commit SHA. Export it into
the benchmark run so results are reproducible:

```sh
export ENGINE_COMMIT=$(jq -r .engine_commit ~/llama.cpp/build-info.json)
export DFLASH_PR=27342
```

An unpinned SHA makes a result worthless later — that branch is rebasable and
llama.cpp performance moves between revisions.
