# Results

Machine-readable benchmark artifacts. `scripts/generate-report.py` builds the
Markdown tables from these — never hand-edit generated tables.

## Layout

```
results/
  <host>-hardware.json                 from scripts/detect-hardware.sh
  rtx5090/
    1gpu/
      llama-cpp-<commit-short>/
        1x5090-fast-python-c1.json
        1x5090-256k-long_context-c1.json
    2gpu/
      llama-cpp-<commit-short>/
        2x5090-replicas-python-c2.json     # aggregate: true
    4gpu/
  vllm/
    <version>/
```

## Rules

* Schema: `bench/schema.json` (`schema_version: 1`).
* `status`: only `measured` may be cited. `UNTESTED` is a placeholder;
  `failed` records a config that did not work — keep those, they're useful.
* `aggregate: true` means summed over concurrent requests. Never present those
  as single-request speed.
* Unmeasured fields are `null`. A `0` asserts a measurement.
* Engine version must be an exact commit SHA.

## Current state

**Empty — no configuration in this repository has been benchmarked on
llama.cpp yet.** The only measured results are the vLLM baseline recorded in
`docs/findings.md`, from hardware that has since been released.

Contributions with real RTX 5090 hardware are the main thing this repo needs.
