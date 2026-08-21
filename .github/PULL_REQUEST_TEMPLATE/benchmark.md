## Type
- [ ] Benchmark result
- [ ] Optimization (include before/after artifacts)
- [ ] Docs / tooling

## Hardware
| | |
|---|---|
| GPU model | |
| GPU count | |
| VRAM per GPU | |
| Driver | |
| CUDA | |
| **P2P supported** | yes / no / n-a |
| PCIe topology | paste `nvidia-smi topo -m` |

## Software
| | |
|---|---|
| Engine | llama.cpp / vLLM / other |
| **Exact commit SHA** | (not "latest") |
| DFlash PR (if any) | e.g. 27342 |
| Build flags | |

## Model
| | |
|---|---|
| Target model id | |
| SHA256 | |
| Quant | |
| **Lineage** | stock / abliterated / requantized-from-fp8 |
| Draft model id + SHA256 + quant | |

## Run
| | |
|---|---|
| Exact command | |
| Profile | |
| Context size (`-c`) | |
| **Measured context depth** (prompt_tokens) | |
| KV K / V | |
| Split mode / tensor split | |
| Concurrency | |
| **Single-stream or aggregate?** | |
| Warmup runs / measured runs | / |

## Results
Attach the raw JSON under `results/`. Paste the summary:

| Metric | Value |
|---|---|
| Decode tok/s | |
| TTFT ms | |
| ITL ms | |
| VRAM MiB | |
| Draft acceptance rate | |
| Mean acceptance length | |

## Correctness
- [ ] No OOM across the run
- [ ] No corrupted / truncated generations
- [ ] Tool calling verified (if applicable)
- [ ] Greedy output matches non-speculative baseline (if speculation on)
- [ ] Long-context recall attached (if claiming a context length)

## Checklist
- [ ] Raw artifact committed (not only a screenshot)
- [ ] Artifact validates against `bench/schema.json`
- [ ] `aggregate` field set correctly
- [ ] Unmeasured fields are `null`, not `0`
- [ ] Engine SHA pinned
