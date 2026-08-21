#!/bin/bash
# Sanity/benchmark script for a running vLLM server. Run anywhere that can
# reach it (defaults to localhost) after scripts/serve.sh is up.
#
# Checks, in order:
#   1. Server is responding.
#   2. Decode throughput on a fixed, deterministic prompt.
#   3. Prefix caching actually produces hits — using a prefix long enough to
#      cross this model's KV cache block size (784 tokens; see README.md
#      "Benchmarking gotcha" before shrinking this prompt — a shorter test
#      prefix will show zero hits regardless of whether caching works).
#      Reads /metrics directly for ground truth, not the periodic log line.
#
# Env overrides: VLLM_URL (default http://127.0.0.1:8000), MODEL (default
# below).
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen3.8-27B-Uncensored-FP8}"

echo "=== 1. models endpoint ==="
curl -sf -m 10 "$VLLM_URL/v1/models" | python3 -m json.tool

echo
echo "=== 2. decode throughput (temperature=0, fixed prompt, 3 runs) ==="
VLLM_URL="$VLLM_URL" MODEL="$MODEL" python3 <<'PYEOF'
import json, os, time, urllib.request

url = os.environ['VLLM_URL'] + '/v1/chat/completions'
model = os.environ['MODEL']

payload = json.dumps({
    'model': model,
    'messages': [{'role': 'user', 'content':
        'Write a 150-word explanation of how tensor parallelism works in '
        'transformer inference, no markdown.'}],
    'max_tokens': 220,
    'temperature': 0,
}).encode()

for i in range(3):
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    t1 = time.time()
    u = resp['usage']
    print(f'run {i}: completion_tokens={u["completion_tokens"]} '
          f'elapsed_s={round(t1 - t0, 3)} tok/s={round(u["completion_tokens"] / (t1 - t0), 2)}')
PYEOF

echo
echo "=== 3. prefix-cache correctness + real hit rate ==="
VLLM_URL="$VLLM_URL" MODEL="$MODEL" python3 <<'PYEOF'
import json, os, urllib.request

base = os.environ['VLLM_URL']
model = os.environ['MODEL']

# Deliberately long: must clear the KV cache block size (784 tokens for
# this hybrid mamba/attention model) or every query is a guaranteed miss
# regardless of whether caching is actually working. ~3500 tokens here.
SENTENCE = (
    'You are a precise technical assistant. Repository context: this is a '
    'Python service for order processing. Function signature: '
    'def compute_total(items: list[dict]) -> float. Each item has keys '
    '"price" (float) and "qty" (int). Answer questions about this exactly. '
)
SHARED_PREFIX = SENTENCE * 60

def ask(question):
    payload = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': SHARED_PREFIX},
            {'role': 'user', 'content': question},
        ],
        'max_tokens': 60,
        'temperature': 0,
    }).encode()
    req = urllib.request.Request(base + '/v1/chat/completions', data=payload,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp['choices'][0]['message']['content'], resp['usage']['prompt_tokens']

def read_metric(name):
    req = urllib.request.Request(base + '/metrics')
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read().decode()
    for line in body.splitlines():
        if line.startswith(name + '{'):
            return float(line.rsplit(' ', 1)[1])
    return None

before = read_metric('vllm:prefix_cache_hits_total')

q = 'Write the one-line body of compute_total using a generator expression.'
a1, pt1 = ask(q)
a2, pt2 = ask(q)

after = read_metric('vllm:prefix_cache_hits_total')

print('prompt_tokens (should match, same prompt both times):', pt1, pt2)
print('output identical on repeat (expected under temperature=0):', a1 == a2)
print('prefix_cache_hits_total before/after:', before, '->', after)
if after is not None and before is not None and after > before:
    print('PASS: prefix caching produced real hits.')
else:
    print('WARN: no new hits recorded — check config (see README "vLLM flags" table).')
PYEOF
