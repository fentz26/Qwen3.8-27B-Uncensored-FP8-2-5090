# Replica router

Session-affinity HTTP router for independent replicas (Profiles B and E).

```sh
python3 router.py --replicas http://127.0.0.1:9000 http://127.0.0.1:9001 --port 8080
curl localhost:8080/status
```

## Why affinity beats balance here

Each replica keeps **its own prefix cache**. An agent session that bounces
between replicas re-prefills its entire system prompt and tool schema on every
hop — which, for agent traffic, usually costs far more than imperfect load
balance saves.

Policy:
1. **Sticky** — a known session returns to its replica while healthy and under
   `--max-inflight`.
2. **Least-loaded** otherwise — fewest in-flight, ties broken on tokens served.

Session id resolution: `X-Session-Id` header → `user` field in the body → hash
of the leading system message. That last one gives **prefix affinity** for free:
two requests sharing a system prompt land on the same replica.

Response headers `X-Routed-To` / `X-Route-Reason` make routing decisions
visible during benchmarking.

## Reporting

Throughput measured through this router across concurrent requests is
**aggregate**. `/status` says so, and `bench/run.py --urls ...` sets
`aggregate: true` automatically.

## Scope

Stdlib-only threaded proxy, sized for benchmarking a handful of local replicas.
Not a production ingress: no TLS, no auth beyond pass-through, no retry
budget, in-memory session table.
