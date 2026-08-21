#!/usr/bin/env python3
"""
Session-affinity router for independent replicas (Profiles B and E).

Why affinity matters more than load balance here: each replica keeps its own
prefix cache. A multi-turn agent session that bounces between replicas
re-prefills its whole system prompt + tool schema on every hop. Sticky routing
usually beats "perfectly balanced" routing for agent traffic.

Routing policy:
  1. Sticky: an existing session id goes back to its replica while that
     replica is healthy and not overloaded.
  2. Otherwise least-loaded: fewest in-flight requests, ties broken by
     least total tokens served.

Session id resolution order: X-Session-Id header, then `user` field in the
JSON body, then a hash of the leading system message (prefix affinity - two
requests sharing a system prompt should land on the same replica).

Stdlib only. Single-process threaded proxy - adequate for benchmarking a
handful of local replicas, not a production ingress.

  ./router.py --replicas http://127.0.0.1:9000 http://127.0.0.1:9001 --port 8080
"""
import argparse, hashlib, json, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Replica:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.inflight = 0
        self.tokens = 0
        self.errors = 0
        self.healthy = True
        self.lock = threading.Lock()

    def to_dict(self):
        return {"url": self.url, "inflight": self.inflight, "tokens": self.tokens,
                "errors": self.errors, "healthy": self.healthy}

class Pool:
    def __init__(self, urls, max_inflight):
        self.replicas = [Replica(u) for u in urls]
        self.sessions = {}   # session id -> Replica
        self.max_inflight = max_inflight
        self.lock = threading.Lock()

    def pick(self, session):
        with self.lock:
            live = [r for r in self.replicas if r.healthy] or self.replicas
            if session and session in self.sessions:
                r = self.sessions[session]
                # Honour affinity unless that replica is saturated.
                if r.healthy and r.inflight < self.max_inflight:
                    return r, "sticky"
            r = min(live, key=lambda x: (x.inflight, x.tokens))
            if session:
                self.sessions[session] = r
            return r, "least-loaded"

def session_key(headers, body):
    sid = headers.get("X-Session-Id")
    if sid:
        return sid, "header"
    try:
        obj = json.loads(body)
    except Exception:
        return None, "none"
    if obj.get("user"):
        return str(obj["user"]), "user-field"
    for m in obj.get("messages") or []:
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c:
                # Prefix affinity: identical system prompt -> same replica -> cache reuse.
                return hashlib.sha256(c[:4096].encode()).hexdigest()[:16], "system-prefix"
            break
    return None, "none"

def make_handler(pool):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/healthz", "/status"):
                blob = {"replicas": [r.to_dict() for r in pool.replicas],
                        "sessions": len(pool.sessions),
                        "note": "Throughput across these replicas is AGGREGATE, "
                                "not single-request speed."}
                return self._send(200, json.dumps(blob, indent=2).encode())
            return self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            sid, how = session_key(self.headers, body)
            replica, why = pool.pick(sid)
            with replica.lock:
                replica.inflight += 1
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(replica.url + self.path, data=body,
                                             headers={"Content-Type": "application/json"})
                auth = self.headers.get("Authorization")
                if auth:
                    req.add_header("Authorization", auth)
                with urllib.request.urlopen(req, timeout=900) as r:
                    payload = r.read()
                    status = r.status
                replica.healthy = True
                try:
                    used = json.loads(payload).get("usage", {}).get("completion_tokens") or 0
                    with replica.lock:
                        replica.tokens += used
                except Exception:
                    pass
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Routed-To", replica.url)
                self.send_header("X-Route-Reason", f"{why}/{how}")
                self.send_header("X-Route-Latency-Ms", f"{(time.perf_counter()-t0)*1000:.1f}")
                self.end_headers()
                self.wfile.write(payload)
            except urllib.error.HTTPError as e:
                self._send(e.code, e.read())
            except Exception as e:
                with replica.lock:
                    replica.errors += 1
                if replica.errors >= 3:
                    replica.healthy = False
                self._send(502, json.dumps({"error": f"{type(e).__name__}: {e}",
                                            "replica": replica.url}).encode())
            finally:
                with replica.lock:
                    replica.inflight -= 1
    return H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-inflight", type=int, default=4,
                    help="Break session affinity once a replica exceeds this.")
    a = ap.parse_args()
    pool = Pool(a.replicas, a.max_inflight)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(pool))
    print(f"router on :{a.port} -> {', '.join(a.replicas)}")
    print("status: curl localhost:%d/status" % a.port)
    srv.serve_forever()

if __name__ == "__main__":
    main()
