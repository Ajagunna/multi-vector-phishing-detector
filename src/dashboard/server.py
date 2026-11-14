"""
PhishGuard dashboard — classify emails with multi-vector model.
Static UI lives in /static (HTML / CSS / JS).
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.infer import get_detector

HOST = "0.0.0.0"
PORT = int(os.environ.get("PHISHGUARD_PORT", "8765"))

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"

AUDIT_LOG: list = []
STATS = {"legitimate": 0, "phishing": 0, "total": 0}


def add_audit(result: dict, subject: str = ""):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "subject": subject[:80] if subject else "(no subject)",
        **{k: v for k, v in result.items() if k not in ("vt_results", "sandbox_verdicts")},
    }
    AUDIT_LOG.insert(0, entry)
    if len(AUDIT_LOG) > 200:
        AUDIT_LOG.pop()
    STATS["total"] += 1
    key = result.get("classification", "legitimate")
    STATS[key] = STATS.get(key, 0) + 1


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: Any):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str | None = None):
        if not path.is_file():
            return self._json(404, {"error": "not found"})
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if path.suffix in (".css", ".js", ".png", ".jpg", ".svg", ".woff2"):
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path == "/api/status":
            det = get_detector()
            return self._json(200, {
                "loaded": True,
                "trained": getattr(det, "trained", False),
                "features": [
                    "text_encoder", "url_entropy", "urgency", "credential",
                    "virustotal_heuristic", "attachment_sandbox",
                ],
            })
        if path == "/api/stats":
            return self._json(200, STATS)
        if path == "/api/audit":
            return self._json(200, AUDIT_LOG[:100])
        if path == "/api/benchmarks":
            bp = ROOT / "models" / "benchmark_results.json"
            if bp.exists():
                return self._json(200, json.loads(bp.read_text()))
            return self._json(200, {"models": {}, "note": "Run: python src/evaluation/run_benchmarks.py"})

        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            candidate = (STATIC_DIR / rel).resolve()
            if not str(candidate).startswith(str(STATIC_DIR.resolve())):
                return self._json(403, {"error": "forbidden"})
            return self._send_file(candidate)

        if path.startswith("/css/") or path.startswith("/js/") or path.startswith("/img/"):
            candidate = (STATIC_DIR / path.lstrip("/")).resolve()
            if str(candidate).startswith(str(STATIC_DIR.resolve())):
                return self._send_file(candidate)

        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/classify":
            body = self._read_json()
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "text required"})
            det = get_detector()
            t0 = time.perf_counter()
            result = det.classify(
                text,
                enable_vt=bool(body.get("enable_vt", True)),
                enable_sandbox=bool(body.get("enable_sandbox", True)),
            )
            if "latency_ms" not in result:
                result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            subject = text.split("\n", 1)[0][:80]
            add_audit(result, subject=subject)
            return self._json(200, result)
        return self._json(404, {"error": "not found"})


def main():
    port = PORT
    print("[DASHBOARD] Loading detector…")
    get_detector()
    print(f"[DASHBOARD] Serving on http://127.0.0.1:{port}/")
    print(f"[DASHBOARD] Static root: {STATIC_DIR}")
    HTTPServer((HOST, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
