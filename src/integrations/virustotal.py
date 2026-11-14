"""
VirusTotal URL reputation client (async-capable).

Uses the public VT v3 API when VIRUSTOTAL_API_KEY is set.
Falls back to a deterministic local heuristic when the key is absent
so the prototype remains runnable offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

VT_API_BASE = "https://www.virustotal.com/api/v3"
DEFAULT_TIMEOUT = 12
_CACHE: Dict[str, "VTResult"] = {}
_CACHE_TTL = 3600  # seconds


@dataclass
class VTResult:
    url: str
    malicious: int = 0
    suspicious: int = 0
    harmless: int = 0
    undetected: int = 0
    reputation: float = 0.0  # 0..1 higher = more malicious
    source: str = "heuristic"
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _api_key() -> Optional[str]:
    return os.environ.get("VIRUSTOTAL_API_KEY") or os.environ.get("VT_API_KEY")


def _url_id(url: str) -> str:
    # VT uses base64url of the URL without padding
    import base64
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def _heuristic_score(url: str) -> VTResult:
    """Offline reputation proxy used when no API key is configured."""
    lower = url.lower()
    mal = 0
    sus = 0
    if re.search(r"\.(xyz|top|club|tk|gq|ml|cf|ga|pw|cc)(/|$)", lower):
        mal += 2
    if re.search(r"(login|secure|verify|account|auth|update|confirm)[-.]?", lower):
        sus += 2
    if re.search(r"(paypa1|micros0ft|app1e|g00gle|amaz0n|bank-secure)", lower):
        mal += 4
    if len(url) > 90:
        sus += 1
    if url.count(".") > 4:
        sus += 1
    total = mal + sus
    rep = min(1.0, total / 8.0)
    return VTResult(
        url=url,
        malicious=mal,
        suspicious=sus,
        harmless=max(0, 10 - total),
        undetected=0,
        reputation=rep,
        source="heuristic",
    )


def _vt_get(path: str, api_key: str) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{VT_API_BASE}{path}",
        headers={"x-apikey": api_key, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def lookup_url(url: str, force: bool = False) -> VTResult:
    """
    Synchronous lookup. Results are cached in-process for _CACHE_TTL seconds.
    """
    url = (url or "").strip()
    if not url:
        return VTResult(url="", error="empty url")

    cache_key = hashlib.sha256(url.encode()).hexdigest()
    if not force and cache_key in _CACHE:
        hit = _CACHE[cache_key]
        # shallow copy flag
        return VTResult(**{**hit.to_dict(), "cached": True})

    key = _api_key()
    if not key:
        result = _heuristic_score(url)
        _CACHE[cache_key] = result
        return result

    try:
        data = _vt_get(f"/urls/{_url_id(url)}", key)
        attrs = data.get("data", {}).get("attributes", {})
        stats = attrs.get("last_analysis_stats", {}) or {}
        mal = int(stats.get("malicious", 0))
        sus = int(stats.get("suspicious", 0))
        harm = int(stats.get("harmless", 0))
        und = int(stats.get("undetected", 0))
        denom = max(1, mal + sus + harm + und)
        rep = (mal * 1.0 + sus * 0.5) / denom
        result = VTResult(
            url=url,
            malicious=mal,
            suspicious=sus,
            harmless=harm,
            undetected=und,
            reputation=min(1.0, rep),
            source="virustotal",
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # URL not known to VT yet — treat as neutral heuristic
            result = _heuristic_score(url)
            result.source = "virustotal_not_found"
        else:
            result = _heuristic_score(url)
            result.error = f"HTTP {e.code}"
            result.source = "heuristic_fallback"
    except Exception as e:
        result = _heuristic_score(url)
        result.error = str(e)[:120]
        result.source = "heuristic_fallback"

    _CACHE[cache_key] = result
    return result


def lookup_urls_async(urls: List[str], max_workers: int = 4) -> List[VTResult]:
    """
    Concurrent lookups via thread pool (async-style for the stdlib HTTP server).
    Deduplicates URLs first.
    """
    unique = list(dict.fromkeys(u.strip() for u in urls if u and u.strip()))
    if not unique:
        return []
    results: Dict[str, VTResult] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique))) as pool:
        futures = {pool.submit(lookup_url, u): u for u in unique}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                results[u] = fut.result()
            except Exception as e:
                results[u] = VTResult(url=u, error=str(e)[:120], source="error")
    return [results[u] for u in unique]


def aggregate_reputation(results: List[VTResult]) -> float:
    """Max reputation across URLs (worst URL dominates)."""
    if not results:
        return 0.0
    return max(r.reputation for r in results)


def clear_cache() -> None:
    _CACHE.clear()
