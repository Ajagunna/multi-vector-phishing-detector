"""
Feature extraction for multi-vector phishing detection.

Implements text sanitation, URL entropy, metadata derivation, VirusTotal
reputation enrichment, and attachment-sandbox signals.

MeAJOR offline sheets provide text + label; live IMAP / API paths supply
the full multi-feature schema (headers, attachments, VT, sandbox).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

# Optional integrations (safe if missing during early import)
try:
    from src.integrations.virustotal import (
        lookup_urls_async,
        aggregate_reputation,
        VTResult,
    )
except Exception:  # pragma: no cover
    lookup_urls_async = None  # type: ignore
    aggregate_reputation = None  # type: ignore
    VTResult = None  # type: ignore

try:
    from src.integrations.attachment_sandbox import (
        analyse_attachment,
        analyse_attachments,
        aggregate_attachment_score,
        filename_list_from_text,
        SandboxVerdict,
    )
except Exception:  # pragma: no cover
    analyse_attachment = None  # type: ignore
    analyse_attachments = None  # type: ignore
    aggregate_attachment_score = None  # type: ignore
    filename_list_from_text = None  # type: ignore
    SandboxVerdict = None  # type: ignore


# Model still consumes a 6-d vector (matches trained checkpoint).
# Richer schema is returned alongside for audit / future retraining.
NUM_MODEL_METADATA = 6
FULL_FEATURE_KEYS = [
    "url_count",
    "url_length_avg",
    "url_subdom_avg",
    "has_attachments",
    "domain_age_proxy",
    "url_entropy",
    "vt_reputation",
    "attachment_sandbox_score",
    "urgency_score",
    "credential_score",
    "spf_fail_proxy",
    "dkim_fail_proxy",
]


def unicode_normalize(text: str) -> str:
    """NFKC normalisation + strip non-printables / zero-width chars (Section 3.7.1)."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_html(text: str) -> str:
    """Remove HTML tags / scripts."""
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def extract_urls(text: str) -> List[str]:
    """Extract URLs (simple regex; MeAJOR already masks many as [URL])."""
    pattern = (
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        r"|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
    )
    return re.findall(pattern, text, flags=re.IGNORECASE)


def shannon_entropy(s: str) -> float:
    """Character-level Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def url_features(urls: List[str]) -> Dict[str, float]:
    if not urls:
        return {
            "url_count": 0.0,
            "url_length_avg": 0.0,
            "url_length_max": 0.0,
            "url_subdom_avg": 0.0,
            "url_entropy": 0.0,
        }
    lengths = [len(u) for u in urls]
    subdoms = []
    for u in urls:
        host = re.sub(r"^https?://", "", u).split("/")[0]
        parts = host.split(".")
        subdoms.append(max(0, len(parts) - 2))
    entropies = [shannon_entropy(u) for u in urls]
    return {
        "url_count": float(len(urls)),
        "url_length_avg": sum(lengths) / len(lengths),
        "url_length_max": float(max(lengths)),
        "url_subdom_avg": sum(subdoms) / len(subdoms),
        "url_entropy": sum(entropies) / len(entropies),
    }


def detect_urgency_and_credentials(text: str) -> Dict[str, float]:
    """Lightweight lexical signals that complement the transformer."""
    lower = text.lower()
    urgency_words = [
        "urgent", "immediately", "asap", "action required", "verify", "confirm",
        "suspend", "locked", "expire", "deadline", "within 24", "account will",
    ]
    cred_words = [
        "password", "login", "sign in", "credential", "otp", "verification code",
        "bank", "account", "ssn", "credit card", "wire transfer", "click here",
    ]
    urgency = sum(1 for w in urgency_words if w in lower)
    creds = sum(1 for w in cred_words if w in lower)
    return {"urgency_score": float(urgency), "credential_score": float(creds)}


def has_attachments_heuristic(text: str) -> float:
    """Heuristic from body mentions (real attachment info not in simplified MeAJOR)."""
    lower = text.lower()
    strong = [
        "please find attached", "see attached", ".pdf", ".doc", ".docx",
        ".xls", ".xlsx", ".zip", "attachment:",
    ]
    if any(i in lower for i in strong):
        return 1.0
    if "attachment" in lower and "attached are the" not in lower:
        return 1.0
    return 0.0


def domain_age_proxy(text: str) -> float:
    """
    Proxy for domain age: presence of newly-registered looking domains or
    brand-impersonation patterns. Real WHOIS is used only in live IMAP mode.
    Returns a normalised value in [0, 1] where low = suspicious.
    """
    lower = text.lower()
    suspicious = [".xyz", ".top", ".club", ".info", "secure-", "login-", "verify-", "account-"]
    score = 0.0
    for s in suspicious:
        if s in lower:
            score += 0.25
    return max(0.0, 1.0 - min(score, 1.0))


def spf_dkim_proxy(headers: str = "", text: str = "") -> Dict[str, float]:
    """
    Parse Authentication-Results style headers when present; otherwise
    return neutral proxies. Used in the full multi-feature schema.
    """
    blob = f"{headers}\n{text}".lower()
    spf_fail = 1.0 if "spf=fail" in blob or "spf=softfail" in blob else 0.0
    dkim_fail = 1.0 if "dkim=fail" in blob else 0.0
    if "spf=pass" in blob:
        spf_fail = 0.0
    if "dkim=pass" in blob:
        dkim_fail = 0.0
    return {"spf_fail_proxy": spf_fail, "dkim_fail_proxy": dkim_fail}


def extract_metadata_vector(text: str) -> List[float]:
    """
    Build the 6-dimensional metadata vector used by the *trained* model:
    [url_count, url_length_avg, url_subdom_avg, has_attachments, domain_age_proxy, url_entropy]
    """
    clean = unicode_normalize(strip_html(text))
    urls = extract_urls(clean)
    uf = url_features(urls)
    return [
        uf["url_count"],
        uf["url_length_avg"],
        uf["url_subdom_avg"],
        has_attachments_heuristic(clean),
        domain_age_proxy(clean),
        uf["url_entropy"],
    ]


def extract_full_features(
    text: str,
    headers: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    enable_vt: bool = True,
    enable_sandbox: bool = True,
) -> Dict[str, Any]:
    """
    Full multi-feature schema for audit, dashboard enrichment, and future
    MeAJOR retraining (beyond text+label).

    Returns:
      metadata_vector  — 6-d list for the current model
      features         — named full schema
      vt_results       — per-URL VirusTotal (or heuristic) results
      sandbox_verdicts — attachment sandbox verdicts
    """
    clean = unicode_normalize(strip_html(text))
    urls = extract_urls(clean)
    uf = url_features(urls)
    lex = detect_urgency_and_credentials(clean)
    auth = spf_dkim_proxy(headers, clean)
    has_att = has_attachments_heuristic(clean)

    vt_results: List[Any] = []
    vt_rep = 0.0
    if enable_vt and urls and lookup_urls_async is not None:
        # limit concurrent lookups for demo responsiveness
        vt_results = lookup_urls_async(urls[:8], max_workers=4)
        if aggregate_reputation is not None:
            vt_rep = aggregate_reputation(vt_results)

    sandbox_verdicts: List[Any] = []
    att_score = 0.0
    if enable_sandbox:
        items = list(attachments or [])
        if not items and filename_list_from_text is not None:
            # synthesise soft attachment infos from body mentions
            for fn in filename_list_from_text(clean)[:5]:
                items.append({"filename": fn, "content": b"", "content_type": ""})
        if items and analyse_attachments is not None:
            sandbox_verdicts = analyse_attachments(items)
            if aggregate_attachment_score is not None:
                att_score = aggregate_attachment_score(sandbox_verdicts)
            if sandbox_verdicts:
                has_att = 1.0

    meta6 = [
        uf["url_count"],
        uf["url_length_avg"],
        uf["url_subdom_avg"],
        has_att,
        domain_age_proxy(clean),
        uf["url_entropy"],
    ]

    features = {
        "url_count": uf["url_count"],
        "url_length_avg": uf["url_length_avg"],
        "url_length_max": uf["url_length_max"],
        "url_subdom_avg": uf["url_subdom_avg"],
        "url_entropy": uf["url_entropy"],
        "has_attachments": has_att,
        "domain_age_proxy": domain_age_proxy(clean),
        "vt_reputation": vt_rep,
        "attachment_sandbox_score": att_score,
        "urgency_score": lex["urgency_score"],
        "credential_score": lex["credential_score"],
        "spf_fail_proxy": auth["spf_fail_proxy"],
        "dkim_fail_proxy": auth["dkim_fail_proxy"],
    }

    return {
        "metadata_vector": meta6,
        "features": features,
        "vt_results": [r.to_dict() if hasattr(r, "to_dict") else r for r in vt_results],
        "sandbox_verdicts": [
            v.to_dict() if hasattr(v, "to_dict") else v for v in sandbox_verdicts
        ],
        "urls": urls,
    }


def prepare_text_for_model(text: str, max_chars: int = 2000) -> str:
    """Sanitise and truncate text for the encoder."""
    t = unicode_normalize(strip_html(str(text)))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def generate_reason_codes(
    text: str,
    metadata: List[float],
    phishing_prob: float,
    threshold: float = 0.5,
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Human-readable reason codes for the dashboard (Section 4.5 / 5.4)."""
    codes: List[str] = []
    if phishing_prob < threshold:
        return ["text_legitimate"]

    url_count, url_len_avg, url_sub, has_att, dom_age, url_ent = (
        metadata + [0.0] * 6
    )[:6]
    if url_count >= 2:
        codes.append("multiple_urls")
    if url_ent > 4.0:
        codes.append("url_entropy_high")
    if url_len_avg > 60:
        codes.append("long_urls")
    if has_att > 0.5:
        codes.append("attachment_mentioned")
    if dom_age < 0.5:
        codes.append("suspicious_domain_patterns")

    lower = text.lower()
    if any(w in lower for w in ["password", "login", "verify your", "account suspended"]):
        codes.append("credential_request")
    if any(w in lower for w in ["urgent", "immediately", "within 24 hours", "action required"]):
        codes.append("urgency_language")
    if phishing_prob > 0.85:
        codes.append("high_confidence_transformer")

    extra = extra or {}
    if float(extra.get("vt_reputation") or 0) >= 0.4:
        codes.append("virustotal_reputation_high")
    if float(extra.get("attachment_sandbox_score") or 0) >= 0.5:
        codes.append("attachment_sandbox_risk")
    if float(extra.get("spf_fail_proxy") or 0) > 0.5:
        codes.append("spf_fail")
    if float(extra.get("dkim_fail_proxy") or 0) > 0.5:
        codes.append("dkim_fail")

    if not codes:
        codes.append("model_signal")
    return codes


def simple_tokenize(text: str, max_length: int = 128) -> Dict[str, List[int]]:
    """
    Character / byte-level surrogate tokenizer matching the training pipeline.
    Maps printable ASCII + a few specials into a small vocab.
    """
    t = prepare_text_for_model(text)
    # 0 = pad, 1 = unk, 2.. = printable
    ids: List[int] = []
    for ch in t[: max_length - 1]:
        o = ord(ch)
        if 32 <= o < 127:
            ids.append(o - 30)  # 2..98
        else:
            ids.append(1)
    attn = [1] * len(ids)
    while len(ids) < max_length:
        ids.append(0)
        attn.append(0)
    return {"input_ids": ids, "attention_mask": attn}
