"""
Attachment sandbox verdict module.

Production path: submit file hashes / samples to an external sandbox
(e.g. Hybrid Analysis, Any.Run, or a private Cuckoo instance) when
ATTACHMENT_SANDBOX_URL + ATTACHMENT_SANDBOX_KEY are configured.

Prototype path: multi-signal heuristic that inspects filename, extension,
MIME hints, and common phishing attachment patterns. Returns a structured
verdict consumable by the detector and dashboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

DANGEROUS_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".jse",
    ".wsf", ".hta", ".msi", ".dll", ".com", ".pif", ".cpl",
    ".iso", ".img", ".lnk", ".reg",
}
DOUBLE_EXTENSION = re.compile(
    r"\.(pdf|doc|docx|xls|xlsx|jpg|png|txt)\.(exe|scr|bat|cmd|js|vbs|ps1)$",
    re.IGNORECASE,
)
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".iso"}
OFFICE_MACRO_HINTS = {".docm", ".xlsm", ".pptm", ".dotm"}


@dataclass
class AttachmentInfo:
    filename: str
    size: int = 0
    content_type: str = ""
    sha256: Optional[str] = None
    content_preview: bytes = b""


@dataclass
class SandboxVerdict:
    filename: str
    verdict: str = "unknown"  # clean | suspicious | malicious | unknown
    score: float = 0.0  # 0..1 maliciousness
    signals: List[str] = field(default_factory=list)
    source: str = "heuristic"
    external_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ext(name: str) -> str:
    name = name.lower().rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def heuristic_verdict(att: AttachmentInfo) -> SandboxVerdict:
    signals: List[str] = []
    score = 0.0
    name = att.filename or "unknown"

    if DOUBLE_EXTENSION.search(name):
        signals.append("double_extension")
        score += 0.55

    ext = _ext(name)
    if ext in DANGEROUS_EXTENSIONS:
        signals.append(f"dangerous_extension:{ext}")
        score += 0.7
    if ext in OFFICE_MACRO_HINTS:
        signals.append("macro_enabled_office")
        score += 0.35
    if ext in ARCHIVE_EXTS:
        signals.append("archive_attachment")
        score += 0.15

    lower = name.lower()
    for token in ("invoice", "payment", "urgent", "wire", "remittance", "statement", "scan"):
        if token in lower:
            signals.append(f"filename_lure:{token}")
            score += 0.12

    # Content-type mismatch (e.g. application/octet-stream named .pdf)
    ct = (att.content_type or "").lower()
    if ct and "octet-stream" in ct and ext in {".pdf", ".doc", ".docx"}:
        signals.append("content_type_mismatch")
        score += 0.2

    # Tiny executable-like payload
    if att.size and att.size < 512 and ext in DANGEROUS_EXTENSIONS:
        signals.append("tiny_binary")
        score += 0.15

    # Magic-byte peek
    preview = att.content_preview or b""
    if preview.startswith(b"MZ"):
        signals.append("pe_header")
        score += 0.5
    elif preview[:4] == b"%PDF" and ext in DANGEROUS_EXTENSIONS:
        signals.append("pdf_named_as_executable")
        score += 0.4

    score = min(1.0, score)
    if score >= 0.65:
        verdict = "malicious"
    elif score >= 0.3:
        verdict = "suspicious"
    else:
        verdict = "clean"

    return SandboxVerdict(
        filename=name,
        verdict=verdict,
        score=score,
        signals=signals,
        source="heuristic",
    )


def _external_submit(att: AttachmentInfo) -> Optional[SandboxVerdict]:
    """Optional call to a configured external sandbox HTTP endpoint."""
    base = os.environ.get("ATTACHMENT_SANDBOX_URL")
    key = os.environ.get("ATTACHMENT_SANDBOX_KEY", "")
    if not base:
        return None

    payload = {
        "filename": att.filename,
        "size": att.size,
        "content_type": att.content_type,
        "sha256": att.sha256,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/verdict",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}" if key else "",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return SandboxVerdict(
            filename=att.filename,
            verdict=str(body.get("verdict", "unknown")),
            score=float(body.get("score", 0.0)),
            signals=list(body.get("signals") or []),
            source="external_sandbox",
            external_id=body.get("id"),
        )
    except Exception as e:
        return SandboxVerdict(
            filename=att.filename,
            verdict="unknown",
            score=0.0,
            signals=[],
            source="external_error",
            error=str(e)[:120],
        )


def analyse_attachment(
    filename: str,
    content: bytes = b"",
    content_type: str = "",
    use_external: bool = True,
) -> SandboxVerdict:
    sha = hashlib.sha256(content).hexdigest() if content else None
    att = AttachmentInfo(
        filename=filename or "unknown",
        size=len(content) if content else 0,
        content_type=content_type,
        sha256=sha,
        content_preview=content[:64] if content else b"",
    )
    if use_external:
        external = _external_submit(att)
        if external is not None and external.source == "external_sandbox":
            return external
    return heuristic_verdict(att)


def analyse_attachments(items: List[Dict[str, Any]]) -> List[SandboxVerdict]:
    """
    items: list of dicts with keys filename, content (bytes|optional), content_type.
    """
    out: List[SandboxVerdict] = []
    for it in items:
        out.append(
            analyse_attachment(
                filename=it.get("filename") or "unknown",
                content=it.get("content") or b"",
                content_type=it.get("content_type") or "",
            )
        )
    return out


def aggregate_attachment_score(verdicts: List[SandboxVerdict]) -> float:
    if not verdicts:
        return 0.0
    return max(v.score for v in verdicts)


def filename_list_from_text(text: str) -> List[str]:
    """Extract likely attachment filenames mentioned in body text."""
    pattern = r"[\w.\-]+\.(?:pdf|docx?|xlsx?|zip|rar|exe|scr|js|vbs|html?|iso)"
    return list(dict.fromkeys(re.findall(pattern, text, flags=re.IGNORECASE)))
