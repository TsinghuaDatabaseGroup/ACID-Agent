"""Deterministic observation digests for prompt-facing context.

The full raw observation is still stored in logs/trajectory.  This helper only
builds a compact, structured view for LLM prompts so long stdout/stderr output
does not snowball through memory and review prompts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _compact_channel(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return "(empty)"
    if limit <= 0 or len(text) <= limit:
        return text

    head_len = limit // 2
    tail_len = limit - head_len
    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip()
    omitted = text[head_len: len(text) - tail_len]
    omitted_lines = len(omitted.splitlines())
    omitted_chars = len(omitted)
    return (
        "[head]\n"
        f"{head}\n"
        f"[... omitted {omitted_lines} lines / {omitted_chars} chars ...]\n"
        "[tail]\n"
        f"{tail}"
    )


def _has_structured_execution_meta(execution_meta: Optional[Dict[str, Any]]) -> bool:
    return isinstance(execution_meta, dict) and any(
        key in execution_meta for key in ("exit_code", "stdout", "stderr", "timed_out")
    )


def build_observation_digest(
    observation: str,
    execution_meta: Optional[Dict[str, Any]] = None,
    *,
    max_total_chars: int = 3000,
    max_stdout_chars: int = 2000,
    max_stderr_chars: int = 800,
) -> str:
    """Return a compact deterministic digest of an execution observation."""
    observation = _as_text(observation)
    meta = execution_meta or {}
    has_meta = _has_structured_execution_meta(meta)

    stdout = _as_text(meta.get("stdout")) if has_meta else observation
    stderr = _as_text(meta.get("stderr")) if has_meta else ""
    if has_meta and not stdout.strip() and not stderr.strip() and observation.strip():
        stdout = observation

    timed_out = bool(meta.get("timed_out"))
    action_type = _as_text(meta.get("action_type")).strip()
    exit_code = meta.get("exit_code") if has_meta else None

    lines = [
        "# OBSERVATION DIGEST",
        f"action_type: {action_type or 'unknown'}",
        f"exit_code: {exit_code if exit_code is not None else 'unknown'}",
        f"timed_out: {str(timed_out).lower()}",
        f"stdout_empty: {str(not bool(stdout.strip())).lower()}",
        f"stderr_empty: {str(not bool(stderr.strip())).lower()}",
        f"observation_chars: {len(observation.strip())}",
        f"stdout_chars: {len(stdout.strip())}",
        f"stderr_chars: {len(stderr.strip())}",
        "",
        "stdout_digest:",
        _compact_channel(stdout, max_stdout_chars),
    ]
    if stderr.strip():
        lines.extend([
            "",
            "stderr_digest:",
            _compact_channel(stderr, max_stderr_chars),
        ])

    digest = "\n".join(lines).strip()
    return _compact_channel(digest, max_total_chars)
