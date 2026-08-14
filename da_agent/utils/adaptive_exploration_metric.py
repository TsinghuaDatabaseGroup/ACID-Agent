"""Adaptive exploration metrics for ACID exploration rounds.

This module keeps only the round-level novelty signal:

    P(current_observation | previous_observation) / P(current_observation)

For a new exploration observation, it compares against all prior exploration
observations in the same exploration phase and returns the most predictable
match. A high PMI/token means the new observation is largely redundant with a
previous one, so exploration can stop early after repeated redundancy.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger("da_agent")

MODEL_PATH = (
    os.environ.get("ADAPTIVE_EXPLORATION_MODEL_PATH")
    or os.environ.get("QWEN_SURPRISE_MODEL_PATH")
    or "Qwen/Qwen3-0.6B-Base"
)
MAX_PREFIX_TOKENS = int(
    os.environ.get("ADAPTIVE_EXPLORATION_MAX_PREFIX_TOKENS")
    or os.environ.get("SURPRISE_MAX_PREFIX_TOKENS")
    or "1024"
)
MAX_TARGET_TOKENS = int(
    os.environ.get("ADAPTIVE_EXPLORATION_MAX_TARGET_TOKENS")
    or os.environ.get("SURPRISE_MAX_TARGET_TOKENS")
    or "256"
)

_tok = None
_model = None
_device = None
_torch = None


def _lazy_load() -> None:
    global _tok, _model, _device, _torch
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _torch = torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[ADAPTIVE-EXPLORATION] loading {MODEL_PATH} on {_device} ...")
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(_device).eval()
    logger.info("[ADAPTIVE-EXPLORATION] model ready.")


def _truncate_prefix(text: str) -> str:
    ids = _tok(text or "", add_special_tokens=False).input_ids
    if len(ids) <= MAX_PREFIX_TOKENS:
        return text or ""
    return _tok.decode(ids[-MAX_PREFIX_TOKENS:])


def _score_span(prefix_text: str, scored_text: str):
    """Return (sum_logprob, token_count) for scored_text conditioned on prefix_text."""
    _lazy_load()
    s_ids = _tok(scored_text, add_special_tokens=False).input_ids[:MAX_TARGET_TOKENS]
    if not s_ids:
        return 0.0, 0

    p_ids = _tok(prefix_text, add_special_tokens=False).input_ids
    full = p_ids + s_ids
    input_ids = _torch.tensor([full], device=_device)

    with _torch.no_grad():
        logits = _model(input_ids).logits[0].float()
        logprobs = _torch.log_softmax(logits, dim=-1)

    start = len(p_ids)
    total = 0.0
    for i in range(len(s_ids)):
        pos = start + i
        total += logprobs[pos - 1, full[pos]].item()
    return total, len(s_ids)


def compute_round_novelty(prior_obs_list: List[str], curr_obs: str) -> Optional[Dict[str, Any]]:
    """Measure whether the current exploration observation is redundant.

    Args:
        prior_obs_list: Prior exploration observations from the current phase.
        curr_obs: Current exploration observation.

    Returns:
        Dict with n, PMI, PMI_per_tok, S_COM, and match_round, or None if the
        signal cannot be computed.
    """
    try:
        current = (curr_obs or "").strip()
        if not current or not prior_obs_list:
            return None

        _lazy_load()
        scored = current + "</observation>"
        logp_marg, n_tokens = _score_span("<observation>", scored)
        if n_tokens == 0:
            return None

        best = None
        for idx, previous in enumerate(prior_obs_list):
            previous = (previous or "").strip()
            if not previous:
                continue
            previous = _truncate_prefix(previous)
            cond_prefix = f"<observation>{previous}</observation>\n<observation>"
            logp_cond, _ = _score_span(cond_prefix, scored)
            pmi_per_tok = (logp_cond - logp_marg) / n_tokens
            if best is None or pmi_per_tok > best[0]:
                best = (pmi_per_tok, idx)

        if best is None:
            return None

        pmi_per_tok, match_round = best
        return {
            "n": n_tokens,
            "PMI": round(pmi_per_tok * n_tokens, 3),
            "PMI_per_tok": round(pmi_per_tok, 4),
            "S_COM": round(math.exp(-pmi_per_tok), 4),
            "match_round": match_round,
        }
    except Exception as exc:
        logger.warning(f"[ROUND-NOV] compute failed: {exc}")
        return None


__all__ = ["compute_round_novelty"]
