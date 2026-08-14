"""Anchor-conditioned decision surprise diagnostics.

This module scores semantic decisions extracted by the current-decision
alignment prompt, not raw code spans. It is diagnostic-only: it never decides
retry policy by itself.

For each aligned current policy, it measures:

    P(current_policy | task + structured anchor context)
    -----------------------------------------------------
                  P(current_policy | task)

If adding the anchor context suppresses the current policy, the decision is
poorly supported by the trusted evidence anchor.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from da_agent.utils.probability_contrast import (
    DEFAULT_MODEL_PATH,
    ContinuationScore,
    ContinuationScorer,
    HFContinuationScorer,
    compact_policy_text,
    compact_text,
    extract_anchor_alignment,
    extract_alignment_coverage,
    alignment_check_is_same_anchor_contrast,
)


@dataclass
class AnchorDecisionSurpriseConfig:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    enabled: bool = True
    max_probes: int = 8
    context_char_limit: int = 6000
    warning_surprise_threshold: float = 0.10
    retry_surprise_threshold: float = 0.50


@dataclass
class AnchorDecisionProbe:
    name: str
    anchor_id: str
    decision_type: str
    decision_point: str
    status: str
    expected_policy: str
    current_policy: str
    reason: str = ""
    source_code: str = ""


@dataclass
class AnchorDecisionProbeResult:
    name: str
    anchor_id: str
    decision_type: str
    decision_point: str
    status: str
    expected_policy: str
    current_policy: str
    reason: str = ""
    source_code: str = ""
    prior_avg_logp: Optional[float] = None
    conditioned_avg_logp: Optional[float] = None
    delta_cond_minus_prior: Optional[float] = None
    ratio_p_cond_over_prior: Optional[float] = None
    anchor_surprise_score: Optional[float] = None
    n_tokens: int = 0
    support_status: str = "no_signal"
    error: str = ""


@dataclass
class AnchorDecisionSurpriseResult:
    enabled: bool
    model: str
    available: bool
    diagnostic_only: bool = True
    affects_objective_confidence: bool = False
    risk_status: str = "no_signal"
    max_anchor_surprise_score: float = 0.0
    min_ratio: Optional[float] = None
    strongest_delta: Optional[float] = None
    alignment_coverage: Dict[str, Any] = field(default_factory=dict)
    candidate_probe_count: int = 0
    selected_probe_count: int = 0
    signals: List[str] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def finite(value: Optional[float]) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def safe_exp(value: Optional[float]) -> Optional[float]:
    if value is None or not finite(value):
        return None
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf") if value > 0 else 0.0


def _policy_is_missing(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    return lower in {
        "missing",
        "missing from code",
        "not visible",
        "not visible from code",
        "not visible from code/observation",
        "unclear",
    }


def _probe_prefix(*, task_text: str, anchor_context: str) -> str:
    task_part = f"# TASK\n{task_text.strip()}\n\n" if task_text.strip() else ""
    return (
        task_part
        + "# STRUCTURED EVIDENCE ANCHOR\n"
        + anchor_context.strip()
        + "\n\n# CURRENT COMPUTATIONAL DECISION\nPolicy:"
    )


def _prior_prefix(task_text: str) -> str:
    task_part = f"# TASK\n{task_text.strip()}\n\n" if task_text.strip() else ""
    return task_part + "# CURRENT COMPUTATIONAL DECISION\nPolicy:"


def _anchor_context_for_probe(probe: AnchorDecisionProbe, limit: int) -> str:
    parts = [
        f"anchor_id: {probe.anchor_id}",
        f"decision_type: {probe.decision_type}",
        f"decision_point: {probe.decision_point}",
        f"expected_policy: {probe.expected_policy}",
    ]
    if probe.reason:
        parts.append(f"evidence_reason: {probe.reason}")
    return compact_text("\n".join(parts), limit)


def build_anchor_decision_surprise_probes(
    current_anchor_text: str,
    max_probes: int = 8,
) -> List[AnchorDecisionProbe]:
    """Build anchor-conditioned probes from LLM-produced ANCHOR_ALIGNMENT."""
    probes: List[AnchorDecisionProbe] = []
    seen: set[Tuple[str, str]] = set()
    for index, check in enumerate(extract_anchor_alignment(current_anchor_text), start=1):
        if len(probes) >= max_probes:
            break

        valid_same_anchor, reason_not_valid = alignment_check_is_same_anchor_contrast(check)
        if not valid_same_anchor:
            continue

        current_policy = compact_policy_text(
            check.get("current_policy")
            or check.get("actual_policy")
            or check.get("code_policy")
            or check.get("implemented_policy")
        )
        if _policy_is_missing(current_policy):
            continue

        expected_policy = compact_policy_text(
            check.get("expected_policy")
            or check.get("evidence_policy")
            or check.get("expected_decision")
        )
        if not expected_policy:
            continue

        anchor_id = compact_policy_text(check.get("anchor_id"), 80)
        decision_type = compact_policy_text(check.get("decision_type") or "decision", 80)
        decision_point = compact_policy_text(
            check.get("evidence_decision_point")
            or check.get("decision_point"),
            240,
        )
        if not anchor_id or not decision_point:
            continue

        key = (anchor_id.lower(), current_policy.lower())
        if key in seen:
            continue
        seen.add(key)

        name_key = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            f"{decision_type}_{anchor_id}",
        ).strip("_").lower()
        probes.append(
            AnchorDecisionProbe(
                name=f"anchor_decision_surprise_{name_key or index}",
                anchor_id=anchor_id,
                decision_type=decision_type,
                decision_point=decision_point,
                status=compact_policy_text(check.get("status") or "unknown", 40),
                expected_policy=expected_policy,
                current_policy=current_policy,
                reason=compact_policy_text(
                    check.get("reason")
                    or check.get("source_observation")
                    or reason_not_valid,
                    500,
                ),
                source_code=compact_policy_text(
                    check.get("source_code")
                    or check.get("code_reference"),
                    500,
                ),
            )
        )
    return probes


class AnchorDecisionSurpriseEngine:
    def __init__(
        self,
        config: Optional[AnchorDecisionSurpriseConfig] = None,
        scorer: Optional[ContinuationScorer] = None,
    ) -> None:
        env_enabled = os.getenv("ACID_ANCHOR_DECISION_ES_DIAGNOSTIC", "")
        default_enabled = env_enabled.lower() not in {"0", "false", "no"} if env_enabled else True
        self.config = config or AnchorDecisionSurpriseConfig(
            model_path=os.getenv("ACID_ANCHOR_DECISION_ES_MODEL", DEFAULT_MODEL_PATH),
            device=os.getenv("ACID_ANCHOR_DECISION_ES_DEVICE", "auto"),
            enabled=default_enabled,
        )
        self.scorer = scorer
        self._score_cache: Dict[Tuple[str, str], ContinuationScore] = {}
        self._unavailable_error = ""

    def _get_scorer(self) -> Optional[ContinuationScorer]:
        if not self.config.enabled:
            self._unavailable_error = "anchor decision surprise disabled"
            return None
        if self.scorer is not None:
            return self.scorer
        try:
            self.scorer = HFContinuationScorer(
                model_path=self.config.model_path,
                device=self.config.device,
            )
            return self.scorer
        except Exception as exc:
            self._unavailable_error = str(exc)
            return None

    def _score(self, prefix: str, continuation: str) -> ContinuationScore:
        key = (prefix, continuation)
        if key not in self._score_cache:
            scorer = self._get_scorer()
            if scorer is None:
                raise RuntimeError(
                    self._unavailable_error
                    or "anchor decision surprise scorer unavailable"
                )
            self._score_cache[key] = scorer.score(prefix, continuation)
        return self._score_cache[key]

    def _evaluate_probe(
        self,
        *,
        probe: AnchorDecisionProbe,
        task_text: str,
    ) -> AnchorDecisionProbeResult:
        result = AnchorDecisionProbeResult(**asdict(probe))
        anchor_context = _anchor_context_for_probe(
            probe,
            self.config.context_char_limit,
        )
        cond_prefix = _probe_prefix(
            task_text=task_text,
            anchor_context=anchor_context,
        )
        prior_prefix = _prior_prefix(task_text)
        continuation = " " + probe.current_policy

        try:
            conditioned = self._score(cond_prefix, continuation)
            prior = self._score(prior_prefix, continuation)
            delta = None
            ratio = None
            surprise = None
            if finite(conditioned.avg_logp) and finite(prior.avg_logp):
                delta = conditioned.avg_logp - prior.avg_logp
                ratio = safe_exp(delta)
                surprise = max(0.0, -delta)

            result.prior_avg_logp = (
                round(prior.avg_logp, 4) if finite(prior.avg_logp) else None
            )
            result.conditioned_avg_logp = (
                round(conditioned.avg_logp, 4)
                if finite(conditioned.avg_logp)
                else None
            )
            result.delta_cond_minus_prior = round(delta, 4) if delta is not None else None
            result.ratio_p_cond_over_prior = round(ratio, 4) if ratio is not None else None
            result.anchor_surprise_score = round(surprise, 4) if surprise is not None else None
            result.n_tokens = conditioned.n_tokens
            if surprise is None:
                result.support_status = "no_signal"
            elif surprise > self.config.retry_surprise_threshold:
                result.support_status = "unsupported"
            elif surprise > self.config.warning_surprise_threshold:
                result.support_status = "weak_support"
            else:
                result.support_status = "supported"
            return result
        except Exception as exc:
            result.error = str(exc)
            result.support_status = "unavailable"
            return result

    def evaluate(
        self,
        *,
        task_text: str,
        current_anchor_text: str,
    ) -> AnchorDecisionSurpriseResult:
        result = AnchorDecisionSurpriseResult(
            enabled=bool(self.config.enabled),
            model=self.config.model_path,
            available=False,
        )
        result.alignment_coverage = extract_alignment_coverage(current_anchor_text)

        if not self.config.enabled:
            result.risk_status = "unavailable"
            result.signals.append("anchor decision surprise disabled")
            return result
        if not (current_anchor_text or "").strip():
            result.signals.append("no current anchor alignment available")
            return result

        probes = build_anchor_decision_surprise_probes(
            current_anchor_text,
            max_probes=self.config.max_probes,
        )
        result.candidate_probe_count = len(probes)
        result.selected_probe_count = len(probes)
        if not probes:
            result.signals.append(
                "no same-anchor current decision probes available for anchor decision surprise"
            )
            return result

        if self._get_scorer() is None:
            result.risk_status = "unavailable"
            result.signals.append(
                f"anchor decision surprise unavailable: {self._unavailable_error}"
            )
            result.probes = [asdict(probe) for probe in probes]
            return result

        probe_results = [
            self._evaluate_probe(probe=probe, task_text=task_text)
            for probe in probes
        ]
        scored = [probe for probe in probe_results if not probe.error]
        result.probes = [
            asdict(probe)
            for probe in sorted(
                probe_results,
                key=lambda item: item.anchor_surprise_score or 0.0,
                reverse=True,
            )
        ]
        if not scored:
            result.risk_status = "unavailable"
            result.signals.append(
                "anchor decision surprise unavailable: no probes scored successfully"
            )
            return result

        result.available = True
        scores = [
            probe.anchor_surprise_score or 0.0
            for probe in scored
            if probe.anchor_surprise_score is not None
        ]
        ratios = [
            probe.ratio_p_cond_over_prior
            for probe in scored
            if probe.ratio_p_cond_over_prior is not None
        ]
        deltas = [
            probe.delta_cond_minus_prior
            for probe in scored
            if probe.delta_cond_minus_prior is not None
        ]
        result.max_anchor_surprise_score = round(max(scores or [0.0]), 4)
        result.min_ratio = round(min(ratios), 4) if ratios else None
        result.strongest_delta = round(min(deltas), 4) if deltas else None

        unsupported = [
            probe for probe in scored
            if probe.support_status == "unsupported"
        ]
        weak = [
            probe for probe in scored
            if probe.support_status == "weak_support"
        ]
        if unsupported:
            result.risk_status = "diagnostic_retry"
            result.signals.append(
                "anchor decision surprise diagnostic found unsupported current policies"
            )
        elif weak:
            result.risk_status = "diagnostic_warning"
            result.signals.append(
                "anchor decision surprise diagnostic found weakly supported current policies"
            )
        else:
            result.risk_status = "pass"
            result.signals.append(
                "anchor decision surprise found no unsupported current policy"
            )
        return result


def evaluate_anchor_decision_surprise(
    *,
    task_text: str,
    current_anchor_text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    enabled: bool = True,
    max_probes: int = 8,
    scorer: Optional[ContinuationScorer] = None,
) -> Dict[str, Any]:
    config = AnchorDecisionSurpriseConfig(
        model_path=model_path,
        device=device,
        enabled=enabled,
        max_probes=max_probes,
    )
    engine = AnchorDecisionSurpriseEngine(config=config, scorer=scorer)
    return engine.evaluate(
        task_text=task_text,
        current_anchor_text=current_anchor_text,
    ).to_dict()


__all__ = [
    "AnchorDecisionSurpriseConfig",
    "AnchorDecisionSurpriseEngine",
    "AnchorDecisionSurpriseResult",
    "evaluate_anchor_decision_surprise",
]
