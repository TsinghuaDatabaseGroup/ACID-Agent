"""Probability contrast probes for evidence-conditioned retry decisions.

This module is intentionally independent from ``acid_agent.py``.  It consumes
LLM-produced anchor alignment and builds small two-option probes:

    current decision vs evidence-supported alternative

Then a causal language model scores each option as a continuation under
``task + evidence + probe_prefix``.  If the current option is much less likely
than the alternative, the result is classified as ``warning`` or ``retry``.

The module does not infer semantic conflicts with regex/keyword heuristics.  It
only measures relative support for explicit conflicts already reported in
``ANCHOR_ALIGNMENT``. Missing policies are intentionally skipped because they do
not provide a concrete current-vs-alternative pair.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from da_agent.utils.runtime_config import PROB_CONTRAST_MODEL_PATH

DEFAULT_MODEL_PATH = PROB_CONTRAST_MODEL_PATH
ANCHOR_ALIGNMENT_MARKER = "ANCHOR_ALIGNMENT"


@dataclass
class ContinuationScore:
    total_logp: float
    avg_logp: float
    n_tokens: int


@dataclass
class ContrastProbe:
    name: str
    source: str
    prefix: str
    current: str
    alternative: str
    code_line: str = ""
    evidence: str = ""
    anchor_id: str = ""
    decision_type: str = ""
    decision_point: str = ""

@dataclass
class ContrastProbeResult:
    name: str
    source: str
    prefix: str
    current: str
    alternative: str
    code_line: str = ""
    evidence: str = ""
    anchor_id: str = ""
    decision_type: str = ""
    decision_point: str = ""
    current_avg_logp: Optional[float] = None
    alternative_avg_logp: Optional[float] = None
    delta_current_minus_alternative: Optional[float] = None
    ratio_current_to_alternative: Optional[float] = None
    current_n_tokens: Optional[int] = None
    alternative_n_tokens: Optional[int] = None
    current_disfavored: bool = False
    retry_recommendation: str = ""
    risk_status: str = "pass"
    error: str = ""


@dataclass
class ProbabilityContrastConfig:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    enabled: bool = True
    max_probes: int = 4
    context_char_limit: int = 0
    retry_ratio_threshold: float = 0.25
    warning_ratio_threshold: float = 0.75
    retry_delta_threshold: float = math.log(0.25)
    warning_delta_threshold: float = math.log(0.75)


@dataclass
class ProbabilityContrastResult:
    enabled: bool
    model: str
    available: bool
    retry_recommendation: str = ""
    risk_status: str = "no_signal"
    min_ratio: Optional[float] = None
    strongest_delta: Optional[float] = None
    alignment_coverage: Dict[str, Any] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuationScorer(Protocol):
    def score(self, prefix: str, continuation: str) -> ContinuationScore:
        ...


class HFContinuationScorer:
    """Continuation log-probability scorer for HuggingFace causal LMs.

    The model is loaded lazily on first use.  This keeps importing the module
    cheap and lets the caller construct the engine even when transformers/torch
    are not installed in a local debugging environment.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._loaded = False
        self._torch = None
        self._F = None
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.device != "auto":
            model.to(self.device)
        model.eval()

        self._torch = torch
        self._F = F
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    def score(self, prefix: str, continuation: str) -> ContinuationScore:
        self._load()
        torch = self._torch
        F = self._F
        tokenizer = self._tokenizer
        model = self._model

        text = prefix + continuation
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        try:
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        except Exception:
            pass

        labels = input_ids.clone()
        prefix_len = prefix_ids.shape[1]
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels != -100
            if not bool(valid.any()):
                return ContinuationScore(total_logp=0.0, avg_logp=0.0, n_tokens=0)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            token_logps = token_logps.masked_select(valid)
            total = float(token_logps.sum().item())
            n_tokens = int(token_logps.numel())
            return ContinuationScore(total_logp=total, avg_logp=total / max(1, n_tokens), n_tokens=n_tokens)


def compact_text(text: str, limit: int = 6000) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    head_len = limit // 2
    tail_len = limit - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    return head + "\n... (middle truncated for probability contrast) ...\n" + tail


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def extract_json_payloads_after_marker(text: str, marker_text: str) -> List[Dict[str, Any]]:
    """Parse JSON objects after a marker, accepting fenced or raw objects."""
    payloads: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for marker in re.finditer(re.escape(marker_text), text or "", re.IGNORECASE):
        rest = (text or "")[marker.end():]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", rest, re.IGNORECASE | re.DOTALL)
        candidates: List[str] = []
        if fenced and fenced.start() < 800:
            candidates.append(fenced.group(1))
        brace_index = rest.find("{")
        if brace_index >= 0 and brace_index < 800:
            candidates.append(rest[brace_index:])
        for candidate in candidates:
            try:
                payload, _ = decoder.raw_decode(candidate.strip())
            except Exception:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
                break
    return payloads


def extract_anchor_alignment_payloads(current_anchor_text: str) -> List[Dict[str, Any]]:
    return extract_json_payloads_after_marker(current_anchor_text or "", ANCHOR_ALIGNMENT_MARKER)


def extract_anchor_alignment(current_anchor_text: str) -> List[Dict[str, Any]]:
    """Parse LLM-produced evidence-vs-code alignment checks.

    Newer current-decision extraction asks the LLM to compare structured
    evidence anchors against what the code actually implements.  Those checks
    are the most general source for probability contrast because the module no
    longer has to hard-code every possible decision type.
    """
    checks: List[Dict[str, Any]] = []
    for payload in extract_anchor_alignment_payloads(current_anchor_text):
        for item in as_list(payload.get("anchor_checks")):
            if isinstance(item, dict):
                checks.append(item)
    return checks


def extract_alignment_coverage(current_anchor_text: str) -> Dict[str, Any]:
    for payload in extract_anchor_alignment_payloads(current_anchor_text):
        coverage = payload.get("alignment_coverage")
        if isinstance(coverage, dict):
            return dict(coverage)
    return {}


def normalize_string_list(value: Any) -> List[str]:
    items: List[str] = []
    for item in as_list(value):
        text = str(item or "").strip()
        if text and text not in items:
            items.append(text)
    return items


def annotate_alignment_coverage(
    result: ProbabilityContrastResult,
    current_anchor_text: str,
) -> ProbabilityContrastResult:
    coverage = extract_alignment_coverage(current_anchor_text)
    if not coverage:
        if (current_anchor_text or "").strip():
            result.signals.append("alignment coverage missing from ANCHOR_ALIGNMENT output")
            if result.risk_status == "pass":
                result.risk_status = "warning"
                result.retry_recommendation = "warning"
        return result

    coverage = dict(coverage)
    unaddressed = normalize_string_list(coverage.get("unaddressed_anchor_ids"))
    missing = normalize_string_list(coverage.get("missing_current_policy_anchor_ids"))
    complete = bool(coverage.get("alignment_complete"))
    coverage["unaddressed_anchor_ids"] = unaddressed
    coverage["missing_current_policy_anchor_ids"] = missing
    result.alignment_coverage = coverage

    if not complete or unaddressed:
        result.signals.append(
            "alignment coverage incomplete: "
            f"unaddressed_anchor_ids={unaddressed}"
        )
        if result.risk_status == "pass":
            result.risk_status = "warning"
            result.retry_recommendation = "warning"
    if missing:
        result.signals.append(f"alignment coverage has missing current policies: {missing}")
    return result


def compact_policy_text(value: Any, limit: int = 0) -> str:
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value or "")
    return re.sub(r"\s+", " ", text).strip(" `")


def alignment_status_needs_contrast(status: Any) -> bool:
    return str(status or "").strip().lower() == "conflict"


def parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def alignment_check_is_same_anchor_contrast(check: Dict[str, Any]) -> Tuple[bool, str]:
    """Return whether an ANCHOR_ALIGNMENT item is safe to score.

    Probability contrast is meaningful only for two policies that answer the
    same evidence anchor. The LLM extraction prompt must explicitly certify
    this with same_anchor_policy=true and comparison_valid=true. Missing or
    cross-dimensional comparisons are skipped instead of scored.
    """
    anchor_id = compact_policy_text(check.get("anchor_id"), 80)
    if not anchor_id or anchor_id.lower().startswith("check_"):
        return False, "missing explicit evidence anchor_id"

    decision_type = compact_policy_text(check.get("decision_type"), 80)
    if not decision_type:
        return False, "missing decision_type"

    decision_point = compact_policy_text(
        check.get("evidence_decision_point")
        or check.get("decision_point"),
        240,
    )
    if not decision_point:
        return False, "missing evidence decision point"

    if not parse_bool_flag(check.get("same_anchor_policy")):
        return False, "same_anchor_policy is not true"
    if not parse_bool_flag(check.get("comparison_valid")):
        return False, "comparison_valid is not true"
    return True, ""


def alignment_probe_prefix(decision_type: str) -> str:
    readable_type = re.sub(r"[_-]+", " ", decision_type or "decision").strip()
    return (
        "Given the task and evidence, choose the safer computational policy "
        f"for {readable_type}.\nPolicy:"
    )


def build_alignment_contrast_probes(
    current_anchor_text: str,
    max_probes: int = 4,
) -> List[ContrastProbe]:
    probes: List[ContrastProbe] = []
    seen = set()
    for idx, check in enumerate(extract_anchor_alignment(current_anchor_text), start=1):
        if len(probes) >= max_probes:
            break
        status = str(check.get("status") or "").strip().lower()
        if not alignment_status_needs_contrast(status):
            continue
        valid_same_anchor, _ = alignment_check_is_same_anchor_contrast(check)
        if not valid_same_anchor:
            continue
        expected_policy = compact_policy_text(
            check.get("expected_policy")
            or check.get("evidence_policy")
            or check.get("expected_decision")
        )
        current_policy = compact_policy_text(
            check.get("current_policy")
            or check.get("actual_policy")
            or check.get("code_policy")
            or check.get("implemented_policy")
        )
        if not expected_policy or not current_policy:
            continue
        if expected_policy.lower() == current_policy.lower():
            continue

        decision_type = compact_policy_text(check.get("decision_type") or "decision", 80)
        anchor_id = compact_policy_text(check.get("anchor_id") or f"check_{idx}", 80)
        decision_point = compact_policy_text(
            check.get("evidence_decision_point")
            or check.get("decision_point"),
            240,
        )
        name_key = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{decision_type}_{anchor_id}").strip("_").lower()
        key = (name_key, current_policy.lower(), expected_policy.lower())
        if key in seen:
            continue
        seen.add(key)
        evidence_text = compact_policy_text(
            check.get("reason") or check.get("evidence_decision_point") or check.get("source_observation"),
            500,
        )
        probes.append(ContrastProbe(
            name=f"llm_anchor_alignment_{name_key or idx}",
            source="llm_anchor_alignment",
            prefix=alignment_probe_prefix(decision_type),
            current=f" {current_policy}",
            alternative=f" {expected_policy}",
            code_line=compact_policy_text(check.get("source_code") or check.get("code_reference"), 500),
            evidence=evidence_text,
            anchor_id=anchor_id,
            decision_type=decision_type,
            decision_point=decision_point,
        ))
    return probes


def build_probability_contrast_probes(
    code: str = "",
    observation: str = "",
    evidence_text: str = "",
    current_anchor_text: str = "",
    max_probes: int = 4,
) -> List[ContrastProbe]:
    """Build probes only from LLM-produced ANCHOR_ALIGNMENT checks.

    Legacy regex/keyword heuristics are intentionally removed.  If the LLM does
    not produce an explicit conflict check with both policies, no probability-
    ratio probe is constructed.
    """
    return build_alignment_contrast_probes(current_anchor_text, max_probes=max_probes)


class ProbabilityContrastEngine:
    """Build and score probability contrast probes."""

    def __init__(
        self,
        config: Optional[ProbabilityContrastConfig] = None,
        scorer: Optional[ContinuationScorer] = None,
    ):
        self.config = config or ProbabilityContrastConfig(
            model_path=os.getenv("ACID_PROB_CONTRAST_MODEL", DEFAULT_MODEL_PATH),
            device=os.getenv("ACID_PROB_CONTRAST_DEVICE", "auto"),
        )
        self.scorer = scorer
        self._score_cache: Dict[Tuple[str, str], ContinuationScore] = {}
        self._unavailable_error: str = ""

    def _get_scorer(self) -> Optional[ContinuationScorer]:
        if not self.config.enabled:
            self._unavailable_error = "probability contrast disabled"
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
                raise RuntimeError(self._unavailable_error or "probability scorer unavailable")
            self._score_cache[key] = scorer.score(prefix, continuation)
        return self._score_cache[key]

    def score_probes(
        self,
        probes: Sequence[ContrastProbe],
        task_text: str,
        evidence_text: str,
    ) -> ProbabilityContrastResult:
        result = ProbabilityContrastResult(
            enabled=bool(self.config.enabled),
            model=self.config.model_path,
            available=False,
        )
        if not probes:
            result.signals.append("no clear current/alternative contrast probe could be constructed")
            return result
        if not self.config.enabled:
            result.risk_status = "unavailable"
            result.signals.append("probability contrast disabled; probes were constructed but not scored")
            result.probes = [asdict(probe) for probe in probes]
            return result

        scorer = self._get_scorer()
        if scorer is None:
            result.risk_status = "unavailable"
            result.signals.append(f"probability contrast unavailable: {self._unavailable_error}")
            result.probes = [asdict(probe) for probe in probes]
            return result

        result.available = True
        context = (
            f"# TASK\n{task_text or ''}\n\n"
            "# EVIDENCE\n"
            f"{compact_text(evidence_text or '', self.config.context_char_limit)}\n\n"
        )

        observed_ratios: List[float] = []
        observed_deltas: List[float] = []
        retry_ratios: List[float] = []
        warning_ratios: List[float] = []
        scored_count = 0
        for probe in probes:
            probe_result = ContrastProbeResult(**asdict(probe))
            try:
                current_score = self._score(context + probe.prefix, probe.current)
                alternative_score = self._score(context + probe.prefix, probe.alternative)
                delta = current_score.avg_logp - alternative_score.avg_logp
                try:
                    probe_ratio = math.exp(delta)
                except OverflowError:
                    probe_ratio = float("inf") if delta > 0 else 0.0

                observed_ratios.append(probe_ratio)
                observed_deltas.append(delta)
                retry_recommendation = ""
                current_disfavored = False
                if (
                    probe_ratio < self.config.retry_ratio_threshold
                    or delta < self.config.retry_delta_threshold
                ):
                    retry_recommendation = "retry"
                    current_disfavored = True
                    retry_ratios.append(probe_ratio)
                elif (
                    probe_ratio < self.config.warning_ratio_threshold
                    or delta < self.config.warning_delta_threshold
                ):
                    retry_recommendation = "warning"
                    current_disfavored = True
                    warning_ratios.append(probe_ratio)

                probe_result.current_avg_logp = round(current_score.avg_logp, 4)
                probe_result.alternative_avg_logp = round(alternative_score.avg_logp, 4)
                probe_result.delta_current_minus_alternative = round(delta, 4)
                probe_result.ratio_current_to_alternative = round(probe_ratio, 4)
                probe_result.current_n_tokens = current_score.n_tokens
                probe_result.alternative_n_tokens = alternative_score.n_tokens
                probe_result.current_disfavored = bool(current_disfavored)
                probe_result.retry_recommendation = retry_recommendation
                probe_result.risk_status = retry_recommendation or "pass"
                scored_count += 1

                if current_disfavored:
                    result.signals.append(
                        "probability contrast disfavors current decision relative to "
                        f"evidence-supported alternative: {probe.name} "
                        f"current={probe.current!r} alternative={probe.alternative!r} "
                        f"ratio={probe_ratio:.4f} delta={delta:.4f} "
                        f"recommendation={retry_recommendation}"
                    )
            except Exception as exc:
                probe_result.error = str(exc)
                result.signals.append(f"probability contrast probe failed: {probe.name}: {exc}")
            result.probes.append(asdict(probe_result))

        if scored_count == 0:
            result.available = False
            result.risk_status = "unavailable"
            result.signals.append(
                "probability contrast unavailable: no probes were scored successfully"
            )
            return result

        if observed_ratios:
            result.min_ratio = round(min(observed_ratios), 4)
            result.strongest_delta = round(min(observed_deltas), 4)

        if retry_ratios:
            result.risk_status = "retry"
            result.retry_recommendation = "retry"
        elif warning_ratios:
            result.risk_status = "warning"
            result.retry_recommendation = "warning"
        else:
            result.risk_status = "pass"
            result.signals.append("no probability contrast signal disfavored the current decision")
        return result

    def evaluate(
        self,
        *,
        task_text: str,
        evidence_text: str,
        code: str = "",
        observation: str = "",
        current_anchor_text: str = "",
        probes: Optional[Sequence[ContrastProbe]] = None,
    ) -> ProbabilityContrastResult:
        built_probes = list(probes) if probes is not None else build_probability_contrast_probes(
            code=code,
            observation=observation,
            evidence_text=evidence_text,
            current_anchor_text=current_anchor_text,
            max_probes=self.config.max_probes,
        )
        result = self.score_probes(built_probes, task_text=task_text, evidence_text=evidence_text)
        return annotate_alignment_coverage(result, current_anchor_text)
