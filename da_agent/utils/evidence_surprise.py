"""Evidence-conditioned Python span surprise signals.

This module implements the online evidence surprise probe used by ACID retry
review:

1. Only Python actions are scored.
2. Python code is split into statement spans using AST structure. If parsing
   fails, the signal is marked unavailable instead of guessing spans with
   string heuristics. When there are more spans than the scoring budget, an LLM
   selector chooses the spans to score.
3. Each selected span is scored by comparing how likely it is before and after
   adding exploration evidence:

   logP(span | task + evidence + code_prefix) - logP(span | task + code_prefix)

If evidence suppresses a span, the span surprise score increases. Retry
severity is decided by ``da_agent.review.retry_policy``.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from da_agent.utils.llm_utils import call_llm_with_usage
from da_agent.utils.probability_contrast import (
    DEFAULT_MODEL_PATH,
    ContinuationScore,
    ContinuationScorer,
    HFContinuationScorer,
)


@dataclass
class EvidenceSurpriseConfig:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    enabled: bool = True
    max_spans: int = 10
    context_char_limit: int = 0
    code_prefix_char_limit: int = 0
    warning_span_surprise_threshold: float = 0.10
    retry_span_surprise_threshold: float = 0.50
    span_selector_model: str = ""
    span_selector_max_tokens: int = 500


@dataclass
class DecisionSpan:
    text: str
    start: int
    end: int
    line_no: int
    category: str
    reason: str
    code_prefix: str


@dataclass
class EvidenceSurpriseSpanResult:
    line_no: int
    category: str
    reason: str
    span: str
    prior_avg_logp: Optional[float] = None
    conditioned_avg_logp: Optional[float] = None
    delta_cond_minus_prior: Optional[float] = None
    ratio_p_cond_over_prior: Optional[float] = None
    span_surprise_score: Optional[float] = None
    n_tokens: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSurpriseResult:
    enabled: bool
    model: str
    available: bool
    max_span_surprise_score: float = 0.0
    retry_recommendation: str = ""
    risk_status: str = "no_signal"
    span_selection_method: str = ""
    candidate_span_count: int = 0
    selected_span_count: int = 0
    signals: List[str] = field(default_factory=list)
    spans: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def finite(value: Optional[float]) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def safe_exp(value: Optional[float]) -> Optional[float]:
    if value is None or not finite(value):
        return None
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf") if value > 0 else 0.0


def compact_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    head_len = limit // 2
    tail_len = limit - head_len
    return text[:head_len] + "\n... (middle truncated) ...\n" + text[-tail_len:]


def split_semicolon_lines(code: str) -> str:
    if "\n" in code:
        return code
    if code.count(";") < 2:
        return code
    return "\n".join(part.strip() for part in code.split(";") if part.strip())


def line_offsets(code: str) -> List[int]:
    offsets = [0]
    cursor = 0
    for line in code.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    if not code.endswith("\n"):
        offsets.append(len(code))
    return offsets


def offset_for_position(offsets: Sequence[int], line_no: int, col: int) -> int:
    if line_no <= 0:
        return 0
    index = min(line_no - 1, len(offsets) - 1)
    return min(offsets[index] + col, offsets[-1])


def node_source_range(code: str, offsets: Sequence[int], node: ast.AST) -> Tuple[int, int]:
    start = offset_for_position(offsets, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))
    end_line = getattr(node, "end_lineno", getattr(node, "lineno", 1))
    end_col = getattr(node, "end_col_offset", getattr(node, "col_offset", 0))
    end = offset_for_position(offsets, end_line, end_col)
    if end <= start:
        segment = ast.get_source_segment(code, node) or ""
        end = start + len(segment)
    return start, min(end, len(code))


def ast_feature_counts(node: ast.AST) -> Dict[str, int]:
    features = {
        "comparison": 0,
        "boolean_logic": 0,
        "subscript": 0,
        "calculation": 0,
        "function_call": 0,
        "control_flow": 0,
        "comprehension": 0,
        "assignment": 0,
        "return": 0,
    }
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            features["comparison"] += 1
        elif isinstance(child, ast.BoolOp):
            features["boolean_logic"] += 1
        elif isinstance(child, ast.Subscript):
            features["subscript"] += 1
        elif isinstance(child, ast.BinOp):
            features["calculation"] += 1
        elif isinstance(child, ast.Call):
            features["function_call"] += 1
        elif isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            features["control_flow"] += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            features["comprehension"] += 1
        elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            features["assignment"] += 1
        elif isinstance(child, ast.Return):
            features["return"] += 1
    return features


def is_output_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if isinstance(func, ast.Name):
        return func.id in {"print", "display"}
    if isinstance(func, ast.Attribute):
        return func.attr in {"show", "head", "tail", "info", "describe"}
    return False


def assignment_value(node: ast.AST) -> Optional[ast.AST]:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def is_empty_container_initialization(value: Optional[ast.AST]) -> bool:
    if value is None:
        return False
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return len(value.elts) == 0
    if isinstance(value, ast.Dict):
        return len(value.keys) == 0
    if not isinstance(value, ast.Call):
        return False
    if value.args or value.keywords:
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id in {"list", "dict", "set", "tuple"}
    if isinstance(func, ast.Attribute):
        return func.attr == "DataFrame"
    return False


def skip_ast_statement(node: ast.AST) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass, ast.Break, ast.Continue)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return True
    if is_output_expr(node):
        return True
    if is_empty_container_initialization(assignment_value(node)):
        return True
    return False


def ast_statement_category(features: Dict[str, int], node: ast.AST) -> str:
    if features["control_flow"]:
        return "control_flow"
    if features["comparison"] and features["subscript"]:
        return "conditional_selection"
    if features["comparison"]:
        return "comparison"
    if features["comprehension"]:
        return "comprehension"
    if features["calculation"]:
        return "calculation"
    if features["subscript"]:
        return "selection"
    if features["function_call"]:
        return "function_call"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "assignment"
    if isinstance(node, ast.Return):
        return "return"
    return "statement"


def ast_statement_reason(features: Dict[str, int]) -> str:
    active = [name for name, count in features.items() if count]
    if not active:
        return "generic Python statement"
    return "AST structural features: " + ", ".join(active)


def iter_ast_decision_nodes(tree: ast.Module) -> List[ast.AST]:
    nodes: List[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in getattr(node, "body", []) or []:
                nodes.append(child)
            continue
        nodes.append(node)
    return nodes


def extract_ast_decision_span_candidates(code: str) -> List[DecisionSpan]:
    normalized = split_semicolon_lines(code or "")
    tree = ast.parse(normalized)
    offsets = line_offsets(normalized)
    spans: List[DecisionSpan] = []
    seen: set[str] = set()

    for node in iter_ast_decision_nodes(tree):
        if skip_ast_statement(node):
            continue
        start, end = node_source_range(normalized, offsets, node)
        text = normalized[start:end].strip()
        if not text or text.startswith("#"):
            continue
        key = re.sub(r"\s+", " ", text)
        if key in seen:
            continue
        seen.add(key)

        features = ast_feature_counts(node)
        category = ast_statement_category(features, node)
        spans.append(
            DecisionSpan(
                text=text,
                start=start,
                end=end,
                line_no=getattr(node, "lineno", 1),
                category=category,
                reason=ast_statement_reason(features),
                code_prefix=normalized[:start],
            )
        )
    return spans


def extract_decision_span_candidates(code: str) -> Tuple[List[DecisionSpan], str]:
    try:
        return extract_ast_decision_span_candidates(code), "ast"
    except SyntaxError:
        return [], "ast_parse_failed"
    except Exception:
        return [], "ast_parse_failed"


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def build_span_selection_prompt(
    *,
    task: str,
    evidence: str,
    spans: Sequence[DecisionSpan],
    max_spans: int,
) -> str:
    lines = [
        "You are selecting Python code spans for an objective evidence-surprise test.",
        "Choose the spans whose decisions are most likely to affect the final answer.",
        "Select only from the numbered candidates. Do not rewrite code.",
        "Prefer spans that define data scope, filtering, parsing/coercion, grouping/selection, aggregation, or final calculation.",
        "Return strict JSON only:",
        '{"selected_indices":[1,2],"reason":"brief"}',
        "",
        "# Task",
        task,
        "",
        "# Exploration Evidence",
        compact_text(evidence, 3000),
        "",
        f"# Candidate Spans (choose up to {max_spans})",
    ]
    for idx, span in enumerate(spans, start=1):
        lines.append(
            f"\n[{idx}] line={span.line_no} category={span.category}\n"
            f"{compact_text(span.text, 700)}"
        )
    return "\n".join(lines)


def build_prefix_parts(
    task: str,
    summary: str,
    code_prefix: str,
    context_char_limit: int,
    code_prefix_char_limit: int,
) -> Tuple[str, str, str]:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    before_summary = task_part + "# EXPLORATION SUMMARY\n"
    summary_part = compact_text(summary, context_char_limit) + "\n\n"
    after_summary = (
        "# GENERATED PYTHON CODE PREFIX\n"
        "The following code continuation was generated by the agent.\n"
        "```python\n"
        + compact_text(code_prefix, code_prefix_char_limit)
    )
    return before_summary, summary_part, after_summary


def build_prior_prefix(task: str, code_prefix: str, code_prefix_char_limit: int) -> str:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    return (
        task_part
        + "# GENERATED PYTHON CODE PREFIX\n"
        + "The following code continuation was generated by the agent.\n"
        + "```python\n"
        + compact_text(code_prefix, code_prefix_char_limit)
    )


class EvidenceSurpriseEvaluator:
    def __init__(
        self,
        config: Optional[EvidenceSurpriseConfig] = None,
        scorer: Optional[ContinuationScorer] = None,
        candidate_extractor: Optional[
            Callable[[str], Tuple[List[DecisionSpan], str]]
        ] = None,
        llm_call: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Any, str]]] = None,
        llm_call_accepts_metadata: bool = False,
    ):
        env_enabled = os.getenv("ACID_EVIDENCE_SURPRISE_ENABLED", "")
        self.config = config or EvidenceSurpriseConfig(
            model_path=os.getenv("ACID_EVIDENCE_SURPRISE_MODEL", DEFAULT_MODEL_PATH),
            device=os.getenv("ACID_EVIDENCE_SURPRISE_DEVICE", "auto"),
            enabled=env_enabled.lower() not in {"0", "false", "no"} if env_enabled else True,
            span_selector_model=os.getenv("ACID_EVIDENCE_SPAN_SELECTOR_MODEL", ""),
        )
        self.scorer = scorer
        self.candidate_extractor = candidate_extractor or extract_decision_span_candidates
        self.llm_call = llm_call or call_llm_with_usage
        self.llm_call_accepts_metadata = llm_call_accepts_metadata
        self._score_cache: Dict[Tuple[str, str], ContinuationScore] = {}
        self._unavailable_error = ""

    def _get_scorer(self) -> Optional[ContinuationScorer]:
        if not self.config.enabled:
            self._unavailable_error = "evidence surprise disabled"
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
                raise RuntimeError(self._unavailable_error or "evidence surprise scorer unavailable")
            self._score_cache[key] = scorer.score(prefix, continuation)
        return self._score_cache[key]

    def _select_spans(
        self,
        *,
        task_id: str = "",
        task_text: str,
        evidence_text: str,
        candidates: Sequence[DecisionSpan],
        extraction_method: str,
    ) -> Tuple[List[DecisionSpan], str]:
        if len(candidates) <= self.config.max_spans:
            return list(candidates), f"{extraction_method}_all"

        selector_model = self.config.span_selector_model or os.getenv(
            "ACID_EVIDENCE_SPAN_SELECTOR_MODEL", ""
        )
        if not selector_model:
            return [], f"{extraction_method}_llm_selector_missing"

        prompt = build_span_selection_prompt(
            task=task_text,
            evidence=evidence_text,
            spans=candidates,
            max_spans=self.config.max_spans,
        )
        try:
            request = {
                "model": selector_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.config.span_selector_max_tokens,
            }
            if self.llm_call_accepts_metadata:
                request["task_id"] = task_id
                request["llm_phase"] = "evidence_surprise_span_selection"
            status, response, _, _ = self.llm_call(request)
        except Exception:
            status, response = False, ""

        if not status or not response:
            return [], f"{extraction_method}_llm_selector_failed"

        payload = parse_json_object(response)
        raw_indices = payload.get("selected_indices", [])
        selected: List[DecisionSpan] = []
        seen: set[int] = set()
        if isinstance(raw_indices, list):
            for value in raw_indices:
                try:
                    index = int(value)
                except Exception:
                    continue
                if index < 1 or index > len(candidates) or index in seen:
                    continue
                seen.add(index)
                selected.append(candidates[index - 1])
                if len(selected) >= self.config.max_spans:
                    break

        if not selected:
            return [], f"{extraction_method}_llm_selector_empty"

        return selected, f"{extraction_method}_llm_selected"

    def _evaluate_span(
        self,
        *,
        span: DecisionSpan,
        task_text: str,
        evidence_text: str,
    ) -> EvidenceSurpriseSpanResult:
        before_summary, summary_part, after_summary = build_prefix_parts(
            task=task_text,
            summary=evidence_text,
            code_prefix=span.code_prefix,
            context_char_limit=self.config.context_char_limit,
            code_prefix_char_limit=self.config.code_prefix_char_limit,
        )
        cond_prefix = before_summary + summary_part + after_summary
        prior_prefix = build_prior_prefix(
            task_text,
            span.code_prefix,
            self.config.code_prefix_char_limit,
        )

        try:
            conditioned = self._score(cond_prefix, span.text)
            prior = self._score(prior_prefix, span.text)
            delta: Optional[float] = None
            ratio: Optional[float] = None
            span_surprise_score: Optional[float] = None
            if finite(conditioned.avg_logp) and finite(prior.avg_logp):
                delta = conditioned.avg_logp - prior.avg_logp
                ratio = safe_exp(delta)
                span_surprise_score = max(0.0, -delta)

            return EvidenceSurpriseSpanResult(
                line_no=span.line_no,
                category=span.category,
                reason=span.reason,
                span=span.text,
                prior_avg_logp=round(prior.avg_logp, 4) if finite(prior.avg_logp) else None,
                conditioned_avg_logp=round(conditioned.avg_logp, 4)
                if finite(conditioned.avg_logp)
                else None,
                delta_cond_minus_prior=round(delta, 4) if delta is not None else None,
                ratio_p_cond_over_prior=round(ratio, 4) if ratio is not None else None,
                span_surprise_score=round(span_surprise_score, 4)
                if span_surprise_score is not None
                else None,
                n_tokens=conditioned.n_tokens,
            )
        except Exception as exc:
            return EvidenceSurpriseSpanResult(
                line_no=span.line_no,
                category=span.category,
                reason=span.reason,
                span=span.text,
                error=str(exc),
            )

    def evaluate(
        self,
        *,
        task_text: str,
        evidence_text: str,
        code: str,
        action_type: str = "",
        step_id: str = "",
        task_id: str = "",
    ) -> EvidenceSurpriseResult:
        result = EvidenceSurpriseResult(
            enabled=bool(self.config.enabled),
            model=self.config.model_path,
            available=False,
        )
        if not self.config.enabled:
            result.risk_status = "unavailable"
            result.signals.append("evidence surprise disabled")
            return result
        if action_type and action_type.lower() != "python":
            result.signals.append(f"evidence surprise skipped for non-python action: {action_type}")
            return result
        if not (code or "").strip():
            result.signals.append("no code available for evidence surprise scoring")
            return result
        if not (evidence_text or "").strip():
            result.signals.append("no evidence text available for evidence surprise scoring")
            return result

        candidates, extraction_method = self.candidate_extractor(code)
        result.candidate_span_count = len(candidates)
        if not candidates:
            result.span_selection_method = extraction_method
            if extraction_method.endswith("parse_failed"):
                result.risk_status = "unavailable"
                result.signals.append("evidence surprise unavailable: Python AST parse failed")
            else:
                result.signals.append("no Python decision spans found for evidence surprise scoring")
            return result

        spans, selection_method = self._select_spans(
            task_id=task_id,
            task_text=task_text,
            evidence_text=evidence_text,
            candidates=candidates,
            extraction_method=extraction_method,
        )
        result.span_selection_method = selection_method
        result.selected_span_count = len(spans)
        if not spans:
            result.signals.append(
                "evidence surprise skipped: span selection requires LLM output when candidates exceed budget"
            )
            return result

        if self._get_scorer() is None:
            result.risk_status = "unavailable"
            result.signals.append(f"evidence surprise unavailable: {self._unavailable_error}")
            result.spans = [asdict(span) for span in spans]
            return result

        span_results = [
            self._evaluate_span(span=span, task_text=task_text, evidence_text=evidence_text)
            for span in spans
        ]
        scored = [span for span in span_results if not span.error]
        result.spans = [
            span.to_dict()
            for span in sorted(
                span_results,
                key=lambda item: item.span_surprise_score or 0.0,
                reverse=True,
            )
        ]
        if not scored:
            result.risk_status = "unavailable"
            result.signals.append("evidence surprise unavailable: no spans scored successfully")
            return result

        result.available = True
        result.max_span_surprise_score = round(
            max((span.span_surprise_score or 0.0) for span in scored),
            4,
        )
        if result.max_span_surprise_score > self.config.retry_span_surprise_threshold:
            result.risk_status = "retry"
            result.retry_recommendation = "retry"
            result.signals.append(
                "evidence surprise recommends retry: "
                f"max_span_surprise={result.max_span_surprise_score:.4f}"
            )
        elif result.max_span_surprise_score > self.config.warning_span_surprise_threshold:
            result.risk_status = "warning"
            result.retry_recommendation = "warning"
            result.signals.append(
                "evidence surprise warning: "
                f"max_span_surprise={result.max_span_surprise_score:.4f}"
            )
        else:
            result.risk_status = "pass"
            result.signals.append("no evidence surprise signal crossed warning thresholds")

        for span in result.spans[:3]:
            span_score = span.get("span_surprise_score") or 0.0
            if span_score >= self.config.warning_span_surprise_threshold:
                result.signals.append(
                    "suspicious Python span: "
                    f"line={span.get('line_no')} category={span.get('category')} "
                    f"span_surprise={span_score:.4f} "
                    f"span={span.get('span', '')!r}"
                )
        return result
