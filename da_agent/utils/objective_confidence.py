"""Objective review signals for ACID agent steps.

This module extracts the objective step-review logic out of ``acid_agent.py``.
It is intentionally written against plain strings and dictionaries instead of
agent Action classes, so it can be used both online by the agent and offline on
logs.

The current schema is decision-oriented: retry-level objective risks set
``need_retry``; watch-level risks are recorded in ``watchlist`` and do not
trigger retry by themselves. Probability contrast and evidence surprise are
merged by the caller using the same pass/watch/retry convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 9

STATUS_PASS = "pass"
STATUS_WATCH = "watch"
STATUS_RETRY = "retry"
STATUS_SKIPPED = "skipped"
INBOUND_WARNING = "warning"


@dataclass
class ObjectiveConfidenceInput:
    task_id: str
    step_id: str
    thought: str
    action_type: str
    code: str
    observation: str
    evidence_text: str
    action_output: str = ""
    execution_meta: Dict[str, Any] = field(default_factory=dict)
    observation_use_judgment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveConfidenceConfig:
    schema_version: int = SCHEMA_VERSION


def is_terminate_action(action_type: str) -> bool:
    return (action_type or "").strip().lower() == "terminate"


def execution_meta_available(execution_meta: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(execution_meta, dict) or not execution_meta:
        return False
    return any(key in execution_meta for key in ("exit_code", "stdout", "stderr", "timed_out"))


def classify_execution_for_objective(
    observation: str,
    execution_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """Classify execution using structured exit/stdout/stderr signals.

    This intentionally avoids traceback/error keyword extraction in the
    execution layer.  If structured metadata is unavailable, it only checks
    whether any visible observation was produced.
    """
    if execution_meta_available(execution_meta):
        meta = execution_meta or {}
        if meta.get("timed_out"):
            return "error", ["execution timed out"]

        exit_code = meta.get("exit_code")
        try:
            exit_code_int = int(exit_code) if exit_code is not None else None
        except Exception:
            exit_code_int = None

        stdout = str(meta.get("stdout") or "")
        stderr = str(meta.get("stderr") or "")
        stdout_nonempty = bool(stdout.strip())
        stderr_nonempty = bool(stderr.strip())

        if exit_code_int is not None and exit_code_int != 0:
            return "error", [f"execution exit_code={exit_code_int}"]
        if not stdout_nonempty:
            stderr_note = (
                f"; stderr output was present ({len(stderr.strip())} chars)"
                if stderr_nonempty else ""
            )
            return "empty_observation", [
                "exit_code=0 but execution produced no stdout" + stderr_note
            ]
        if stderr_nonempty:
            return "stderr_nonempty", [
                f"exit_code=0 with stderr output ({len(stderr.strip())} chars)"
            ]
        return "standard_output", [
            f"exit_code=0 with stdout output ({len(stdout.strip())} chars)"
        ]

    if not (observation or "").strip():
        return "empty_observation", [
            "execution metadata unavailable and observation is empty"
        ]
    if "executed successfully. No output" in (observation or ""):
        return "empty_observation", [
            "execution metadata unavailable and observation reports no visible output"
        ]
    return "standard_output", [
        "execution metadata unavailable; scored by visible output presence only"
    ]


def compact_execution_meta(execution_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not execution_meta_available(execution_meta):
        return {}
    meta = execution_meta or {}
    stdout = str(meta.get("stdout") or "")
    stderr = str(meta.get("stderr") or "")
    compact = {
        "exit_code": meta.get("exit_code"),
        "stdout_chars": len(stdout.strip()),
        "stderr_chars": len(stderr.strip()),
    }
    if meta.get("timed_out"):
        compact["timed_out"] = True
    if meta.get("action_type"):
        compact["action_type"] = meta.get("action_type")
    return compact


class ObjectiveConfidenceEvaluator:
    def __init__(self, config: Optional[ObjectiveConfidenceConfig] = None):
        self.config = config or ObjectiveConfidenceConfig()

    @staticmethod
    def _component(
        status: str,
        signals: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "signals": list(signals or []),
        }

    @staticmethod
    def _normalize_status(status: str) -> str:
        status = (status or STATUS_PASS).lower()
        if status == INBOUND_WARNING:
            return STATUS_WATCH
        if status in {STATUS_PASS, STATUS_WATCH, STATUS_RETRY, STATUS_SKIPPED}:
            return status
        if status in {"unavailable", "no_signal"}:
            return STATUS_SKIPPED
        return STATUS_PASS

    def score_execution_observation(
        self,
        action_type: str,
        action_output: str,
        observation: str,
        execution_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if is_terminate_action(action_type):
            if action_output and str(action_output).strip() and str(action_output).strip().upper() != "FAIL":
                return self._component(STATUS_PASS, ["terminate produced a non-empty answer"])
            return self._component(STATUS_RETRY, ["terminate output is empty or FAIL"])

        obs_class, class_signals = classify_execution_for_objective(
            observation or "",
            execution_meta,
        )
        if obs_class == "empty_observation":
            return self._component(STATUS_RETRY, class_signals)
        if obs_class == "error":
            return self._component(
                STATUS_RETRY,
                [f"execution classified as {obs_class}"] + class_signals,
            )
        if obs_class == "stderr_nonempty":
            return self._component(STATUS_WATCH, class_signals)
        return self._component(STATUS_PASS, class_signals)

    def score_thought_observation_consistency(
        self,
        thought: str,
        observation: str,
        observation_use_judgment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current_obs = (observation or "").strip()
        if not current_obs:
            return self._component(
                STATUS_SKIPPED,
                ["observation-use judge skipped because current observation/stdout is empty"],
            )

        judgment = observation_use_judgment or {}
        if not judgment:
            return self._component(STATUS_SKIPPED, ["observation-use judge not run"])

        status = str(judgment.get("status") or judgment.get("risk_status") or STATUS_PASS).lower()
        if status in {"unavailable", "no_signal"}:
            reason = str(judgment.get("reason") or "").strip()
            signals = [f"observation-use judge status={status}"]
            if reason:
                signals.append(reason[:300])
            return self._component(STATUS_SKIPPED, signals)
        status = self._normalize_status(status)

        ignored = judgment.get("ignored_key_facts") or []
        if not isinstance(ignored, list):
            ignored = [str(ignored)]
        reason = str(judgment.get("reason") or "").strip()
        signals = [f"observation-use judge status={status}"]
        if reason:
            signals.append(reason[:300])
        if ignored:
            signals.append("ignored key facts: " + ", ".join(str(item) for item in ignored[:5]))

        return self._component(status, signals)

    def evaluate(self, data: ObjectiveConfidenceInput) -> Dict[str, Any]:
        execution_component = self.score_execution_observation(
            data.action_type,
            data.action_output,
            data.observation,
            data.execution_meta,
        )
        execution_signals = list(execution_component.get("signals", []))

        reasoning_component = self.score_thought_observation_consistency(
            data.thought,
            data.observation,
            data.observation_use_judgment,
        )

        components = {
            "execution_observation": execution_component,
            "reasoning_observation": reasoning_component,
        }
        need_retry = False
        retry_reasons: List[Dict[str, Any]] = []
        watchlist: List[Dict[str, Any]] = []
        for name, component in components.items():
            status = component.get("status")
            if status == STATUS_RETRY:
                need_retry = True
                retry_reasons.append({
                    "component": name,
                    "status": status,
                    "signals": component.get("signals", [])[:5],
                })
            elif status == STATUS_WATCH:
                watchlist.append({
                    "component": name,
                    "status": status,
                    "signals": component.get("signals", [])[:5],
                })

        return {
            "schema_version": self.config.schema_version,
            "step_id": data.step_id,
            "task_id": data.task_id,
            "review_decision": STATUS_RETRY if need_retry else STATUS_PASS,
            "need_retry": need_retry,
            "retry_reasons": retry_reasons,
            "watchlist": watchlist,
            "execution_meta": compact_execution_meta(data.execution_meta),
            "components": components,
            "signals": {
                "execution": execution_signals,
                "reasoning_observation": reasoning_component.get("signals", []),
            },
        }
