"""Objective-review driven retry policy.

This module decides whether an executed step should receive a follow-up review
step. It does not execute code and does not mutate the agent directly; callers
append the returned pending feedback to the next prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from da_agent.agent.prompts import RETRY_REVIEW_PROMPT_BLOCK


STATUS_RETRY = "retry"


@dataclass
class RetryPolicyConfig:
    enabled: bool = True
    max_retry_per_step: int = 2
    max_issues_per_retry: int = 4

    def __post_init__(self) -> None:
        self.max_retry_per_step = int(self.max_retry_per_step)


@dataclass
class RetryIssue:
    issue_key: str
    issue_type: str
    severity: str
    title: str
    reason: str
    current_path: str = ""
    alternative_paths: List[str] = field(default_factory=list)
    required_checks: List[str] = field(default_factory=list)
    snippets: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    retry_count_before: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetryDecision:
    should_retry: bool
    task_id: str = ""
    step_id: str = ""
    prompt_block: str = ""
    issues: List[RetryIssue] = field(default_factory=list)
    budget_exhausted: bool = False
    step_exhausted: bool = False
    retry_budget_used: int = 0
    retry_chain_root_step_id: str = ""
    retry_chain_used: int = 0
    retry_chain_limit: int = 0
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data

    def to_pending_feedback(self) -> Dict[str, Any]:
        return {
            "retry_review": True,
            "step_id": self.step_id,
            "retry_chain_root_step_id": self.retry_chain_root_step_id or self.step_id,
            "prompt_block": self.prompt_block,
            "retry_decision": self.to_dict(),
        }


class RetryPolicy:
    def __init__(self, config: Optional[RetryPolicyConfig] = None):
        self.config = config or RetryPolicyConfig()
        self.reset()

    def reset(self) -> None:
        self.retry_budget_used_by_task: Dict[str, int] = {}
        self.retry_chain_counts: Dict[str, Dict[str, int]] = {}
        self.unresolved_issues_by_task: Dict[str, List[Dict[str, Any]]] = {}

    def evaluate(
        self,
        *,
        task_id: str,
        step_id: str,
        thought: str,
        action_text: str,
        code: str,
        action_type: str,
        observation: str,
        exploration_summary: str,
        objective_confidence: Dict[str, Any],
        is_retry_step: bool = False,
        retry_chain_root_step_id: str = "",
    ) -> RetryDecision:
        if not self.config.enabled:
            return self._no_retry(task_id, step_id, ["retry policy disabled"])

        chain_root = retry_chain_root_step_id or step_id

        issues = self._build_issues(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action_text=action_text,
            code=code,
            action_type=action_type,
            observation=observation,
            exploration_summary=exploration_summary,
            objective_confidence=objective_confidence,
        )
        if not issues:
            return self._no_retry(task_id, step_id, ["no retry-worthy objective review issue found"])

        task_used = self.retry_budget_used_by_task.get(task_id, 0)
        chain_counts = self.retry_chain_counts.setdefault(task_id, {})
        chain_used = chain_counts.get(chain_root, 0)
        if chain_used >= self.config.max_retry_per_step:
            self._remember_unresolved(task_id, issues, "step retry chain cap reached")
            return self._no_retry(
                task_id,
                step_id,
                ["step retry chain cap reached; continue normal solving"],
                step_exhausted=True,
                retry_chain_root_step_id=chain_root,
            )

        for issue in issues:
            issue.retry_count_before = chain_used
        selected = self._select_issues(issues)
        for issue in selected:
            issue.retry_count_before = chain_used
        chain_counts[chain_root] = chain_used + 1
        self.retry_budget_used_by_task[task_id] = task_used + 1

        prompt_block = self._build_prompt_block(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action_text=action_text,
            code=code,
            observation=observation,
            objective_confidence=objective_confidence,
            issues=selected,
            retry_budget_used=task_used + 1,
            retry_chain_root_step_id=chain_root,
            retry_chain_used=chain_used + 1,
            is_retry_step=is_retry_step,
        )

        return RetryDecision(
            should_retry=True,
            task_id=task_id,
            step_id=step_id,
            prompt_block=prompt_block,
            issues=selected,
            retry_budget_used=task_used + 1,
            retry_chain_root_step_id=chain_root,
            retry_chain_used=chain_used + 1,
            retry_chain_limit=self.config.max_retry_per_step,
            signals=[
                f"review retry scheduled for {step_id}",
                f"task retry count used {task_used + 1}",
                f"step retry chain used {chain_used + 1}/{self.config.max_retry_per_step}",
                f"retry chain root={chain_root}",
            ],
        )

    def _remember_unresolved(
        self,
        task_id: str,
        issues: Sequence[RetryIssue],
        reason: str,
    ) -> None:
        bucket = self.unresolved_issues_by_task.setdefault(task_id, [])
        existing = {
            (item.get("issue_key"), item.get("unresolved_reason"))
            for item in bucket
        }
        for issue in issues:
            key = (issue.issue_key, reason)
            if key in existing:
                continue
            data = issue.to_dict()
            data["unresolved_reason"] = reason
            bucket.append(data)
            existing.add(key)

    def _no_retry(
        self,
        task_id: str,
        step_id: str,
        signals: Sequence[str],
        *,
        budget_exhausted: bool = False,
        step_exhausted: bool = False,
        retry_chain_root_step_id: str = "",
    ) -> RetryDecision:
        chain_root = retry_chain_root_step_id or step_id
        return RetryDecision(
            should_retry=False,
            task_id=task_id,
            step_id=step_id,
            budget_exhausted=budget_exhausted,
            step_exhausted=step_exhausted,
            retry_budget_used=self.retry_budget_used_by_task.get(task_id, 0),
            retry_chain_root_step_id=chain_root,
            retry_chain_used=self.retry_chain_counts.get(task_id, {}).get(chain_root, 0),
            retry_chain_limit=self.config.max_retry_per_step,
            signals=list(signals),
        )

    def _select_issues(self, issues: Sequence[RetryIssue]) -> List[RetryIssue]:
        severity_rank = {STATUS_RETRY: 0}
        sorted_issues = sorted(
            issues,
            key=lambda issue: (
                severity_rank.get(issue.severity, 2),
                issue.issue_type,
            ),
        )
        return list(sorted_issues[: self.config.max_issues_per_retry])

    def _build_issues(
        self,
        *,
        task_id: str,
        step_id: str,
        thought: str,
        action_text: str,
        code: str,
        action_type: str,
        observation: str,
        exploration_summary: str,
        objective_confidence: Dict[str, Any],
    ) -> List[RetryIssue]:
        issues: List[RetryIssue] = []
        need_retry = objective_confidence.get("need_retry")
        if need_retry is not True:
            return []

        components = objective_confidence.get("components") or {}
        probability_contrast = objective_confidence.get("probability_contrast", {}) or {}
        evidence_surprise = objective_confidence.get("evidence_surprise", {}) or {}
        for component_name, component in components.items():
            if not isinstance(component, dict):
                continue
            status = str(component.get("status") or "").lower()
            if status != STATUS_RETRY:
                continue
            title, reason, checks = self._component_issue_text(component_name, status)
            issues.append(self._issue(
                task_id,
                step_id,
                component_name,
                status,
                title,
                reason,
                current_path=self._current_path(thought, action_text, code),
                alternative_paths=(
                    self._alternatives_from_probability(probability_contrast)
                    if component_name == "probability_contrast"
                    else self._alternatives_from_evidence_surprise(evidence_surprise)
                    if component_name == "evidence_surprise"
                    else []
                ),
                snippets=(
                    self._probability_snippets(probability_contrast)
                    if component_name == "probability_contrast"
                    else self._evidence_surprise_snippets(evidence_surprise)
                    if component_name == "evidence_surprise"
                    else []
                ),
                signals=component.get("signals", []),
                required_checks=checks,
            ))

        if not issues:
            issues.append(self._issue(
                task_id, step_id, "review_decision", STATUS_RETRY,
                "Objective review requires retry",
                "The objective review requires retry even though no single component created a more specific issue.",
                current_path=self._current_path(thought, action_text, code),
                signals=self._flatten_signal_items(objective_confidence.get("retry_reasons", []))[:4],
                required_checks=[
                    "Run an independent minimal verification of the final quantity.",
                    "Print intermediate counts and compare them with exploration anchors.",
                ],
            ))

        return self._dedupe_issues(issues)

    @staticmethod
    def _component_issue_text(component_name: str, status: str) -> tuple[str, str, List[str]]:
        severity = "requires retry"
        if component_name == "execution_observation":
            return (
                f"Execution/stdout channel {severity}",
                "The action did not provide reliable executable evidence, or stderr introduced watch-level risk.",
                [
                    "Run a minimal command that prints visible output.",
                    "Check exit code, stdout, and stderr before reusing this step.",
                    "If the step writes a file, verify the file exists and print a small preview.",
                ],
            )
        if component_name == "reasoning_observation":
            return (
                f"Reasoning-observation consistency {severity}",
                "The thought/action appears to ignore or contradict usable observation output.",
                [
                    "Re-read the source observation and use its concrete values.",
                    "Do not claim output is missing if stdout/observation contains usable output.",
                ],
            )
        if component_name == "evidence_surprise":
            return (
                f"Evidence-conditioned code surprise {severity}",
                "A selected Python decision span became unlikely after conditioning on exploration evidence.",
                [
                    "Verify the suspicious code span against concrete exploration anchors.",
                    "Print counts/results before and after the suspect filter, parse, or aggregation.",
                    "If there are two plausible interpretations, implement both paths side by side.",
                ],
            )
        if component_name == "probability_contrast":
            return (
                f"Probability contrast {severity}",
                "A contrast probe found an evidence-supported alternative that is more plausible than the current decision.",
                [
                    "Implement current and alternative decisions in small checks.",
                    "Print the counts/results from both paths side by side.",
                    "Keep the current result only if code verification supports it with concrete evidence.",
                ],
            )
        return (
            f"{component_name} {severity}",
            "This objective review component requires targeted verification.",
            ["Run targeted code verification for this component before trusting the step."],
        )

    def _issue(
        self,
        task_id: str,
        step_id: str,
        issue_type: str,
        severity: str,
        title: str,
        reason: str,
        *,
        current_path: str = "",
        alternative_paths: Optional[List[str]] = None,
        required_checks: Optional[List[str]] = None,
        snippets: Optional[List[str]] = None,
        signals: Optional[List[str]] = None,
    ) -> RetryIssue:
        alternative_paths = self._compact_list(alternative_paths or [])
        required_checks = self._compact_list(required_checks or [])
        snippets = self._compact_list(snippets or [])
        signals = self._compact_list(signals or [])
        issue_key = self._issue_key(
            task_id=task_id,
            issue_type=issue_type,
            severity=severity,
            snippets=snippets,
            signals=signals,
            title=title,
        )
        return RetryIssue(
            issue_key=issue_key,
            issue_type=issue_type,
            severity=severity,
            title=title,
            reason=reason,
            current_path=current_path,
            alternative_paths=alternative_paths,
            required_checks=required_checks,
            snippets=snippets,
            signals=signals,
        )

    def _issue_key(
        self,
        *,
        task_id: str,
        issue_type: str,
        severity: str,
        snippets: Sequence[str],
        signals: Sequence[str],
        title: str,
    ) -> str:
        payload = {
            "task_id": task_id,
            "issue_type": issue_type,
            "severity": severity,
            "title": title,
            "snippets": list(snippets)[:3],
            "signals": list(signals)[:2],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{issue_type}:{digest}"

    @staticmethod
    def _compact_list(items: Iterable[Any], limit: int = 6, item_limit: int = 240) -> List[str]:
        out: List[str] = []
        for item in items or []:
            text = str(item).strip()
            if not text or text in out:
                continue
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _current_path(thought: str, action_text: str, code: str) -> str:
        parts = []
        if thought:
            parts.append("Thought:\n" + str(thought).strip())
        if code:
            parts.append("Code:\n" + str(code).strip())
        elif action_text:
            parts.append("Action:\n" + str(action_text).strip())
        return "\n".join(parts)

    @staticmethod
    def _alternatives_from_probability(probability_contrast: Dict[str, Any]) -> List[str]:
        alternatives = []
        for probe in probability_contrast.get("probes", []) or []:
            alternative = probe.get("alternative")
            current = probe.get("current")
            name = probe.get("name", "contrast")
            if alternative:
                alternatives.append(
                    f"{name}: compare current={current!r} with alternative={alternative!r}"
                )
        return alternatives[:4]

    @staticmethod
    def _probability_snippets(probability_contrast: Dict[str, Any]) -> List[str]:
        snippets = []
        for probe in probability_contrast.get("probes", []) or []:
            if probe.get("code_line"):
                snippets.append(str(probe["code_line"]))
            elif probe.get("current"):
                snippets.append(str(probe["current"]))
        return snippets[:4]

    @staticmethod
    def _alternatives_from_evidence_surprise(evidence_surprise: Dict[str, Any]) -> List[str]:
        alternatives = []
        for span in evidence_surprise.get("spans", []) or []:
            category = span.get("category", "span")
            line_no = span.get("line_no", "?")
            if span.get("span"):
                alternatives.append(
                    f"{category} line {line_no}: verify whether this span is supported by exploration evidence"
                )
        return alternatives[:4]

    @staticmethod
    def _evidence_surprise_snippets(evidence_surprise: Dict[str, Any]) -> List[str]:
        snippets = []
        for span in evidence_surprise.get("spans", []) or []:
            text = span.get("span")
            span_surprise = span.get("span_surprise_score")
            if text:
                snippets.append(f"span_surprise={span_surprise}: {text}")
        return snippets[:4]

    def _dedupe_issues(self, issues: Sequence[RetryIssue]) -> List[RetryIssue]:
        seen = set()
        deduped: List[RetryIssue] = []
        for issue in issues:
            if issue.issue_key in seen:
                continue
            seen.add(issue.issue_key)
            deduped.append(issue)
        return deduped

    def _build_prompt_block(
        self,
        *,
        task_id: str,
        step_id: str,
        thought: str,
        action_text: str,
        code: str,
        observation: str,
        objective_confidence: Dict[str, Any],
        issues: Sequence[RetryIssue],
        retry_budget_used: int,
        retry_chain_root_step_id: str,
        retry_chain_used: int,
        is_retry_step: bool,
    ) -> str:
        issues_text = self._format_issues(issues)
        objective_summary = self._objective_summary(objective_confidence)
        watchlist_text = self._format_watchlist(objective_confidence.get("watchlist", []))
        if watchlist_text:
            objective_summary = (
                objective_summary
                + "\n\n# WATCHLIST / NON-BLOCKING SIGNALS #\n"
                + "These signals did not trigger retry by themselves, but inspect them while reviewing this step:\n"
                + watchlist_text
            )
        prior_mode = "This review follows a previous retry step." if is_retry_step else "This review follows a normal task step."
        return RETRY_REVIEW_PROMPT_BLOCK.format(
            step_id=step_id,
            task_id=task_id,
            retry_budget_used=retry_budget_used,
            retry_chain_root_step_id=retry_chain_root_step_id,
            retry_chain_used=retry_chain_used,
            retry_chain_limit=self.config.max_retry_per_step,
            prior_mode=prior_mode,
            objective_summary=objective_summary,
            issues=issues_text,
            current_path=self._current_path(thought, action_text, code) or "(No code/action text captured.)",
            observation=(observation or "").strip()[:3000] or "(empty observation)",
        )

    def _format_issues(self, issues: Sequence[RetryIssue]) -> str:
        blocks = []
        for idx, issue in enumerate(issues, start=1):
            block = [
                f"{idx}. [{issue.severity.upper()}] {issue.title}",
                f"   Issue key: {issue.issue_key}",
                f"   Reason: {issue.reason}",
            ]
            if issue.signals:
                block.append("   Signals:")
                block.extend(f"   - {signal}" for signal in issue.signals[:4])
            if issue.snippets:
                block.append("   Related snippets/terms:")
                block.extend(f"   - {snippet}" for snippet in issue.snippets[:4])
            if issue.alternative_paths:
                block.append("   Alternative path(s) to compare:")
                block.extend(f"   - {alt}" for alt in issue.alternative_paths[:4])
            if issue.required_checks:
                block.append("   Required code checks:")
                block.extend(f"   - {check}" for check in issue.required_checks[:5])
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    @staticmethod
    def _flatten_signal_items(items: Any) -> List[str]:
        flattened: List[str] = []
        if not isinstance(items, list):
            items = [items] if items else []
        for item in items:
            if isinstance(item, dict):
                component = str(item.get("component") or "").strip()
                status = str(item.get("status") or "").strip()
                signals = item.get("signals") or []
                prefix = " / ".join(part for part in (component, status) if part)
                if isinstance(signals, list) and signals:
                    for signal in signals:
                        text = str(signal).strip()
                        if text:
                            flattened.append(f"{prefix}: {text}" if prefix else text)
                elif prefix:
                    flattened.append(prefix)
            else:
                text = str(item).strip()
                if text:
                    flattened.append(text)
        return flattened

    @classmethod
    def _format_watchlist(cls, watchlist: Any) -> str:
        signals = cls._flatten_signal_items(watchlist)
        if not signals:
            return ""
        return "\n".join(f"- {signal}" for signal in signals[:8])

    @staticmethod
    def _objective_summary(objective_confidence: Dict[str, Any]) -> str:
        keys = [
            "schema_version",
            "review_decision",
            "need_retry",
            "probability_contrast_recommendation",
            "probability_contrast_min_ratio",
            "evidence_surprise_max_span_score",
            "evidence_surprise_recommendation",
        ]
        parts = []
        for key in keys:
            if key in objective_confidence:
                parts.append(f"{key}={objective_confidence.get(key)}")

        components = objective_confidence.get("components") or {}
        if isinstance(components, dict) and components:
            compact = [
                f"{name}:{value.get('status')}"
                for name, value in list(components.items())[:6]
                if isinstance(value, dict)
            ]
            if compact:
                parts.append("components=" + "; ".join(compact))
        return ", ".join(parts)
