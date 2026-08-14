"""Exploration evidence packaging and summary reconciliation.

This module keeps the evidence-flow utilities out of the main agent loop. It
does not run exploration actions; it only formats, merges, revises, finalizes,
and packages summaries that the agent has already produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from da_agent.agent.prompts import (
    EXPLORATION_CONFLICT_REVISION_PROMPT,
    EXPLORATION_FINAL_SUMMARY_PROMPT,
)
from da_agent.utils.llm_utils import call_llm_with_usage


NO_EVIDENCE = "No evidence gathered yet."
RESOLVED_SUMMARY_MARKER = "# RESOLVED EXPLORATION SUMMARY #"
STRUCTURED_ANCHORS_MARKER = "STRUCTURED_EVIDENCE_ANCHORS:"
CONFLICT_LEDGER_MARKER = "EXPLORATION_CONFLICTS:"
STRUCTURED_ANCHOR_FAMILIES = (
    "source_scope_anchors",
    "filter_scope_anchors",
    "field_semantics_anchors",
    "aggregation_grain_anchors",
    "additional_decision_anchors",
)


def merge_evidence(*parts: str) -> str:
    """Join non-empty evidence blocks while preserving their order."""
    return "\n\n".join(str(part).strip() for part in parts if str(part or "").strip())


def default_compact_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head_len = limit // 2
    tail_len = limit - head_len
    return text[:head_len] + "\n... (middle truncated) ...\n" + text[-tail_len:]


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def parse_first_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    start = raw.find("{")
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return parse_json_object(raw[start:])


@dataclass
class ExplorationEvidenceBundle:
    known_evidence: str = NO_EVIDENCE
    current_summary: str = ""
    source_retry_context: str = ""
    effective_summary: str = ""
    scoring_evidence_text: str = ""
    structured_anchor_text: str = ""
    structured_anchors: Dict[str, Any] = field(default_factory=dict)
    authoritative_facts: List[str] = field(default_factory=list)
    verified_policies: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


class ExplorationEvidenceManager:
    """Builds the evidence strings shared by code generation, OC, ES, and PC."""

    def __init__(
        self,
        *,
        model: str,
        compact_text: Optional[Callable[[str, int], str]] = None,
        llm_call: Callable[..., Any] = call_llm_with_usage,
        llm_call_accepts_metadata: bool = False,
    ) -> None:
        self.model = model
        self.compact_text = compact_text or default_compact_text
        self.llm_call = llm_call
        self.llm_call_accepts_metadata = llm_call_accepts_metadata

    @staticmethod
    def as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value in (None, ""):
            return []
        return [value]

    @staticmethod
    def as_string_list(value: Any) -> List[str]:
        items = value if isinstance(value, list) else ([] if value in (None, "") else [value])
        result: List[str] = []
        for item in items:
            if isinstance(item, (dict, list)):
                text = json.dumps(item, ensure_ascii=False)
            else:
                text = str(item)
            text = text.strip()
            if text:
                result.append(text)
        return result

    def build_known_evidence(
        self,
        task_exploration_summaries: Dict[str, List[str]],
        task_id: str,
        current_local_summaries: Optional[Sequence[str]] = None,
    ) -> str:
        summaries = task_exploration_summaries.get(task_id, [])
        current_local = [
            s for s in (current_local_summaries or [])
            if s and str(s).strip() and str(s).strip().upper() != "SKIP"
        ][-3:]
        if not summaries and not current_local:
            return NO_EVIDENCE
        usable = [s for s in summaries if s.strip().upper() != "SKIP"]
        resolved = [s for s in usable if RESOLVED_SUMMARY_MARKER in s]

        parts: List[str] = []
        prior = resolved[-2:] if resolved else ([] if current_local else usable[-2:])
        if prior:
            header = "# PRIOR RESOLVED EVIDENCE #" if resolved else "# RECENT KNOWN EVIDENCE #"
            parts.append(header + "\n" + "\n".join(prior))
        if current_local:
            parts.append(
                "# CURRENT EPISODE LOCAL EVIDENCE (PROVISIONAL) #\n"
                + self.format_local_summaries(current_local)
            )
        if not parts:
            return NO_EVIDENCE
        return "\n\n".join(parts)

    def format_local_summaries(self, summaries: Sequence[str]) -> str:
        parts: List[str] = []
        for idx, summary in enumerate(summaries, start=1):
            text = self.compact_text(summary, 4000)
            parts.append(f"## local_summary_{idx}\n{text}")
        return "\n\n".join(parts)

    def fallback_merge_state(
        self,
        summaries: Sequence[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        merged = self.format_local_summaries(list(summaries))
        payload: Dict[str, Any] = {
            "merged_summary": merged,
            "deterministic_merge": True,
            "local_summary_count": len([summary for summary in summaries if summary]),
            "needs_targeted_exploration": False,
        }
        if reason:
            payload["merge_error"] = reason
        return payload

    def merge_summaries(self, summaries: Sequence[str]) -> Dict[str, Any]:
        usable = [summary for summary in summaries if (summary or "").strip()]
        if not usable:
            return self.fallback_merge_state([])
        return self.fallback_merge_state(usable)

    @staticmethod
    def format_merge_state(merge_state: Dict[str, Any]) -> str:
        return json.dumps(merge_state, ensure_ascii=False, indent=2)

    @staticmethod
    def merge_needs_conflict_revision(merge_state: Dict[str, Any]) -> bool:
        return bool((merge_state.get("merged_summary") or "").strip())

    def normalize_conflict_revision_state(
        self,
        payload: Dict[str, Any],
        merge_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            payload = {
                "authoritative_facts": [],
                "verified_policies": [],
                "conflicts": [],
                "conflict_revision_failed": True,
            }

        normalized = dict(payload)
        normalized.setdefault("merged_summary", merge_state.get("merged_summary", ""))
        normalized["authoritative_facts"] = self.as_string_list(
            normalized.get("authoritative_facts")
        )
        normalized["conflicts"] = self.normalize_conflicts(normalized.get("conflicts"))
        verified_policies = self.as_string_list(normalized.get("verified_policies"))
        unverified_conflict_options = {
            str(option.get("policy") or "").strip().lower()
            for conflict in normalized["conflicts"]
            if conflict.get("status") != "verified"
            for option in conflict.get("options", [])
        }
        normalized["verified_policies"] = [
            policy for policy in verified_policies
            if policy.strip().lower() not in unverified_conflict_options
        ]
        verified_policy_keys = {
            policy.strip().lower() for policy in normalized["verified_policies"]
        }
        for conflict in normalized["conflicts"]:
            if conflict.get("status") != "verified":
                continue
            selected_id = conflict.get("provisional_choice")
            selected_policy = next(
                (
                    str(option.get("policy") or "").strip().lower()
                    for option in conflict.get("options", [])
                    if option.get("option_id") == selected_id
                ),
                "",
            )
            if selected_policy not in verified_policy_keys:
                conflict["status"] = "provisional"
        return normalized

    @classmethod
    def normalize_conflicts(cls, value: Any) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        for index, raw_conflict in enumerate(cls.as_list(value), start=1):
            if not isinstance(raw_conflict, dict):
                continue
            decision_point = str(raw_conflict.get("decision_point") or "").strip()
            options: List[Dict[str, Any]] = []
            used_option_ids: set[str] = set()
            for option_index, raw_option in enumerate(
                cls.as_list(raw_conflict.get("options")), start=1
            ):
                if not isinstance(raw_option, dict):
                    continue
                policy = str(raw_option.get("policy") or "").strip()
                if not policy:
                    continue
                option_id = str(
                    raw_option.get("option_id") or chr(64 + min(option_index, 26))
                ).strip()
                if not option_id or option_id in used_option_ids:
                    option_id = f"option_{option_index}"
                used_option_ids.add(option_id)
                options.append({
                    "option_id": option_id,
                    "policy": policy,
                    "supporting_evidence": cls.as_string_list(
                        raw_option.get("supporting_evidence")
                    ),
                })

            if not decision_point or len(options) < 2:
                continue
            status = str(raw_conflict.get("status") or "unresolved").strip().lower()
            if status not in {"unresolved", "provisional", "verified"}:
                status = "unresolved"
            option_ids = {option["option_id"] for option in options}
            provisional_choice = str(
                raw_conflict.get("provisional_choice") or ""
            ).strip()
            if provisional_choice not in option_ids:
                provisional_choice = ""
            verification = str(raw_conflict.get("verification") or "").strip()
            if status == "verified" and (not provisional_choice or not verification):
                status = "provisional" if provisional_choice else "unresolved"
            if status == "provisional" and not provisional_choice:
                status = "unresolved"
            if status == "unresolved":
                provisional_choice = ""
            conflicts.append({
                "conflict_id": str(
                    raw_conflict.get("conflict_id") or f"conflict_{index}"
                ).strip(),
                "decision_point": decision_point,
                "options": options,
                "provisional_choice": provisional_choice,
                "status": status,
                "verification": verification,
            })
        return conflicts

    def resolve_conflicts(
        self,
        *,
        task_id: str = "",
        task_instruction: str,
        merge_state: Dict[str, Any],
        summaries: Sequence[str],
    ) -> Dict[str, Any]:
        prompt = EXPLORATION_CONFLICT_REVISION_PROMPT.format(
            task=task_instruction,
            merge_state=self.format_merge_state(merge_state),
            summaries=self.format_local_summaries(list(summaries)),
        )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2500,
        }
        if self.llm_call_accepts_metadata:
            request["task_id"] = task_id
            request["llm_phase"] = "exploration_conflict_revision"
        status, response, _, _ = self.llm_call(request)
        if not status or not response:
            payload = {
                "authoritative_facts": [],
                "verified_policies": [],
                "conflicts": [],
                "conflict_revision_failed": True,
                "failure_reason": "conflict revision LLM call failed",
            }
            return self.normalize_conflict_revision_state(payload, merge_state)

        payload = parse_json_object(response)
        if not payload:
            payload = {
                "authoritative_facts": [],
                "verified_policies": [],
                "conflicts": [],
                "conflict_revision_failed": True,
                "failure_reason": "conflict revision returned unparsable JSON",
            }
        return self.normalize_conflict_revision_state(payload, merge_state)

    def apply_conflict_revision_state(
        self,
        merge_state: Dict[str, Any],
        revision_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        updated = dict(merge_state)
        updated["conflict_revision"] = revision_state
        updated["authoritative_facts"] = self.as_string_list(
            revision_state.get("authoritative_facts")
        )
        updated["verified_policies"] = self.as_string_list(
            revision_state.get("verified_policies")
        )
        updated["conflicts"] = self.normalize_conflicts(revision_state.get("conflicts"))
        if revision_state.get("conflict_revision_failed"):
            updated["conflict_revision_failed"] = True
            if revision_state.get("failure_reason"):
                updated["conflict_revision_failure_reason"] = str(revision_state.get("failure_reason"))

        updated["needs_targeted_exploration"] = False
        updated["open_questions"] = [
            conflict["decision_point"]
            for conflict in updated["conflicts"]
            if conflict.get("status") != "verified"
        ]
        updated["has_conflict"] = bool(updated["conflicts"])
        return updated

    @staticmethod
    def format_resolved_summary(
        authoritative_facts: Sequence[str],
        verified_policies: Sequence[str],
        conflicts: Sequence[Dict[str, Any]],
        structured_anchor_text: str = "",
    ) -> str:
        lines = [RESOLVED_SUMMARY_MARKER, "", "## Authoritative facts"]
        lines.extend(f"- {fact}" for fact in authoritative_facts)
        lines.extend(["", "## Verified policies"])
        lines.extend(f"- {policy}" for policy in verified_policies)
        lines.extend([
            "",
            "## Conflicts",
            CONFLICT_LEDGER_MARKER,
            json.dumps(list(conflicts), ensure_ascii=False, indent=2),
        ])
        if structured_anchor_text:
            lines.extend(["", structured_anchor_text])
        return "\n".join(lines).strip()

    @classmethod
    def blocked_conflicts_for_anchor_prompt(
        cls,
        conflicts: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        blocked: List[Dict[str, Any]] = []
        for conflict in cls.normalize_conflicts(conflicts):
            if conflict.get("status") == "verified":
                continue
            blocked_options: List[Dict[str, str]] = []
            for option in conflict.get("options") or []:
                if not isinstance(option, dict):
                    continue
                blocked_options.append({
                    "option_id": str(option.get("option_id") or "").strip(),
                    "policy": str(option.get("policy") or "").strip(),
                })
            blocked.append({
                "conflict_id": conflict.get("conflict_id", ""),
                "decision_point": conflict.get("decision_point", ""),
                "status": conflict.get("status", "unresolved"),
                "provisional_choice": conflict.get("provisional_choice", ""),
                "options": blocked_options,
            })
        return blocked

    def finalize_summary(
        self,
        *,
        task_id: str = "",
        task_instruction: str,
        merge_state: Dict[str, Any],
    ) -> str:
        if merge_state.get("conflict_revision_failed"):
            fallback = merge_state.get("merged_summary") or ""
            return (
                "# UNRESOLVED EXPLORATION SUMMARY #\n"
                "Conflict revision was unavailable, so structured evidence anchors are intentionally omitted.\n"
                "Use the local summaries below as provisional evidence only, and verify disputed decisions with code before trusting them.\n\n"
                + fallback
            )

        authoritative_facts = self.as_string_list(merge_state.get("authoritative_facts"))
        verified_policies = self.as_string_list(merge_state.get("verified_policies"))
        conflicts = self.normalize_conflicts(merge_state.get("conflicts"))
        trusted_evidence = {
            "authoritative_facts": authoritative_facts,
            "verified_policies": verified_policies,
            "blocked_conflicts_not_trusted": self.blocked_conflicts_for_anchor_prompt(conflicts),
        }
        if not authoritative_facts and not verified_policies:
            return self.format_resolved_summary([], [], conflicts)
        prompt = EXPLORATION_FINAL_SUMMARY_PROMPT.format(
            trusted_evidence=json.dumps(trusted_evidence, ensure_ascii=False, indent=2),
        )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2500,
        }
        if self.llm_call_accepts_metadata:
            request["task_id"] = task_id
            request["llm_phase"] = "exploration_final_summary"
        status, response, _, _ = self.llm_call(request)
        if not status or not response:
            return self.format_resolved_summary(
                authoritative_facts,
                verified_policies,
                conflicts,
            )

        anchor_response = response.strip()
        if STRUCTURED_ANCHORS_MARKER not in anchor_response:
            anchor_response = STRUCTURED_ANCHORS_MARKER + "\n" + anchor_response
        anchors = self.extract_structured_anchors(anchor_response)
        return self.format_resolved_summary(
            authoritative_facts,
            verified_policies,
            conflicts,
            self.format_structured_anchors(anchors),
        )

    def build_retry_existing_evidence(
        self,
        *,
        task_id: str,
        task_exploration_summaries: Dict[str, List[str]],
        source_entries: Sequence[Dict[str, Any]],
    ) -> str:
        evidence_parts: List[str] = []
        for src in source_entries:
            summary = (src.get("exploration_summary") or "").strip()
            if summary:
                evidence_parts.append(
                    f"# EXPLORATION SUMMARY USED BY {src.get('step_id', 'source step')} #\n{summary}"
                )

        known = self.build_known_evidence(task_exploration_summaries, task_id)
        if known and known != NO_EVIDENCE:
            evidence_parts.append("# RECENT KNOWN EVIDENCE #\n" + known)
        return "\n\n".join(evidence_parts)

    def build_retry_source_context(
        self,
        source_entries: Sequence[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        for src in source_entries:
            step_id = src.get("step_id", "source step")
            summary = (src.get("exploration_summary") or "").strip()
            if summary:
                parts.append(
                    f"# ORIGINAL EXPLORATION SUMMARY USED BEFORE RETRYING {step_id} #\n"
                    + self.compact_text(summary, 10000)
                )
        return "\n\n".join(parts).strip()

    @staticmethod
    def build_effective_summary(current_summary: str, source_retry_context: str = "") -> str:
        if not source_retry_context:
            return current_summary or ""
        effective_parts = [
            "# SOURCE STEP ORIGINAL EXPLORATION CONTEXT #",
            source_retry_context,
        ]
        if current_summary:
            effective_parts.extend([
                "# CURRENT RETRY EXPLORATION SUMMARY #",
                current_summary,
            ])
        return "\n\n".join(effective_parts)

    @staticmethod
    def extract_structured_anchors(summary: str) -> Dict[str, Any]:
        text = summary or ""
        if STRUCTURED_ANCHORS_MARKER not in text:
            return {}

        merged: Dict[str, Any] = {}
        for family in STRUCTURED_ANCHOR_FAMILIES:
            merged[family] = []

        for tail in text.split(STRUCTURED_ANCHORS_MARKER)[1:]:
            payload = parse_first_json_object(tail)
            if not payload:
                continue
            for key, value in payload.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
                elif key not in merged:
                    merged[key] = value

        for key, value in list(merged.items()):
            if isinstance(value, list):
                merged[key] = ExplorationEvidenceManager.dedupe_anchor_items(value)
            elif value in ({}, None, ""):
                merged.pop(key, None)

        return {key: value for key, value in merged.items() if value}

    @staticmethod
    def extract_conflicts(summary: str) -> List[Dict[str, Any]]:
        text = summary or ""
        if CONFLICT_LEDGER_MARKER not in text:
            return []
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for tail in text.split(CONFLICT_LEDGER_MARKER)[1:]:
            try:
                payload, _ = json.JSONDecoder().raw_decode(tail.lstrip())
            except Exception:
                continue
            for conflict in ExplorationEvidenceManager.normalize_conflicts(payload):
                conflict_id = str(conflict.get("conflict_id") or "").strip()
                if not conflict_id:
                    continue
                if conflict_id not in merged:
                    order.append(conflict_id)
                merged[conflict_id] = conflict
        return [merged[conflict_id] for conflict_id in order]

    @staticmethod
    def dedupe_anchor_items(items: Sequence[Any]) -> List[Any]:
        kept: List[Any] = []
        seen: set[str] = set()
        for item in reversed(list(items or [])):
            if isinstance(item, dict):
                key = str(item.get("anchor_id") or json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                key = str(item)
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)
        return list(reversed(kept))

    @staticmethod
    def format_structured_anchors(anchors: Dict[str, Any]) -> str:
        if not anchors:
            return ""
        return (
            STRUCTURED_ANCHORS_MARKER
            + "\n"
            + json.dumps(anchors, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def extract_resolved_sections(summary: str) -> Dict[str, List[str]]:
        sections = {
            "authoritative_facts": [],
            "verified_policies": [],
        }
        text = summary or ""
        heading_map = {
            "authoritative facts": "authoritative_facts",
            "verified policies": "verified_policies",
        }
        current_key = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = line.strip("#:*- ").lower()
            if normalized in heading_map:
                current_key = heading_map[normalized]
                continue
            if line.startswith((STRUCTURED_ANCHORS_MARKER, CONFLICT_LEDGER_MARKER)):
                current_key = ""
                continue
            if line.startswith("#"):
                current_key = ""
                continue
            if current_key:
                sections[current_key].append(line)
        return sections

    @staticmethod
    def format_trusted_evidence(sections: Dict[str, List[str]]) -> str:
        facts = ExplorationEvidenceManager.dedupe_text_items(
            sections.get("authoritative_facts", [])
        )
        policies = ExplorationEvidenceManager.dedupe_text_items(
            sections.get("verified_policies", [])
        )
        parts: List[str] = []
        if facts:
            parts.append("# TRUSTED AUTHORITATIVE FACTS #")
            parts.extend(f"- {fact}" for fact in facts)
        if policies:
            if parts:
                parts.append("")
            parts.append("# TRUSTED VERIFIED POLICIES #")
            parts.extend(f"- {policy}" for policy in policies)
        return "\n".join(parts).strip()

    @staticmethod
    def dedupe_text_items(items: Sequence[str]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for item in items or []:
            text = str(item or "").strip()
            text = re.sub(r"^[-*]\s+", "", text).strip()
            if not text:
                continue
            key = re.sub(r"\s+", " ", text).lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    def build_bundle(
        self,
        *,
        task_id: str,
        task_exploration_summaries: Dict[str, List[str]],
        current_summary: str,
        source_retry_context: str = "",
    ) -> ExplorationEvidenceBundle:
        known = self.build_known_evidence(task_exploration_summaries, task_id)
        effective = self.build_effective_summary(current_summary, source_retry_context)
        sections = self.extract_resolved_sections(effective)
        anchors = self.extract_structured_anchors(effective)
        conflicts = self.extract_conflicts(effective)
        scoring = self.format_trusted_evidence(sections)
        authoritative_facts = self.dedupe_text_items(sections.get("authoritative_facts", []))
        verified_policies = self.dedupe_text_items(sections.get("verified_policies", []))
        return ExplorationEvidenceBundle(
            known_evidence=known,
            current_summary=current_summary or "",
            source_retry_context=source_retry_context or "",
            effective_summary=effective,
            scoring_evidence_text=scoring,
            structured_anchor_text=self.format_structured_anchors(anchors),
            structured_anchors=anchors,
            authoritative_facts=authoritative_facts,
            verified_policies=verified_policies,
            conflicts=conflicts,
        )
