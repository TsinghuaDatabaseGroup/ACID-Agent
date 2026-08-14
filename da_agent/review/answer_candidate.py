"""Task-level answer candidate tracking and comparison prompts.

This module catches verified-looking final answers without terminating early.
If a later step produces a different final answer, it builds a review prompt
that asks the agent to compare the competing computation paths side by side.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


ANSWER_CANDIDATE_COMPARISON_PROMPT_BLOCK = """# CANDIDATE ANSWER COMPARISON #
Source task: {task_id}
New candidate step: {new_step_id}
Prior candidate step: {prior_step_id}

Two different final-answer candidates have appeared during this task. Do not choose based on recency. Compare the candidate-specific evidence summaries, reasoning paths, and action/method choices before relying on either answer.

# ORIGINAL TASK #
{task_instruction}

# PRIOR CANDIDATE #
Answer: {prior_answer}
Source step: {prior_step_id}

Candidate-specific context:
{prior_excerpts}

# NEW CANDIDATE #
Answer: {new_answer}
Source step: {new_step_id}

Candidate-specific context:
{new_excerpts}

# REQUIRED COMPARISON #
- Do not rerun candidate code or recompute both answers unless the task cannot proceed without a missing fact.
- Do not judge execution success; execution and retry checks already happened before candidate registration.
- Use only the original task and the two candidate-specific contexts above to compare the candidates. Do not use structured anchors, global/revised summaries, or later review context as deciding evidence unless the same fact appears in a candidate-specific context above.
- Identify the exact decision point(s) in the two paths that make the answers differ.
- Compare those decisions against the original task wording and each candidate's own source-step exploration summary.
- Focus on source scope, filter scope, field semantics, aggregation grain, missingness handling, formula, threshold, and output policy when relevant.
- Decide which candidate path should be trusted, or state the exact unresolved decision if the prompt context is insufficient.
- Continue solving the original task using the better-supported path.
"""


@dataclass
class AnswerCandidateResult:
    comparison_scheduled: bool = False
    feedback: Optional[Dict[str, Any]] = None
    candidate: Optional[Dict[str, Any]] = None
    merged_candidate: Optional[Dict[str, Any]] = None
    log_payload: Optional[Dict[str, Any]] = None


class AnswerCandidateTracker:
    def __init__(
        self,
        *,
        compact_text: Optional[Callable[[str, int], str]] = None,
        llm_generate: Optional[Callable[..., str]] = None,
    ) -> None:
        self.compact_text = compact_text or self._default_compact_text
        self.llm_generate = llm_generate
        self.candidates_by_task: Dict[str, List[Dict[str, Any]]] = {}
        self.comparison_keys_by_task: Dict[str, set] = {}
        self.unresolved_comparison_keys_by_task: Dict[str, set] = {}

    def reset(self) -> None:
        self.candidates_by_task = {}
        self.comparison_keys_by_task = {}
        self.unresolved_comparison_keys_by_task = {}

    def setup_task(self, task_id: str) -> None:
        self.candidates_by_task[task_id] = []
        self.comparison_keys_by_task[task_id] = set()
        self.unresolved_comparison_keys_by_task[task_id] = set()

    def register_step(
        self,
        *,
        task_id: str,
        task_instruction: str,
        step_id: str,
        thought: str,
        action_text: str,
        action_output: Optional[str],
        code: str,
        observation: str,
        objective_confidence: Dict[str, Any],
        should_retry: bool,
        is_retry_step: bool,
        exploration_summary: str = "",
        comparison_candidate_ids: Optional[List[str]] = None,
        comparison_candidate_signatures: Optional[List[str]] = None,
    ) -> AnswerCandidateResult:
        if should_retry:
            return AnswerCandidateResult()

        candidate = self._build_answer_candidate(
            task_id=task_id,
            task_instruction=task_instruction,
            step_id=step_id,
            thought=thought,
            action_text=action_text,
            action_output=action_output,
            code=code,
            observation=observation,
            exploration_summary=exploration_summary,
            objective_confidence=objective_confidence,
            is_retry_step=is_retry_step,
        )
        if candidate is None:
            return AnswerCandidateResult()

        candidates = self.candidates_by_task.setdefault(task_id, [])
        comparison_context = bool(comparison_candidate_ids or comparison_candidate_signatures)
        different_candidates: List[Dict[str, Any]] = []
        for existing in candidates:
            if not self._candidate_answers_equivalent(
                task_id=task_id,
                task_instruction=task_instruction,
                prior_candidate=existing,
                new_candidate=candidate,
            ):
                different_candidates.append(existing)
            else:
                self._merge_same_answer_candidate(
                    existing,
                    candidate,
                )
                if comparison_context:
                    self._apply_comparison_result(
                        task_id=task_id,
                        winner=existing,
                        comparison_candidate_ids=comparison_candidate_ids or [],
                        comparison_candidate_signatures=comparison_candidate_signatures or [],
                    )
                return AnswerCandidateResult(
                    merged_candidate=existing,
                    log_payload={
                        "event": "merged",
                        "task_id": task_id,
                        "step_id": step_id,
                        "answer": existing.get("answer"),
                    },
                )

        candidates.append(candidate)
        if comparison_context:
            self._apply_comparison_result(
                task_id=task_id,
                winner=candidate,
                comparison_candidate_ids=comparison_candidate_ids or [],
                comparison_candidate_signatures=comparison_candidate_signatures or [],
            )
        result = AnswerCandidateResult(
            candidate=candidate,
            log_payload={
                "event": "registered",
                "task_id": task_id,
                "step_id": step_id,
                "answer": candidate.get("answer"),
            },
        )
        if not different_candidates:
            return result

        prior = different_candidates[-1]
        feedback = self._build_comparison_feedback(
            task_id,
            task_instruction,
            prior,
            candidate,
        )
        if feedback is None:
            return result
        result.comparison_scheduled = True
        result.feedback = feedback
        result.log_payload = {
            "event": "comparison_scheduled",
            "task_id": task_id,
            "prior_answer": prior.get("answer"),
            "prior_step": prior.get("source_step"),
            "new_answer": candidate.get("answer"),
            "new_step": candidate.get("source_step"),
        }
        return result

    @staticmethod
    def _default_compact_text(text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        head_len = limit // 2
        tail_len = limit - head_len
        head = text[:head_len]
        tail = text[-tail_len:]
        return head + "\n... (middle truncated) ...\n" + tail

    @staticmethod
    def _extract_answer_from_action_output(action_output: Optional[str], action_text: str) -> str:
        direct = (action_output or "").strip()
        if direct and direct.upper() != "FAIL":
            return direct
        text = action_text or ""
        match = re.search(r"Terminate\s*\(\s*output\s*=\s*(['\"])(.*?)\1\s*\)", text, re.S)
        if not match:
            return ""
        answer = (match.group(2) or "").strip()
        if not answer or answer.upper() == "FAIL":
            return ""
        return answer

    @staticmethod
    def _parse_llm_json_object(response: str) -> Optional[Dict[str, Any]]:
        text = (response or "").strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "y", "1"}
        return bool(value)

    def _generate_with_context(
        self,
        prompt: str,
        *,
        task_id: str,
        phase: str,
    ) -> str:
        if not self.llm_generate:
            return ""
        try:
            return self.llm_generate(prompt, task_id=task_id, phase=phase)
        except TypeError:
            return self.llm_generate(prompt)

    def _llm_extract_answer_candidate(
        self,
        *,
        task_id: str,
        task_instruction: str,
        thought: str,
        action_text: str,
        action_output: Optional[str],
        code: str,
        observation: str,
        rule_answer: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_generate:
            return None
        prompt = (
            "You are judging whether this completed agent step proposes an answer candidate "
            "for the original task. Extract only what the step is proposing as the answer; "
            "do not solve the task yourself.\n\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            '  "has_candidate": true or false,\n'
            '  "answer": "candidate answer string, or empty string",\n'
            '  "reason": "brief extraction reason"\n'
            "}\n\n"
            "Rules:\n"
            "- If the action is Terminate(output=...), that output is the candidate unless it is FAIL or empty.\n"
            "- Otherwise, use the observation/action to identify the value the agent is presenting as the final answer for the original task.\n"
            "- Ignore values explicitly described as alternatives, previous wrong answers, intermediate counts, sanity checks, or failed paths.\n"
            "- If there is no proposed final-answer candidate, set has_candidate=false and answer=\"\".\n"
            "- Preserve formatting needed by the task, such as percent signs, decimal places, units, or file paths.\n\n"
            f"Original task:\n{task_instruction}\n\n"
            f"Thought:\n{thought or ''}\n\n"
            f"Action:\n{action_text or ''}\n\n"
            f"Explicit action output, if any:\n{action_output or ''}\n\n"
            f"Code excerpt:\n{code or ''}\n\n"
            f"Observation digest:\n{self.compact_text(observation, 3000)}\n\n"
            f"Legacy rule candidate, if any:\n{rule_answer or '(none)'}\n"
        )
        try:
            response = self._generate_with_context(
                prompt,
                task_id=task_id,
                phase="answer_candidate_extraction",
            )
        except Exception:
            return None
        parsed = self._parse_llm_json_object(response or "")
        if parsed is None:
            return None
        parsed["_llm_extraction_succeeded"] = True
        return parsed

    @staticmethod
    def _answer_signature(answer: str) -> str:
        return re.sub(r"\s+", " ", str(answer or "")).strip()

    def _llm_judge_candidate_equivalence(
        self,
        *,
        task_id: str,
        task_instruction: str,
        prior_candidate: Dict[str, Any],
        new_candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_generate:
            return None
        prompt = (
            "You are judging whether two answer candidates for the same original task "
            "represent the same final answer. Do not solve the task yourself.\n\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            '  "same_answer": true or false,\n'
            '  "preferred_answer": "best final answer string if they are the same, otherwise empty",\n'
            '  "reason": "brief reason"\n'
            "}\n\n"
            "Rules:\n"
            "- Consider the original task when deciding whether formatting, units, rounding, labels, ordering, or structure matter.\n"
            "- If the candidates differ only in harmless formatting, set same_answer=true and choose the preferred final format.\n"
            "- If ordering, units, rounding, labels, set membership, or structure could change the required answer, set same_answer=false.\n"
            "- If uncertain, set same_answer=false so the agent will compare the paths side by side.\n\n"
            f"# ORIGINAL TASK #\n{task_instruction}\n\n"
            f"# PRIOR CANDIDATE ANSWER #\n{prior_candidate.get('answer', '')}\n\n"
            f"# PRIOR CANDIDATE CONTEXT #\n{self._format_candidate_excerpts(prior_candidate)}\n\n"
            f"# NEW CANDIDATE ANSWER #\n{new_candidate.get('answer', '')}\n\n"
            f"# NEW CANDIDATE CONTEXT #\n{self._format_candidate_excerpts(new_candidate)}\n\n"
            "JSON:"
        )
        try:
            response = self._generate_with_context(
                prompt,
                task_id=task_id,
                phase="answer_candidate_equivalence",
            )
        except Exception:
            return None
        return self._parse_llm_json_object(response or "")

    def _candidate_answers_equivalent(
        self,
        *,
        task_id: str,
        task_instruction: str,
        prior_candidate: Dict[str, Any],
        new_candidate: Dict[str, Any],
    ) -> bool:
        prior_signature = self._answer_signature(prior_candidate.get("answer", ""))
        new_signature = self._answer_signature(new_candidate.get("answer", ""))
        if prior_signature and prior_signature == new_signature:
            new_candidate["equivalence_judgment"] = {
                "same_answer": True,
                "judge": "exact_signature",
                "reason": "answer strings match after whitespace cleanup",
            }
            return True

        judgment = self._llm_judge_candidate_equivalence(
            task_id=task_id,
            task_instruction=task_instruction,
            prior_candidate=prior_candidate,
            new_candidate=new_candidate,
        )
        if judgment is None:
            new_candidate["equivalence_judgment"] = {
                "same_answer": False,
                "judge": "unavailable",
                "reason": "candidate equivalence judge unavailable",
            }
            return False

        same = self._as_bool(judgment.get("same_answer"))
        judgment["judge"] = "llm"
        new_candidate["equivalence_judgment"] = judgment
        if not same:
            return False

        preferred = str(judgment.get("preferred_answer") or "").strip()
        if preferred:
            new_candidate["answer"] = preferred
            new_candidate["answer_signature"] = self._answer_signature(preferred)
            new_candidate["candidate_id"] = (
                f"{new_candidate.get('source_step', '')}:"
                f"{new_candidate['answer_signature']}"
            )
        return True

    @staticmethod
    def _format_candidate_excerpts(candidate: Dict[str, Any]) -> str:
        excerpts = candidate.get("excerpts", {}) or {}
        parts = []
        for key, label in (
            ("exploration_summary", "Source-step exploration summary"),
            ("thought", "Thought"),
            ("action", "Action"),
        ):
            text = str(excerpts.get(key) or "").strip()
            parts.append(f"{label}:\n{text if text else '(none)'}")
        return "\n\n".join(parts)

    def _build_answer_candidate(
        self,
        *,
        task_id: str,
        task_instruction: str,
        step_id: str,
        thought: str,
        action_text: str,
        action_output: Optional[str],
        code: str,
        observation: str,
        exploration_summary: str,
        objective_confidence: Dict[str, Any],
        is_retry_step: bool,
    ) -> Optional[Dict[str, Any]]:
        _ = objective_confidence, is_retry_step
        rule_answer = self._extract_answer_from_action_output(action_output, action_text)
        llm_candidate = self._llm_extract_answer_candidate(
            task_id=task_id,
            task_instruction=task_instruction,
            thought=thought,
            action_text=action_text,
            action_output=action_output,
            code=code,
            observation=observation,
            rule_answer=rule_answer,
        )
        if llm_candidate is not None:
            if not self._as_bool(llm_candidate.get("has_candidate")):
                return None
            answer = str(llm_candidate.get("answer") or "").strip()
            if not answer:
                return None
        else:
            answer = rule_answer
        if not answer:
            return None

        answer_signature = self._answer_signature(answer)
        return {
            "candidate_id": f"{step_id}:{answer_signature}",
            "task_id": task_id,
            "answer": answer,
            "answer_signature": answer_signature,
            "candidate_id_history": [f"{step_id}:{answer_signature}"],
            "source_step": step_id,
            "excerpts": {
                "exploration_summary": exploration_summary or "",
                "thought": thought or "",
                "action": action_text or "",
            },
        }

    def _merge_same_answer_candidate(
        self,
        existing: Dict[str, Any],
        new_candidate: Dict[str, Any],
    ) -> None:
        history = list(existing.get("candidate_id_history") or [])
        for candidate_id in (
            existing.get("candidate_id"),
            new_candidate.get("candidate_id"),
            *(new_candidate.get("candidate_id_history") or []),
        ):
            if candidate_id and candidate_id not in history:
                history.append(candidate_id)

        for key in ("candidate_id", "answer", "answer_signature", "source_step", "excerpts", "equivalence_judgment"):
            existing[key] = new_candidate.get(key)
        existing["candidate_id_history"] = history

    @staticmethod
    def _comparison_key_from_signatures(signatures: List[str]) -> str:
        return "||".join(sorted(str(item or "") for item in signatures if str(item or "").strip()))

    @staticmethod
    def _candidate_matches_reference(
        candidate: Dict[str, Any],
        *,
        candidate_ids: set,
        signatures: set,
    ) -> bool:
        history = set(candidate.get("candidate_id_history") or [])
        if candidate.get("candidate_id"):
            history.add(candidate.get("candidate_id"))
        if history.intersection(candidate_ids):
            return True
        signature = str(candidate.get("answer_signature") or candidate.get("answer") or "")
        return bool(signature and signature in signatures)

    def _apply_comparison_result(
        self,
        *,
        task_id: str,
        winner: Dict[str, Any],
        comparison_candidate_ids: List[str],
        comparison_candidate_signatures: List[str],
    ) -> None:
        candidate_ids = {str(item) for item in comparison_candidate_ids or [] if str(item).strip()}
        signatures = {str(item) for item in comparison_candidate_signatures or [] if str(item).strip()}
        winner_signature = str(winner.get("answer_signature") or winner.get("answer") or "")
        for candidate in self.candidates_by_task.get(task_id, []):
            if candidate is winner:
                continue
            candidate_signature = str(candidate.get("answer_signature") or candidate.get("answer") or "")
            if candidate_signature == winner_signature:
                continue
            if not self._candidate_matches_reference(
                candidate,
                candidate_ids=candidate_ids,
                signatures=signatures,
            ):
                continue
            candidate["comparison_result"] = {
                "winner_answer": winner.get("answer", ""),
                "winner_candidate_id": winner.get("candidate_id", ""),
                "reason": "side-by-side comparison selected a different final answer",
            }

        comparison_key = self._comparison_key_from_signatures(list(signatures))
        if comparison_key:
            self.unresolved_comparison_keys_by_task.setdefault(task_id, set()).discard(comparison_key)

    def get_candidate_state(self, task_id: str) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "answer": candidate.get("answer", ""),
                    "answer_signature": candidate.get("answer_signature", ""),
                    "source_step": candidate.get("source_step", ""),
                    "comparison_result": candidate.get("comparison_result", {}),
                }
                for candidate in self.candidates_by_task.get(task_id, [])
            ],
            "unresolved_comparison_count": len(self.unresolved_comparison_keys_by_task.get(task_id, set())),
        }

    def _build_comparison_feedback(
        self,
        task_id: str,
        task_instruction: str,
        prior_candidate: Dict[str, Any],
        new_candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        left_key = prior_candidate.get("answer_signature") or prior_candidate.get("answer", "")
        right_key = new_candidate.get("answer_signature") or new_candidate.get("answer", "")
        comparison_key = "||".join(sorted([str(left_key), str(right_key)]))
        seen = self.comparison_keys_by_task.setdefault(task_id, set())
        if comparison_key in seen:
            return None
        seen.add(comparison_key)
        self.unresolved_comparison_keys_by_task.setdefault(task_id, set()).add(comparison_key)

        prompt_block = ANSWER_CANDIDATE_COMPARISON_PROMPT_BLOCK.format(
            task_id=task_id,
            task_instruction=task_instruction or "",
            new_step_id=new_candidate.get("source_step", ""),
            prior_step_id=prior_candidate.get("source_step", ""),
            prior_answer=prior_candidate.get("answer", ""),
            new_answer=new_candidate.get("answer", ""),
            prior_excerpts=self._format_candidate_excerpts(prior_candidate),
            new_excerpts=self._format_candidate_excerpts(new_candidate),
        )
        return {
            "retry_review": True,
            "step_id": new_candidate.get("source_step", ""),
            "prompt_block": prompt_block,
            "candidate_comparison": True,
            "candidate_ids": [
                prior_candidate.get("candidate_id", ""),
                new_candidate.get("candidate_id", ""),
            ],
            "candidate_signatures": [
                prior_candidate.get("answer_signature", ""),
                new_candidate.get("answer_signature", ""),
            ],
        }
