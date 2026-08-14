"""Lightweight per-task cost metrics collection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional


def _usage_value(usage: Any, *names: str) -> int:
    current = usage
    for name in names:
        if current is None:
            return 0
        if isinstance(current, dict):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    if current is None:
        return 0
    try:
        return int(current)
    except Exception:
        return 0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "model_dump", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def canonical_usage_fields(usage: Any) -> Dict[str, int]:
    input_tokens = (
        _usage_value(usage, "prompt_tokens")
        or _usage_value(usage, "input_tokens")
    )
    output_tokens = (
        _usage_value(usage, "completion_tokens")
        or _usage_value(usage, "output_tokens")
    )
    cache_creation_input_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_input_tokens = _usage_value(usage, "cache_read_input_tokens")
    total_tokens = (
        _usage_value(usage, "total_tokens")
        or input_tokens + output_tokens + cache_creation_input_tokens + cache_read_input_tokens
    )
    reasoning_tokens = (
        _usage_value(usage, "completion_tokens_details", "reasoning_tokens")
        or _usage_value(usage, "output_tokens_details", "reasoning_tokens")
        or _usage_value(usage, "reasoning_tokens")
        or _usage_value(usage, "reasoning_output_tokens")
    )
    cached_input_tokens = (
        _usage_value(usage, "prompt_tokens_details", "cached_tokens")
        or _usage_value(usage, "input_tokens_details", "cached_tokens")
        or _usage_value(usage, "cached_input_tokens")
        or cache_read_input_tokens
        or _usage_value(usage, "cached_tokens")
    )
    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_input_tokens,
    }


def _empty_bucket() -> Dict[str, Any]:
    return {
        "llm_call_count": 0,
        "usage_event_count": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "duration_sec": 0.0,
    }


def _add_to_bucket(
    bucket: Dict[str, Any],
    usage_fields: Dict[str, int],
    *,
    has_usage: bool,
    duration_sec: float,
) -> None:
    bucket["llm_call_count"] = int(bucket.get("llm_call_count", 0)) + 1
    if has_usage:
        bucket["usage_event_count"] = int(bucket.get("usage_event_count", 0)) + 1
    for key in (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
    ):
        bucket[key] = int(bucket.get(key, 0)) + int(usage_fields.get(key, 0))
    bucket["duration_sec"] = round(
        float(bucket.get("duration_sec", 0.0)) + max(0.0, float(duration_sec or 0.0)),
        3,
    )


@dataclass
class TaskCostRecord:
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    llm_call_count: int = 0
    usage_event_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    step_count: int = 0
    turn_count: int = 0
    code_execution_count: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    by_phase: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_step: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        end_time = self.end_time if self.end_time is not None else time.time()
        return {
            "task_id": self.task_id,
            "total_wall_time_sec": round(max(0.0, end_time - self.start_time), 3),
            "llm_call_count": self.llm_call_count,
            "usage_event_count": self.usage_event_count,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "step_count": self.step_count,
            "turn_count": self.turn_count,
            "code_execution_count": self.code_execution_count,
            "by_phase": self.by_phase,
            "by_step": self.by_step,
        }


class TaskCostTracker:
    """Collects metrics that are comparable across agent implementations."""

    def __init__(self) -> None:
        self.records: Dict[str, TaskCostRecord] = {}

    def reset(self) -> None:
        self.records = {}

    def start_task(self, task_id: str) -> None:
        self.records[task_id] = TaskCostRecord(task_id=task_id, start_time=time.time())

    def finish_task(self, task_id: str) -> None:
        record = self.records.get(task_id)
        if record and record.end_time is None:
            record.end_time = time.time()

    def record_llm_call(
        self,
        task_id: str,
        usage: Any = None,
        *,
        count_as_turn: bool = False,
        phase: str = "unknown",
        model: str = "",
        step_id: str = "",
        step_index: Optional[int] = None,
        attempt_index: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        if not task_id:
            return
        record = self.records.setdefault(
            task_id,
            TaskCostRecord(task_id=task_id, start_time=time.time()),
        )
        record.llm_call_count += 1
        if count_as_turn:
            record.turn_count += 1

        usage_fields = canonical_usage_fields(usage)
        raw_usage = _jsonable(usage)
        has_usage = raw_usage not in (None, {}, "")
        if has_usage:
            record.usage_event_count += 1

        input_tokens = usage_fields["input_tokens"]
        output_tokens = usage_fields["output_tokens"]
        total_tokens = usage_fields["total_tokens"]
        reasoning_tokens = usage_fields["reasoning_tokens"]
        cached_input_tokens = usage_fields["cached_input_tokens"]

        record.input_tokens += input_tokens
        record.output_tokens += output_tokens
        record.total_tokens += total_tokens
        record.reasoning_tokens += reasoning_tokens
        record.cached_input_tokens += cached_input_tokens

        call_start = start_time if start_time is not None else time.time()
        call_end = end_time if end_time is not None else call_start
        duration_sec = round(max(0.0, call_end - call_start), 3)
        phase_key = phase or "unknown"
        step_key = step_id or (
            str(step_index) if step_index is not None else "unassigned"
        )

        _add_to_bucket(
            record.by_phase.setdefault(phase_key, _empty_bucket()),
            usage_fields,
            has_usage=has_usage,
            duration_sec=duration_sec,
        )
        _add_to_bucket(
            record.by_step.setdefault(step_key, _empty_bucket()),
            usage_fields,
            has_usage=has_usage,
            duration_sec=duration_sec,
        )

        event = {
            "event_index": len(record.events),
            "task_id": task_id,
            "step_id": step_id,
            "step_index": step_index,
            "attempt_index": attempt_index,
            "phase": phase_key,
            "model": model,
            "start_time": call_start,
            "end_time": call_end,
            "duration_sec": duration_sec,
            "count_as_turn": count_as_turn,
            **usage_fields,
            "raw_usage": raw_usage,
        }
        record.events.append(event)

    def record_step(self, task_id: str, *, is_retry_step: bool = False) -> None:
        if not task_id or is_retry_step:
            return
        record = self.records.setdefault(
            task_id,
            TaskCostRecord(task_id=task_id, start_time=time.time()),
        )
        record.step_count += 1

    def record_code_execution(self, task_id: str, action_type: str) -> None:
        if not task_id or action_type == "Terminate":
            return
        record = self.records.setdefault(
            task_id,
            TaskCostRecord(task_id=task_id, start_time=time.time()),
        )
        record.code_execution_count += 1

    def get(self, task_id: str) -> Dict[str, Any]:
        record = self.records.get(task_id)
        return record.to_dict() if record else {}

    def get_events(self, task_id: str) -> List[Dict[str, Any]]:
        record = self.records.get(task_id)
        return list(record.events) if record else []

    def all_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {task_id: record.to_dict() for task_id, record in self.records.items()}

    def all_events(self) -> Dict[str, List[Dict[str, Any]]]:
        return {task_id: list(record.events) for task_id, record in self.records.items()}

    def write(
        self,
        jsonl_path: str,
        json_path: str,
        events_jsonl_path: Optional[str] = None,
    ) -> None:
        metrics = self.all_metrics()
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for task_id in sorted(metrics):
                f.write(json.dumps(metrics[task_id], ensure_ascii=False) + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        if events_jsonl_path:
            with open(events_jsonl_path, "w", encoding="utf-8") as f:
                for task_id in sorted(self.records):
                    for event in self.records[task_id].events:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
