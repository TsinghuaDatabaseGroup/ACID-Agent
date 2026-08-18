"""Codex Baseline Module - CLI wrapper for benchmark execution.

This module adapts the DA-Agent benchmark wrapper pattern to Codex:
- Starts a Docker-backed task environment
- Generates Codex CLI configuration for an OpenAI-compatible provider
- Invokes `codex exec` on each benchmark task
- Parses Codex JSONL events into a normalized trajectory
- Captures final answer, execution metadata, and cost metrics

References:
- https://github.com/AgenticDataBench/AgenticDataBench/blob/main/testbed/da_agent/agent/codex.py
"""

import json
import logging
import os
import subprocess
import time
from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.utils.cost_metrics import canonical_usage_fields

logger = logging.getLogger("da_agent")

DEFAULT_TIME_OUT = 3600  # 60 minutes
MAX_OBS_LENGTH = 3000
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = os.getenv(
    "DEFAULT_OPENAI_COMPATIBLE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class PromptAgent:
    """PromptAgent wrapper that runs the OpenAI Codex CLI inside Docker."""

    def __init__(
        self,
        model="glm-5.2",
        model_type="openai-compatible",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=50,
    ):
        self.model = model
        self.max_steps = max_steps
        self.env = None
        self.trajectory = []
        self.work_dir = "/workspace"
        self.raw_output = ""
        self.raw_events = []
        self.event_timestamps = []
        self.current_task_id = None
        self.run_start_time = None
        self.run_end_time = None
        self.current_cost_metrics = {}
        self.cost_metrics = {}
        self.current_cost_events = []
        self.cost_metric_events = {}

    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.current_task_id = self.env.task_config.get("id")
        self.instruction = self.env.task_config["question"]
        self.trajectory = []
        self.raw_output = ""
        self.raw_events = []
        self.event_timestamps = []
        self.run_start_time = None
        self.run_end_time = None
        self.current_cost_metrics = {}
        self.current_cost_events = []

    def _build_task_prompt(self):
        task = self.instruction
        task += f"\n\nYou are working in the directory: {self.work_dir}."
        task += " All required data files are available in this directory."
        task += " Complete the task and ensure all output files are saved in this directory."

        image_file_names = self._get_image_file_names()
        if image_file_names:
            task += self._build_plotting_instructions(image_file_names)

        task += self._build_final_answer_instructions()
        return task

    def _build_final_answer_instructions(self):
        return """

### Final Answer Rule

At the very end, output only the final answer requested by the task.
Use the shortest exact format possible:
- number: 9.17
- boolean: True or False
- string: Wang
- list: ["A", "B", "C"]

Write the final answer on exactly one line using this format:
FINAL_ANSWER: <answer>

Do not include explanation, markdown, derivation, file paths, or extra text after FINAL_ANSWER unless the task explicitly asks for it.
"""

    def _get_image_file_names(self):
        image_file_names = []
        for post_process_f in self.env.post_process_func:
            def image_post_process(output_file_name):
                if output_file_name in self.env.task_config.get("output_file_name", []):
                    return output_file_name
                return None

            output_file_name = eval(post_process_f)
            if output_file_name:
                image_file_names.append(output_file_name)
        return image_file_names

    def _build_plotting_instructions(self, image_file_names):
        return f"""
### Plotting (REQUIRED)

If you create a matplotlib plot, you MUST call:

    from image import Plotprocess
    Plotprocess.plot_process(fig, "<image_file_name>")

Use ONLY these file names:
{", ".join(image_file_names)}

Rules:
- Call AFTER plotting is complete
- Call BEFORE saving the figure
- Use: fig = plt.gcf()
- Replace <image_file_name> with one from the list above

Example:
```python
from image import Plotprocess
import matplotlib.pyplot as plt

# plotting code ...

fig = plt.gcf()
Plotprocess.plot_process(fig, "{image_file_names[0]}")
```"""

    def _write_wrapper_script(self):
        wrapper_code = f'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import threading
import time

with open("{self.work_dir}/.task_prompt.txt") as f:
    prompt = f.read()

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
base_url = (
    os.environ.get("OPENAI_API_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or "{DEFAULT_OPENAI_COMPATIBLE_BASE_URL}"
)

if not api_key:
    print("Missing OPENAI_API_KEY or DASHSCOPE_API_KEY for Codex CLI.", file=sys.stderr)
    sys.exit(2)

os.environ["OPENAI_API_KEY"] = api_key

config_toml = f"""model_provider = "OpenAICompatible"

[model_providers.OpenAICompatible]
name = "OpenAICompatible"
base_url = "{{base_url}}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
"""

config_dir = os.path.expanduser("~/.codex")
os.makedirs(config_dir, exist_ok=True)
with open(os.path.join(config_dir, "config.toml"), "w") as f:
    f.write(config_toml)

MAX_RETRIES = 3
RETRY_BACKOFF_429 = 30
RETRY_BACKOFF_400 = 10

current_proc = None


def timeout_handler():
    global current_proc
    if current_proc:
        current_proc.terminate()
        kill_timer = threading.Timer(300, current_proc.kill)
        kill_timer.daemon = True
        kill_timer.start()


timer = threading.Timer({DEFAULT_TIME_OUT}, timeout_handler)
timer.daemon = True
timer.start()


def run_codex(cmd_args):
    global current_proc
    cmd = ["stdbuf", "-oL"] + cmd_args
    current_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    last_line = ""
    for line in iter(current_proc.stdout.readline, b""):
        decoded = line.decode("utf-8", errors="ignore")
        sys.stdout.write(decoded)
        sys.stdout.flush()
        last_line = decoded.strip()

    stderr = current_proc.stderr.read()
    if stderr:
        stderr_text = stderr.decode("utf-8", errors="ignore").strip()
        if stderr_text:
            last_line = stderr_text.split("\\n")[-1]
            sys.stderr.write(stderr_text + "\\n")
            sys.stderr.flush()

    current_proc.wait()
    return current_proc.returncode, last_line


def is_retryable_error(last_line):
    try:
        entry = json.loads(last_line)
        if entry.get("type") == "turn.failed":
            error_msg = json.dumps(entry.get("error", {{}}))
            if "429" in error_msg:
                return "429"
            if "400" in error_msg or "InvalidParameter" in error_msg:
                return "400"
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def codex_exec_help():
    try:
        proc = subprocess.run(
            ["codex", "exec", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return proc.stdout or ""
    except Exception:
        return ""


help_text = codex_exec_help()
cmd_args = [
    "codex",
    "exec",
    "--model",
    "{self.model}",
    "--sandbox",
    "danger-full-access",
    "--skip-git-repo-check",
    "--json",
]

if "--ask-for-approval" in help_text:
    cmd_args.extend(["--ask-for-approval", "never"])
elif "--approval-policy" in help_text:
    cmd_args.extend(["--approval-policy", "never"])

cmd_args.append(prompt)

exit_code, last_line = run_codex(cmd_args)

for attempt in range(MAX_RETRIES):
    error_type = is_retryable_error(last_line)
    if not error_type:
        break

    backoff = RETRY_BACKOFF_429 if error_type == "429" else RETRY_BACKOFF_400
    print(
        f"[RETRY] turn.failed ({{error_type}}), waiting {{backoff}}s before retry {{attempt + 1}}/{{MAX_RETRIES}}...",
        flush=True,
    )
    time.sleep(backoff)
    exit_code, last_line = run_codex(cmd_args)

timer.cancel()
sys.exit(exit_code)
'''
        wrapper_path = os.path.join(self.env.mnt_dir, ".run_codex.py")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)

    def run(self):
        assert self.env is not None, "Environment is not set."

        task_prompt = self._build_task_prompt()
        container_name = self.env.container.name

        task_path = os.path.join(self.env.mnt_dir, ".task_prompt.txt")
        with open(task_path, "w") as f:
            f.write(task_prompt)
        self._write_wrapper_script()

        self.run_start_time = time.time()
        self.run_end_time = None
        process = subprocess.Popen(
            ["docker", "exec", str(container_name), "python3", f"{self.work_dir}/.run_codex.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        output_lines = []
        self.event_timestamps = []
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    decoded = line.decode("utf-8", errors="ignore")
                    output_lines.append(decoded)
                    self.event_timestamps.append(time.time())
                    logger.debug("Codex: %s", decoded.strip())
        except Exception as e:
            process.kill()
            logger.error("Error running Codex: %s", e)
            self.raw_output = "".join(output_lines)
            self.run_end_time = time.time()
            self._parse_trajectory()
            self._finalize_cost_metrics()
            return False, f"Error: {e}"

        self.raw_output = "".join(output_lines)
        exit_code = process.returncode
        self.run_end_time = time.time()

        self._parse_trajectory()
        self._finalize_cost_metrics()

        normal_end_types = {"result", "error", "raw", "turn.failed"}
        if exit_code == 0:
            last_type = self.trajectory[-1].get("type") if self.trajectory else None
            if last_type in normal_end_types:
                final_message = self._extract_final_assistant_message()
                return True, final_message or "Task completed"
            return False, f"Agent stopped without turn completion (last_type={last_type})"
        return False, f"Agent exited with code {exit_code}"

    def _extract_final_assistant_message(self):
        for entry in reversed(self.trajectory):
            if entry.get("type") != "assistant":
                continue
            if entry.get("code_action"):
                continue
            content = (entry.get("content") or "").strip()
            if content:
                return content
        return ""

    def _parse_trajectory(self):
        self.trajectory = []
        self.raw_events = []

        lines = self.raw_output.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            timing = None
            if i < len(self.event_timestamps):
                ts = self.event_timestamps[i]
                prev_ts = self.event_timestamps[i - 1] if i > 0 else ts
                timing = {
                    "start_time": prev_ts,
                    "end_time": ts,
                    "duration": ts - prev_ts,
                }
            try:
                entry = json.loads(line)
                raw_entry = dict(entry) if isinstance(entry, dict) else {"type": "raw", "content": entry}
                if timing:
                    raw_entry["timing"] = timing
                self.raw_events.append(raw_entry)
                normalized = self._normalize_jsonl_entry(entry)
                if timing:
                    normalized["timing"] = timing
                if normalized.get("type") not in ("thread_started", "turn_started"):
                    self.trajectory.append(normalized)
            except json.JSONDecodeError:
                step = {"type": "raw", "content": line}
                if timing:
                    step["timing"] = timing
                self.raw_events.append(dict(step))
                self.trajectory.append(step)

    def _normalize_jsonl_entry(self, entry):
        if not isinstance(entry, dict):
            return {"type": "raw", "content": str(entry)}

        event_type = entry.get("type", "unknown")

        if event_type == "thread.started":
            return {"type": "thread_started"}

        if event_type == "turn.started":
            return {"type": "turn_started"}

        if event_type == "turn.completed":
            result = {"type": "result"}
            usage = entry.get("usage", {})
            if usage:
                result["usage"] = usage
            return result

        if event_type == "item.completed":
            item = entry.get("item", {})
            item_type = item.get("type", "unknown")

            if item_type == "agent_message":
                text = item.get("text", "")
                return {"type": "assistant", "content": text}

            if item_type == "command_execution":
                cmd = item.get("command", "")
                output = item.get("aggregated_output", "")
                exit_code = item.get("exit_code")
                if len(output) > MAX_OBS_LENGTH:
                    output = output[:MAX_OBS_LENGTH] + f"\n... (truncated, original {len(output)} chars)"
                result = {
                    "type": "assistant",
                    "code_action": cmd,
                    "observations": f"Execution logs:\n{output}",
                }
                if exit_code is not None:
                    result["exit_code"] = exit_code
                return result

            if item_type == "todo_list":
                items = item.get("items", [])
                return {"type": "assistant", "content": f"Todo: {json.dumps(items)}"}

            return {"type": item_type, "content": json.dumps(item)}

        if event_type == "item.started":
            item = entry.get("item", {})
            return {"type": "item_started", "content": json.dumps(item)}

        if event_type == "error":
            return {"type": "error", "content": entry.get("message", "")}

        return {"type": event_type, "content": json.dumps(entry)}

    @staticmethod
    def _usage_int(usage, *paths):
        if not isinstance(usage, dict):
            return 0
        for path in paths:
            cursor = usage
            for key in path:
                if not isinstance(cursor, dict) or key not in cursor:
                    cursor = None
                    break
                cursor = cursor[key]
            if isinstance(cursor, (int, float)):
                return int(cursor)
        return 0

    def _compute_cost_metrics(self):
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        cached_input_tokens = 0

        turn_count = 0
        usage_event_count = 0
        code_execution_count = 0
        cost_events = []
        by_phase = {}
        by_step = {}

        for entry in self.trajectory:
            if entry.get("type") == "result":
                turn_index = turn_count
                turn_count += 1
                usage = entry.get("usage") or {}
                if isinstance(usage, dict) and usage:
                    usage_event_count += 1
                    usage_fields = canonical_usage_fields(usage)
                    input_tokens += usage_fields["input_tokens"]
                    output_tokens += usage_fields["output_tokens"]
                    total_tokens += usage_fields["total_tokens"]
                    reasoning_tokens += usage_fields["reasoning_tokens"]
                    cached_input_tokens += usage_fields["cached_input_tokens"]

                    timing = entry.get("timing") if isinstance(entry.get("timing"), dict) else {}
                    start_time = timing.get("start_time")
                    end_time = timing.get("end_time")
                    duration_sec = timing.get("duration")
                    if not isinstance(start_time, (int, float)):
                        start_time = None
                    if not isinstance(end_time, (int, float)):
                        end_time = start_time
                    if not isinstance(duration_sec, (int, float)):
                        duration_sec = (
                            max(0.0, end_time - start_time)
                            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))
                            else 0.0
                        )
                    duration_sec = round(max(0.0, float(duration_sec or 0.0)), 3)

                    event = {
                        "event_index": len(cost_events),
                        "task_id": self.current_task_id or "",
                        "step_id": f"codex_turn_{turn_index}",
                        "step_index": turn_index,
                        "attempt_index": 0,
                        "phase": "codex_turn",
                        "model": self.model,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_sec": duration_sec,
                        "count_as_turn": True,
                        **usage_fields,
                        "raw_usage": usage,
                    }
                    cost_events.append(event)

                    phase_bucket = by_phase.setdefault("codex_turn", {
                        "llm_call_count": 0,
                        "usage_event_count": 0,
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "cached_input_tokens": 0,
                        "duration_sec": 0.0,
                    })
                    step_bucket = by_step.setdefault(event["step_id"], {
                        "llm_call_count": 0,
                        "usage_event_count": 0,
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "cached_input_tokens": 0,
                        "duration_sec": 0.0,
                    })
                    for bucket in (phase_bucket, step_bucket):
                        bucket["llm_call_count"] += 1
                        bucket["usage_event_count"] += 1
                        for key, value in usage_fields.items():
                            bucket[key] += value
                        bucket["duration_sec"] = round(bucket["duration_sec"] + duration_sec, 3)
            if entry.get("code_action"):
                code_execution_count += 1

        wall_time = 0.0
        if self.run_start_time is not None and self.run_end_time is not None:
            wall_time = self.run_end_time - self.run_start_time
        elif self.trajectory:
            starts = [
                entry.get("timing", {}).get("start_time")
                for entry in self.trajectory
                if isinstance(entry.get("timing"), dict)
            ]
            ends = [
                entry.get("timing", {}).get("end_time")
                for entry in self.trajectory
                if isinstance(entry.get("timing"), dict)
            ]
            starts = [value for value in starts if isinstance(value, (int, float))]
            ends = [value for value in ends if isinstance(value, (int, float))]
            if starts and ends:
                wall_time = max(ends) - min(starts)

        self.current_cost_events = cost_events
        return {
            "total_wall_time_sec": round(wall_time, 3),
            "llm_call_count": turn_count,
            "usage_event_count": usage_event_count,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_input_tokens": cached_input_tokens,
            "turn_count": turn_count,
            "code_execution_count": code_execution_count,
            "by_phase": by_phase,
            "by_step": by_step,
        }

    def _finalize_cost_metrics(self):
        metrics = self._compute_cost_metrics()
        self.current_cost_metrics = metrics
        if self.current_task_id:
            self.cost_metrics[self.current_task_id] = metrics
            self.cost_metric_events[self.current_task_id] = list(self.current_cost_events)

    def get_cost_metrics(self, task_id=None):
        if task_id:
            if task_id in self.cost_metrics:
                return self.cost_metrics[task_id]
            if task_id == self.current_task_id:
                return self.current_cost_metrics
            return {}
        return self.current_cost_metrics

    def get_cost_metric_events(self, task_id=None):
        if task_id:
            if task_id in self.cost_metric_events:
                return self.cost_metric_events[task_id]
            if task_id == self.current_task_id:
                return self.current_cost_events
            return []
        return self.current_cost_events

    def get_trajectory(self):
        return {
            "task": self.instruction,
            "trajectory": self.trajectory,
        }

    def get_trace_events(self):
        return list(self.raw_events or self.trajectory)

    def _build_evaluation_actions(self):
        actions = []
        for step in self.trajectory:
            step_type = step.get("type", "")

            if step_type in ("thread_started", "turn_started", "item_started", "item_updated", "result"):
                continue

            if step_type == "assistant":
                content = step.get("content", "")
                code_action = step.get("code_action")
                observations = step.get("observations", "")
                if code_action:
                    action = {
                        "action": "Bash",
                        "content": code_action,
                        "observation": observations,
                        "token_usage": step.get("usage"),
                        "timing": step.get("timing"),
                    }
                    exit_code = step.get("exit_code")
                    if exit_code is not None:
                        action["exit_code"] = exit_code
                    actions.append(action)
                elif content:
                    actions.append({
                        "action": "Think",
                        "content": content,
                        "token_usage": step.get("usage"),
                        "timing": step.get("timing"),
                    })
            elif step_type == "error":
                actions.append({
                    "action": "Error",
                    "content": step.get("content", ""),
                    "token_usage": step.get("usage"),
                    "timing": step.get("timing"),
                })
            elif step_type == "turn.failed":
                actions.append({
                    "action": "TurnFailed",
                    "content": step.get("content", ""),
                    "token_usage": step.get("usage"),
                    "timing": step.get("timing"),
                })
            else:
                actions.append({
                    "action": step_type,
                    "content": str(step.get("content", "")),
                    "token_usage": step.get("usage"),
                    "timing": step.get("timing"),
                })
        return actions

    def get_trace_summary(self, *, finished=None, result=None, result_files=None):
        metrics = self.current_cost_metrics or self._compute_cost_metrics()
        total_token_usage = {
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "reasoning_tokens": metrics.get("reasoning_tokens", 0),
            "cached_input_tokens": metrics.get("cached_input_tokens", 0),
            "total_tokens": metrics.get("total_tokens", 0),
        }
        return {
            "task_id": self.current_task_id,
            "task": self.instruction,
            "model": self.model,
            "finished": finished,
            "steps": len(self.trajectory),
            "result": result,
            "result_files": result_files or {},
            "total_token_usage": total_token_usage,
            "cost_metrics": metrics,
            "actions": self._build_evaluation_actions(),
            "trajectory": self.trajectory,
        }
