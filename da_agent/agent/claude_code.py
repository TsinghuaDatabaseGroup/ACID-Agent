"""Claude Code Baseline Module - CLI wrapper for benchmark execution.

This module adapts the DA-Agent benchmark wrapper pattern to Claude Code:
- Starts a Docker-backed task environment
- Invokes the Claude Code CLI on each benchmark task
- Captures trajectory, raw CLI events, final answer, and cost metrics

Reference: https://github.com/AgenticDataBench/AgenticDataBench/tree/main/testbed/da_agent/agent
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


class PromptAgent:
    def __init__(
        self,
        # model="claude-sonnet-4-20250514",
        # Tested using the same model of our agent.
        model="qwen3.5-397b-a17b",
        model_type="anthropic",
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
        self.instruction = self.env.task_config['question']
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
                if output_file_name in self.env.task_config.get('output_file_name', []):
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
import subprocess
import sys
import threading

with open("{self.work_dir}/.task_prompt.txt") as f:
    prompt = f.read()

proc = subprocess.Popen(
    ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
     "--model", "{self.model}",
     "--dangerously-skip-permissions"],
    stdout=sys.stdout, stderr=sys.stderr
)

def timeout_handler():
    proc.terminate()
    kill_timer = threading.Timer(300, proc.kill)
    kill_timer.daemon = True
    kill_timer.start()

timer = threading.Timer({DEFAULT_TIME_OUT}, timeout_handler)
timer.daemon = True
timer.start()

sys.exit(proc.wait())
'''
        wrapper_path = os.path.join(self.env.mnt_dir, ".run_claude.py")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)

    def run(self):
        assert self.env is not None, "Environment is not set."

        task_prompt = self._build_task_prompt()
        container_name = self.env.container.name

        # Write task prompt and wrapper script to mounted directory
        task_path = os.path.join(self.env.mnt_dir, ".task_prompt.txt")
        with open(task_path, "w") as f:
            f.write(task_prompt)
        self._write_wrapper_script()

        self.run_start_time = time.time()
        self.run_end_time = None

        # Execute wrapper script inside the container (as non-root user)
        docker_user = os.getenv("DA_AGENT_DOCKER_USER", "agent")
        process = subprocess.Popen(
            ["docker", "exec", "--user", docker_user, str(container_name),
             "python3", f"{self.work_dir}/.run_claude.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
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
                    logger.debug("Claude Code: %s", decoded.strip())
        except Exception as e:
            process.kill()
            logger.error("Error running Claude Code: %s", e)
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

        if exit_code == 0:
            for entry in reversed(self.trajectory):
                if entry.get("type") == "result" and entry.get("result"):
                    return True, entry["result"]
            return True, "Task completed"
        else:
            return False, f"Agent exited with code {exit_code}"

    def _parse_trajectory(self):
        self.trajectory = []
        self.raw_events = []

        # Try parsing as a single JSON array (--output-format json)
        try:
            entries = json.loads(self.raw_output.strip())
            if isinstance(entries, list):
                for i, entry in enumerate(entries):
                    raw_entry = dict(entry) if isinstance(entry, dict) else {"type": "raw", "content": str(entry)}
                    raw_entry.setdefault("sequence_index", i)
                    self.raw_events.append(raw_entry)
                    normalized = self._normalize_entry(entry)
                    self.trajectory.append(normalized)
                return
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL parsing (--output-format stream-json)
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
                raw_entry = dict(entry) if isinstance(entry, dict) else {"type": "raw", "content": str(entry)}
                if timing:
                    raw_entry["timing"] = timing
                self.raw_events.append(raw_entry)
                normalized = self._normalize_entry(entry)
                # Add timing info from recorded timestamps (stream-json only)
                if timing:
                    normalized["timing"] = timing
                self.trajectory.append(normalized)
            except json.JSONDecodeError:
                step = {"type": "raw", "content": line}
                if timing:
                    step["timing"] = timing
                self.raw_events.append(dict(step))
                self.trajectory.append(step)

    def _normalize_entry(self, entry):
        entry_type = entry.get("type", "unknown")

        if entry_type == "system":
            subtype = entry.get("subtype", "")
            if subtype == "init":
                return {
                    "type": "system_init",
                    "session_id": entry.get("session_id", ""),
                    "model": entry.get("model", ""),
                    "cwd": entry.get("cwd", ""),
                }
            return {"type": "system", "subtype": subtype}

        if entry_type == "assistant":
            message = entry.get("message", {})
            msg_id = message.get("id", "")
            content = message.get("content", [])
            text_parts = []
            code_action = None
            tool_uses = []

            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    tool_uses.append({"name": tool_name, "input": tool_input})
                    if tool_name == "Bash" and "command" in tool_input:
                        code_action = tool_input["command"]

            result = {"type": "assistant", "content": "\n".join(text_parts)}
            if msg_id:
                result["msg_id"] = msg_id
            if tool_uses:
                result["tool_uses"] = tool_uses
            if code_action:
                result["code_action"] = code_action
            # Extract per-message usage if available
            usage = message.get("usage", {})
            if usage:
                result["usage"] = usage
            return result

        elif entry_type == "user":
            message = entry.get("message", {})
            content = message.get("content", [])
            tool_result_parts = []
            text_parts = []
            tool_use_ids = []

            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        tool_use_ids.append(tool_use_id)
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        nested_text = []
                        for item in tool_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                nested_text.append(item.get("text", ""))
                            elif item is not None:
                                nested_text.append(str(item))
                        tool_content = "\n".join(nested_text)
                    tool_result_parts.append(str(tool_content))
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            if tool_result_parts:
                result = {
                    "type": "tool_result",
                    "observations": "Execution logs:\n" + "\n".join(tool_result_parts),
                }
                if tool_use_ids:
                    result["tool_use_ids"] = tool_use_ids
                return result

            return {"type": "user", "content": "\n".join(text_parts)}

        elif entry_type == "tool_result":
            content = entry.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)
            result = {"type": "tool_result", "observations": f"Execution logs:\n{content}"}
            return result

        elif entry_type == "result":
            result_entry = {
                "type": "result",
                "subtype": entry.get("subtype", ""),
                "is_error": entry.get("is_error", False),
                "result": entry.get("result", ""),
                "stop_reason": entry.get("stop_reason", ""),
                "duration_ms": entry.get("duration_ms", 0),
                "num_turns": entry.get("num_turns", 0),
            }
            # Extract aggregate usage from the result event
            usage = entry.get("usage", {})
            if usage:
                result_entry["usage"] = usage
            return result_entry

        return entry

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
        result_event_index = 0

        for entry in self.trajectory:
            if entry.get("code_action"):
                code_execution_count += 1

            if entry.get("type") != "result":
                continue

            result_index = result_event_index
            result_event_index += 1
            entry_turns = entry.get("num_turns")
            if isinstance(entry_turns, (int, float)) and int(entry_turns) > 0:
                event_turn_count = int(entry_turns)
            else:
                event_turn_count = 1
            turn_count += event_turn_count

            usage = entry.get("usage") or {}
            if not isinstance(usage, dict) or not usage:
                continue

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
            duration_ms = entry.get("duration_ms")
            if not isinstance(start_time, (int, float)):
                start_time = self.run_start_time
            if not isinstance(end_time, (int, float)):
                end_time = self.run_end_time
            if not isinstance(duration_sec, (int, float)):
                if isinstance(duration_ms, (int, float)) and duration_ms > 0:
                    duration_sec = duration_ms / 1000.0
                elif isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    duration_sec = max(0.0, end_time - start_time)
                else:
                    duration_sec = 0.0
            duration_sec = round(max(0.0, float(duration_sec or 0.0)), 3)

            step_id = f"claude_turn_{result_index}"
            event = {
                "event_index": len(cost_events),
                "task_id": self.current_task_id or "",
                "step_id": step_id,
                "step_index": result_index,
                "attempt_index": 0,
                "phase": "claude_code_turn",
                "model": self.model,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "count_as_turn": True,
                "turn_count": event_turn_count,
                **usage_fields,
                "raw_usage": usage,
            }
            cost_events.append(event)

            phase_bucket = by_phase.setdefault("claude_code_turn", {
                "llm_call_count": 0,
                "usage_event_count": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "duration_sec": 0.0,
            })
            step_bucket = by_step.setdefault(step_id, {
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
                bucket["llm_call_count"] += event_turn_count
                bucket["usage_event_count"] += 1
                for key, value in usage_fields.items():
                    bucket[key] += value
                bucket["duration_sec"] = round(bucket["duration_sec"] + duration_sec, 3)

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
            "total_wall_time_sec": round(max(0.0, wall_time), 3),
            "llm_call_count": turn_count,
            "usage_event_count": usage_event_count,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_input_tokens": cached_input_tokens,
            "step_count": len(self.trajectory),
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
            "trajectory": self.trajectory
        }

    def get_trace_events(self):
        return list(self.raw_events or self.trajectory)

    def _build_evaluation_actions(self):
        actions = []
        for step in self.trajectory:
            step_type = step.get("type", "")
            if step_type in ("system_init", "system", "result"):
                continue

            if step_type == "assistant":
                content = step.get("content", "")
                code_action = step.get("code_action")
                if code_action:
                    actions.append({
                        "action": "Bash",
                        "content": code_action,
                        "observation": step.get("observations", ""),
                        "token_usage": step.get("usage"),
                        "timing": step.get("timing"),
                    })
                elif content:
                    actions.append({
                        "action": "Think",
                        "content": content,
                        "token_usage": step.get("usage"),
                        "timing": step.get("timing"),
                    })
            elif step_type == "tool_result":
                actions.append({
                    "action": "Observation",
                    "content": step.get("observations", ""),
                    "token_usage": step.get("usage"),
                    "timing": step.get("timing"),
                })
            else:
                actions.append({
                    "action": step_type,
                    "content": str(step.get("content") or step.get("result") or ""),
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
