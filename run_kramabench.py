import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import uuid

from tqdm import tqdm

from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.agent.acid_agent import ACIDAgent


# Logger Configs
logger = logging.getLogger("da_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

from da_agent.utils.runtime_config import (
    KRAMABENCH_DATA_BASE,
    KRAMABENCH_DATASET_DIR,
    KRAMABENCH_OUTPUT_DIR,
    KRAMABENCH_LOG_DIR,
)

os.makedirs(KRAMABENCH_LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(KRAMABENCH_LOG_DIR, f"kramabench-normal-{datetime_str}.log"), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join(KRAMABENCH_LOG_DIR, f"kramabench-debug-{datetime_str}.log"), encoding="utf-8")

stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("da_agent"))
logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

# Constants
KRAMABENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Kramabench")
DATA_BASE = KRAMABENCH_DATA_BASE
DATASET_DIR = KRAMABENCH_DATASET_DIR
OUTPUT_DIR = KRAMABENCH_OUTPUT_DIR
KRAMABENCH_RESULTS_DIR = os.path.join(KRAMABENCH_DIR, "results")


def _get_agent_cost_metrics(agent, task_id: str) -> dict:
    if agent is None:
        return {}
    getter = getattr(agent, "get_cost_metrics", None)
    if callable(getter):
        try:
            return getter(task_id) or {}
        except Exception:
            return {}
    cost_metrics = getattr(agent, "cost_metrics", None)
    if isinstance(cost_metrics, dict):
        return cost_metrics.get(task_id, {}) or {}
    return {}


def _get_agent_cost_metric_events(agent, task_id: str) -> list:
    if agent is None:
        return []
    getter = getattr(agent, "get_cost_metric_events", None)
    if callable(getter):
        try:
            events = getter(task_id)
            return events if isinstance(events, list) else []
        except Exception:
            return []
    return []


def _write_cost_metric_events(output_dir: str, events: list) -> dict:
    if not events:
        return {}
    filename = "cost_metrics_events.jsonl"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for idx, event in enumerate(events):
            if isinstance(event, dict):
                event_copy = dict(event)
            else:
                event_copy = {"event": str(event)}
            event_copy.setdefault("event_index", idx)
            f.write(json.dumps(event_copy, ensure_ascii=False) + "\n")
    return {"events": filename}


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _build_acid_trace_actions(task_trajectory: list) -> list:
    actions = []
    for entry in task_trajectory:
        if not isinstance(entry, dict):
            continue
        execution_meta = entry.get("execution_meta")
        if not isinstance(execution_meta, dict):
            execution_meta = {}
        action_type = (
            entry.get("action_type")
            or execution_meta.get("action_type")
            or entry.get("type")
            or "Action"
        )
        content = entry.get("code") or entry.get("action") or entry.get("thought") or ""
        timing = entry.get("timing")
        if not isinstance(timing, dict):
            timing = execution_meta.get("timing") if isinstance(execution_meta.get("timing"), dict) else {}

        action = {
            "action": action_type,
            "phase": entry.get("type", ""),
            "step_id": entry.get("step_id", ""),
            "cycle_id": entry.get("cycle_id", ""),
            "content": content,
            "thought": entry.get("thought", ""),
            "observation": entry.get("observation", ""),
            "observation_digest": entry.get("observation_digest", ""),
            "token_usage": entry.get("llm_usage") or entry.get("usage") or {},
            "timing": timing,
            "review_decision": entry.get("review_decision", ""),
        }
        if "exit_code" in execution_meta:
            action["exit_code"] = execution_meta.get("exit_code")
        actions.append(action)
    return actions


def _write_acid_trace_files(
    output_dir: str,
    *,
    task_id: str,
    task: str,
    model: str,
    finished: bool,
    result: str,
    result_files: dict,
    cost_metrics: dict,
    task_trajectory: list,
) -> dict:
    trace_jsonl_name = "acid_trace.jsonl"
    trace_summary_name = "acid_trace_summary.json"
    trace_jsonl_path = os.path.join(output_dir, trace_jsonl_name)
    trace_summary_path = os.path.join(output_dir, trace_summary_name)

    with open(trace_jsonl_path, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(task_trajectory):
            event = _jsonable(entry)
            if not isinstance(event, dict):
                event = {"type": "raw", "content": str(entry)}
            event.setdefault("task_id", task_id)
            event.setdefault("sequence_index", idx)
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    total_token_usage = {
        "input_tokens": cost_metrics.get("input_tokens", 0),
        "output_tokens": cost_metrics.get("output_tokens", 0),
        "reasoning_tokens": cost_metrics.get("reasoning_tokens", 0),
        "cached_input_tokens": cost_metrics.get("cached_input_tokens", 0),
        "total_tokens": cost_metrics.get("total_tokens", 0),
    }
    trace_summary = {
        "task_id": task_id,
        "task": task,
        "model": model,
        "finished": finished,
        "steps": len(task_trajectory),
        "result": result,
        "result_files": result_files or {},
        "total_token_usage": total_token_usage,
        "cost_metrics": cost_metrics or {},
        "actions": _build_acid_trace_actions(task_trajectory),
        "trajectory": task_trajectory,
    }
    with open(trace_summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(trace_summary), f, indent=2, ensure_ascii=False)

    return {
        "jsonl": trace_jsonl_name,
        "summary": trace_summary_name,
    }


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACIDAgent on KramaBench")

    parser.add_argument("--domain", type=str, default="archeology",
                        help="KramaBench domain (archeology, astronomy, biomedical, environment, legal, wildfire)")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Max steps per task")
    parser.add_argument("--max_memory_length", type=int, default=15)
    parser.add_argument("--suffix", "-s", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="qwen3.5-397b-a17b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--image_name", type=str, default="idatascience-kramabench")
    parser.add_argument("--overwriting", action="store_true", default=False)
    parser.add_argument("--retry_failed", action="store_true", default=False)
    parser.add_argument("--task_index", "-i", type=str, default="all",
                        help="Index range of tasks, e.g., '0-5', '2,3', 'all'")
    parser.add_argument("--task_name", "-n", type=str, default="",
                        help="Task id substring filter (e.g., 'easy')")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--agent_type", type=str, default="acid", choices=["prompt", "acid", "claude-code", "codex"])
    return parser.parse_args()


def _get_experiment_id(args: argparse.Namespace) -> str:
    model_name = args.model.split("/")[-1]
    if args.suffix:
        return f"{model_name}-{args.suffix}-{uuid.uuid4().hex[:8]}"
    else:
        return f"{model_name}-{uuid.uuid4().hex[:8]}"


def _get_dataset_dir(domain: str) -> str:
    """Get the dataset input directory for a domain."""
    dataset_dir = os.path.join(DATASET_DIR, domain, "input")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def _load_workload(args: argparse.Namespace) -> list:
    """Load KramaBench workload for the given domain."""
    workload_path = os.path.join(KRAMABENCH_DIR, "workload", f"{args.domain}.json")
    if not os.path.exists(workload_path):
        raise FileNotFoundError(f"Workload not found: {workload_path}")

    with open(workload_path, "r") as f:
        tasks = json.load(f)

    # Filter by task_name
    if args.task_name:
        tasks = [t for t in tasks if args.task_name in t["id"]]

    # Filter by task_index
    if args.task_index != "all":
        if "-" in args.task_index:
            start, end = map(int, args.task_index.split("-"))
            tasks = tasks[start:end]
        else:
            indices = list(map(int, args.task_index.split(",")))
            tasks = [tasks[i] for i in indices if i < len(tasks)]

    return tasks


def _should_skip_task(result_path: str, args: argparse.Namespace, instance_id: str) -> bool:
    """Check if a task should be skipped."""
    if not args.overwriting and os.path.exists(result_path):
        if args.retry_failed:
            try:
                with open(result_path, "r") as f:
                    result = json.load(f)
                if result.get("finished", False):
                    logger.info(f"Skipping {instance_id} (already succeeded)")
                    return True
                else:
                    logger.info(f"Retrying {instance_id} (failed previously)")
                    return False
            except:
                return False
        else:
            logger.info(f"Skipping {instance_id} (result exists)")
            return True
    return False


def _extract_model_output(trajectory: list, task_result: dict, task_id: str = None) -> str:
    """Extract model output from ACIDAgent trajectory for KramaBench format.

    Priority:
    1. If task finished with Terminate, use its output
    2. Otherwise fall back to last non-empty Bash/Python observation
       (e.g., when max_steps was reached before Terminate)
    """
    # If task finished with Terminate, use its output
    if task_result.get("done") and task_result.get("result"):
        result = task_result["result"]
        # Skip system error messages, fall through to observation fallback
        if result not in ("Max steps reached", "Global max steps reached",
                           "Task incomplete", "Parse failure", "Task completed"):
            return result

    # Codex/Claude-style trajectories may store the final response as the last
    # assistant text instead of a Terminate action result.
    target_task_id = task_id or task_result.get("task_id")
    for entry in reversed(trajectory):
        if target_task_id and entry.get("task_id") != target_task_id:
            continue
        if entry.get("type") == "assistant" and not entry.get("code_action"):
            content = (entry.get("content") or "").strip()
            if content:
                return content

    # Fall back to last non-empty observation
    # Use passed task_id (task_result dict doesn't have a task_id field)
    for entry in reversed(trajectory):
        if target_task_id and entry.get("task_id") != target_task_id:
            continue
        obs = entry.get("post_action_observation") or entry.get("observation", "")
        if obs and obs.strip() and obs != "Terminate":
            return obs

    return ""


def _clean_final_answer_text(text: str) -> str:
    """Keep only the final answer value from CLI agent final messages."""
    text = (text or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    markers = ("final_answer:", "final answer:")
    for line in reversed(lines):
        lower = line.lower()
        for marker in markers:
            if lower.startswith(marker):
                return line.split(":", 1)[1].strip().strip("`")

    return lines[-1].strip("`")


def _extract_pipeline_code(trajectory: list, task_id: str) -> str:
    """Extract Python code from trajectory for KramaBench pipeline_code field."""
    code_parts = []
    for entry in trajectory:
        if entry.get("task_id") != task_id:
            continue
        action_str = entry.get("action", "")
        code = entry.get("code", "")
        if code and ("Python" in action_str or "python" in action_str.lower()):
            code_parts.append(code)

    return "\n\n".join(code_parts) if code_parts else ""


def _write_kramabench_cache(tasks: list, agent, experiment_id: str, domain: str) -> None:
    """
    Convert ACIDAgent results to KramaBench cache format.

    Writes per-task cache files and merges into central cache,
    following the same format as KramaBench's executor.
    """
    sut_name = f"ACIDAgent-{experiment_id}"
    results_dir = os.path.join(KRAMABENCH_RESULTS_DIR, sut_name)
    cache_dir = os.path.join(results_dir, "response_cache")
    tasks_cache_dir = os.path.join(cache_dir, "tasks")
    os.makedirs(tasks_cache_dir, exist_ok=True)

    trajectory = agent.trajectory if agent else []
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for task_info in tasks:
        task_id = task_info["task_id"]
        task_result = agent.task_results.get(task_id, {"done": False, "result": ""})
        task_trajectory = [e for e in trajectory if e.get("task_id") == task_id]

        # Compute runtime from trajectory timestamps
        runtime = 0.0
        if task_trajectory:
            try:
                first_ts = datetime.datetime.fromisoformat(task_trajectory[0]["timestamp"])
                last_ts = datetime.datetime.fromisoformat(task_trajectory[-1]["timestamp"])
                runtime = (last_ts - first_ts).total_seconds()
            except:
                runtime = 0.0

        model_output = _extract_model_output(task_trajectory, task_result, task_id=task_id)
        if agent and agent.__class__.__module__.endswith((".codex", ".claude_code")):
            model_output = _clean_final_answer_text(model_output)
        pipeline_code = _extract_pipeline_code(trajectory, task_id)
        cost_metrics = _get_agent_cost_metrics(agent, task_id)

        cache_entry = {
            "task_id": task_id,
            # KramaBench evaluator expects a dict with "id" and "answer" keys.
            "model_output": {"id": "main-task", "answer": model_output},
            "code": pipeline_code,
            "token_usage_sut": cost_metrics.get("total_tokens", 0),
            "token_usage_sut_input": cost_metrics.get("input_tokens", 0),
            "token_usage_sut_output": cost_metrics.get("output_tokens", 0),
            "cost_metrics": cost_metrics,
            "subresponses": [],
            "runtime": cost_metrics.get("total_wall_time_sec", runtime),
        }

        all_results.append(cache_entry)

        # Write per-task cache file
        task_cache_file = os.path.join(
            tasks_cache_dir,
            f"{domain}_task_{task_id}_{timestamp_str}.json"
        )
        with open(task_cache_file, "w") as f:
            json.dump(cache_entry, f, indent=2)

    # Write merged central cache
    central_cache_file = os.path.join(cache_dir, f"{domain}_{timestamp_str}.json")
    with open(central_cache_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"KramaBench cache written: {central_cache_file}")


# Modified on 6/19/2026
def _get_agent_env(args):
    from da_agent.utils.llm_utils import OPENAI_API_KEY, OPENAI_API_BASE
    env = {}
    if args.agent_type == "claude-code":
        env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or OPENAI_API_KEY
        env["ANTHROPIC_BASE_URL"] = (
            os.getenv("ANTHROPIC_BASE_URL")
            or os.getenv("ANTHROPIC_API_BASE")
            or "https://dashscope.aliyuncs.com/apps/anthropic"
        )
    elif args.agent_type == "codex":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or OPENAI_API_KEY
        api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or OPENAI_API_BASE
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_API_BASE"] = api_base
        env["OPENAI_BASE_URL"] = api_base
        if os.getenv("DASHSCOPE_API_KEY"):
            env["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
    if env:
        return {"environment": env}
    return {}


def test_acid_agent(args: argparse.Namespace, workload_tasks: list) -> None:
    """Run ACIDAgent on KramaBench tasks."""
    experiment_id = _get_experiment_id(args)
    domain = args.domain

    # Get dataset directory — DA_Agent_Env copies directly from here
    dataset_dir = _get_dataset_dir(domain)
    logger.info(f"Dataset directory: {dataset_dir}")

    # Initialize ACIDAgent
    agent = ACIDAgent(
        model=args.model,
        max_steps=args.max_steps,
        max_memory_length=args.max_memory_length,
        image_name=args.image_name,
        sequential_mode=True,
    )

    # Build task list
    tasks = []
    env_map = {}

    for workload_task in workload_tasks:
        task_id = workload_task["id"]
        instance_id = f"{experiment_id}/{domain}/{task_id}"
        output_dir = os.path.join(args.output_dir, experiment_id, domain, task_id)

        # Check skip conditions
        result_json_path = os.path.join(output_dir, "result.json")
        if _should_skip_task(result_json_path, args, instance_id):
            continue

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Create DA_Agent_Env
        env_config = {
            "image_name": args.image_name,
            "init_args": {
                "name": f"{experiment_id}-{task_id}",
                "work_dir": "/workspace",
            }
        }

        # KramaBench task config adapted for DA_Agent_Env
        task_config = {
            "id": task_id,
            "instruction": workload_task["query"],
            "post_process": [],
        }

        env = DA_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            source_dir="",  # Not used — dataset_dir overrides it
            cache_dir="./cache",
            mnt_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        env_map[task_id] = env

        tasks.append({
            "task_id": task_id,
            "instruction": workload_task["query"],
            "mnt_dir": output_dir,
            "source_dir": "",  # Not used
            "task_config": task_config,
            "env": env,
        })

        logger.info(f"Prepared task {task_id}")

    if not tasks:
        logger.info("No tasks to run")
        return

    # Run batch
    logger.info(f"Running {len(tasks)} KramaBench tasks for domain {domain}")
    results = agent.run(tasks)

    # Post-process and save per-task results
    for task_info in tasks:
        task_id = task_info["task_id"]
        env = env_map.get(task_id)
        output_dir = task_info["mnt_dir"]

        if env is None:
            continue

        try:
            os.makedirs(os.path.join(output_dir, "kramabench"), exist_ok=True)

            # Get diff files (added/changed by agent)
            diff_result = env.post_process()
            env._cleanup_source_files()

            task_result = results.get(task_id, {"done": False, "result": "Unknown"})
            task_trajectory = [e for e in agent.trajectory
                              if e.get("task_id") == task_id]
            cost_metrics = _get_agent_cost_metrics(agent, task_id)
            cost_metric_files = _write_cost_metric_events(
                output_dir,
                _get_agent_cost_metric_events(agent, task_id),
            )
            trace_files = _write_acid_trace_files(
                output_dir,
                task_id=task_id,
                task=task_info["instruction"],
                model=args.model,
                finished=task_result.get("done", False),
                result=task_result.get("result", ""),
                result_files=diff_result,
                cost_metrics=cost_metrics,
                task_trajectory=task_trajectory,
            )

            result_data = {
                "finished": task_result.get("done", False),
                "steps": len(task_trajectory),
                "result": task_result.get("result", ""),
                "result_files": diff_result,
                "cost_metrics": cost_metrics,
                "cost_metric_files": cost_metric_files,
                "trace_files": trace_files,
                "Task": task_info["instruction"],
                "trajectory": task_trajectory,
            }
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(result_data, f, indent=2)

            logger.info(f"Finished task {task_id}")
        except Exception as e:
            logger.error(f"Post-process failed for {task_id}: {e}")
        finally:
            env.close()

    # Write KramaBench cache files for evaluate.py
    _write_kramabench_cache(tasks, agent, experiment_id, domain)

    logger.info(f"All tasks completed for domain {domain}")


def test_prompt_agent(args: argparse.Namespace, workload_tasks: list) -> None:
    """Run the original DA-Agent PromptAgent on KramaBench tasks."""
    experiment_id = _get_experiment_id(args)
    domain = args.domain

    dataset_dir = _get_dataset_dir(domain)
    logger.info(f"Dataset directory: {dataset_dir}")

    from da_agent.agent.agents import PromptAgent
    agent = PromptAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_memory_length=args.max_memory_length,
        max_steps=args.max_steps,
    )

    tasks = []
    env_map = {}

    for workload_task in workload_tasks:
        task_id = workload_task["id"]
        instance_id = f"{experiment_id}/{domain}/{task_id}"
        output_dir = os.path.join(args.output_dir, experiment_id, domain, task_id)

        result_json_path = os.path.join(output_dir, "result.json")
        if _should_skip_task(result_json_path, args, instance_id):
            continue

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        env_config = {
            "image_name": args.image_name,
            "init_args": {
                "name": f"{experiment_id}-{task_id}",
                "work_dir": "/workspace",
            }
        }

        task_config = {
            "id": task_id,
            "instruction": workload_task["query"],
            "question": workload_task["query"],
            "post_process": [],
        }

        env = DA_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            source_dir="",
            cache_dir="./cache",
            mnt_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        env_map[task_id] = env

        tasks.append({
            "task_id": task_id,
            "instruction": workload_task["query"],
            "mnt_dir": output_dir,
            "source_dir": "",
            "task_config": task_config,
            "env": env,
        })

        logger.info(f"Prepared task {task_id}")

    if not tasks:
        logger.info("No tasks to run")
        return

    task_results = {}
    all_trajectories = []

    for task_info in tasks:
        task_id = task_info["task_id"]
        env = env_map[task_id]
        output_dir = task_info["mnt_dir"]

        try:
            agent.set_env_and_task(env)
            success, message = agent.run()
            logger.info(f"Task {task_id}: success={success}, message={message}")

            task_results[task_id] = {"done": success, "result": message}

            traj_data = agent.get_trajectory()
            traj_entries = traj_data.get("trajectory", []) if isinstance(traj_data, dict) else []
            for entry in traj_entries:
                entry_copy = dict(entry)
                entry_copy["task_id"] = task_id
                all_trajectories.append(entry_copy)

            diff_result = env.post_process()
            env._cleanup_source_files()
            cost_metrics = _get_agent_cost_metrics(agent, task_id)
            cost_metric_files = _write_cost_metric_events(
                output_dir,
                _get_agent_cost_metric_events(agent, task_id),
            )
            trace_jsonl_name = "da_agent_trace.jsonl"
            trace_summary_name = "da_agent_trace_summary.json"
            trace_jsonl_path = os.path.join(output_dir, trace_jsonl_name)
            trace_summary_path = os.path.join(output_dir, trace_summary_name)

            trace_events_getter = getattr(agent, "get_trace_events", None)
            trace_events = trace_events_getter() if callable(trace_events_getter) else traj_entries
            with open(trace_jsonl_path, "w", encoding="utf-8") as f:
                for idx, event in enumerate(trace_events):
                    event_copy = _jsonable(event)
                    if not isinstance(event_copy, dict):
                        event_copy = {"type": "raw", "content": str(event_copy)}
                    event_copy.setdefault("task_id", task_id)
                    event_copy.setdefault("sequence_index", idx)
                    f.write(json.dumps(event_copy, ensure_ascii=False) + "\n")

            trace_summary_getter = getattr(agent, "get_trace_summary", None)
            if callable(trace_summary_getter):
                trace_summary = trace_summary_getter(
                    finished=success,
                    result=message,
                    result_files=diff_result,
                )
            else:
                trace_summary = {
                    "task_id": task_id,
                    "task": task_info["instruction"],
                    "model": args.model,
                    "finished": success,
                    "steps": len(traj_entries),
                    "result": message,
                    "result_files": diff_result,
                    "cost_metrics": cost_metrics,
                    "actions": traj_entries,
                    "trajectory": traj_data,
                }
            with open(trace_summary_path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(trace_summary), f, indent=2, ensure_ascii=False)

            result_data = {
                "finished": success,
                "steps": len(traj_entries),
                "result": message,
                "result_files": diff_result,
                "cost_metrics": cost_metrics,
                "cost_metric_files": cost_metric_files,
                "trace_files": {
                    "jsonl": trace_jsonl_name,
                    "summary": trace_summary_name,
                },
                "Task": task_info["instruction"],
                "trajectory": traj_data,
            }
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(_jsonable(result_data), f, indent=2, ensure_ascii=False)

            logger.info(f"Finished task {task_id}")
        except Exception as e:
            logger.error(f"Post-process failed for {task_id}: {e}")
            task_results[task_id] = {"done": False, "result": ""}
        finally:
            env.close()

    agent.task_results = task_results
    agent.trajectory = all_trajectories
    _write_kramabench_cache(tasks, agent, experiment_id, domain)
    logger.info(f"All tasks completed for domain {domain}")


def test_claude_code_agent(args: argparse.Namespace, workload_tasks: list) -> None:
    """Run Claude Code agent on KramaBench tasks."""
    experiment_id = _get_experiment_id(args)
    domain = args.domain

    dataset_dir = _get_dataset_dir(domain)
    logger.info(f"Dataset directory: {dataset_dir}")

    from da_agent.agent.claude_code import PromptAgent
    agent = PromptAgent(model=args.model)

    tasks = []
    env_map = {}

    for workload_task in workload_tasks:
        task_id = workload_task["id"]
        instance_id = f"{experiment_id}/{domain}/{task_id}"
        output_dir = os.path.join(args.output_dir, experiment_id, domain, task_id)

        result_json_path = os.path.join(output_dir, "result.json")
        if _should_skip_task(result_json_path, args, instance_id):
            continue

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        env_config = {
            "image_name": "claude-code",
            "init_args": {
                "name": f"{experiment_id}-{task_id}",
                "work_dir": "/workspace",
                **_get_agent_env(args)
            }
        }

        task_config = {
            "id": task_id,
            "instruction": workload_task["query"],
            "question": workload_task["query"],
            "post_process": [],
        }

        env = DA_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            source_dir="",
            cache_dir="./cache",
            mnt_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        env_map[task_id] = env

        tasks.append({
            "task_id": task_id,
            "instruction": workload_task["query"],
            "mnt_dir": output_dir,
            "source_dir": "",
            "task_config": task_config,
            "env": env,
        })

        logger.info(f"Prepared task {task_id}")

    if not tasks:
        logger.info("No tasks to run")
        return

    task_results = {}
    all_trajectories = []

    for task_info in tasks:
        task_id = task_info["task_id"]
        env = env_map[task_id]
        output_dir = task_info["mnt_dir"]

        try:
            agent.set_env_and_task(env)
            success, message = agent.run()
            logger.info(f"Task {task_id}: success={success}, message={message}")

            # Bug fix: populate task_results so _write_kramabench_cache can extract answers
            task_results[task_id] = {"done": success, "result": message}

            # Collect trajectory entries tagged with task_id
            traj_data = agent.get_trajectory()
            traj_entries = traj_data.get("trajectory", []) if isinstance(traj_data, dict) else []
            for entry in traj_entries:
                entry_copy = dict(entry)
                entry_copy["task_id"] = task_id
                all_trajectories.append(entry_copy)

            diff_result = env.post_process()
            env._cleanup_source_files()
            cost_metrics = _get_agent_cost_metrics(agent, task_id)
            cost_metric_files = _write_cost_metric_events(
                output_dir,
                _get_agent_cost_metric_events(agent, task_id),
            )
            trace_jsonl_name = "claude_trace.jsonl"
            trace_summary_name = "claude_trace_summary.json"
            trace_jsonl_path = os.path.join(output_dir, trace_jsonl_name)
            trace_summary_path = os.path.join(output_dir, trace_summary_name)

            trace_events_getter = getattr(agent, "get_trace_events", None)
            trace_events = trace_events_getter() if callable(trace_events_getter) else traj_entries
            with open(trace_jsonl_path, "w", encoding="utf-8") as f:
                for idx, event in enumerate(trace_events):
                    event_copy = _jsonable(event)
                    if not isinstance(event_copy, dict):
                        event_copy = {"type": "raw", "content": str(event_copy)}
                    event_copy.setdefault("task_id", task_id)
                    event_copy.setdefault("sequence_index", idx)
                    f.write(json.dumps(event_copy, ensure_ascii=False) + "\n")

            trace_summary_getter = getattr(agent, "get_trace_summary", None)
            if callable(trace_summary_getter):
                trace_summary = trace_summary_getter(
                    finished=success,
                    result=message,
                    result_files=diff_result,
                )
            else:
                trace_summary = {
                    "task_id": task_id,
                    "task": task_info["instruction"],
                    "finished": success,
                    "steps": len(traj_entries),
                    "result": message,
                    "result_files": diff_result,
                    "cost_metrics": cost_metrics,
                    "actions": traj_entries,
                    "trajectory": traj_data,
                }
            with open(trace_summary_path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(trace_summary), f, indent=2, ensure_ascii=False)

            result_data = {
                "finished": success,
                "steps": len(traj_entries),
                "result": message,
                "result_files": diff_result,
                "cost_metrics": cost_metrics,
                "cost_metric_files": cost_metric_files,
                "trace_files": {
                    "jsonl": trace_jsonl_name,
                    "summary": trace_summary_name,
                },
                "Task": task_info["instruction"],
                "trajectory": traj_data,
            }
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(result_data, f, indent=2)

            logger.info(f"Finished task {task_id}")
        except Exception as e:
            logger.error(f"Post-process failed for {task_id}: {e}")
            task_results[task_id] = {"done": False, "result": ""}
        finally:
            env.close()

    agent.task_results = task_results
    agent.trajectory = all_trajectories
    _write_kramabench_cache(tasks, agent, experiment_id, domain)
    logger.info(f"All tasks completed for domain {domain}")


def test_codex_agent(args: argparse.Namespace, workload_tasks: list) -> None:
    """Run Codex CLI agent on KramaBench tasks."""
    experiment_id = _get_experiment_id(args)
    domain = args.domain

    dataset_dir = _get_dataset_dir(domain)
    logger.info(f"Dataset directory: {dataset_dir}")

    from da_agent.agent.codex import PromptAgent
    agent = PromptAgent(model=args.model)

    tasks = []
    env_map = {}

    for workload_task in workload_tasks:
        task_id = workload_task["id"]
        instance_id = f"{experiment_id}/{domain}/{task_id}"
        output_dir = os.path.join(args.output_dir, experiment_id, domain, task_id)

        result_json_path = os.path.join(output_dir, "result.json")
        if _should_skip_task(result_json_path, args, instance_id):
            continue

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        env_config = {
            "image_name": "codex",
            "init_args": {
                "name": f"{experiment_id}-{task_id}",
                "work_dir": "/workspace",
                **_get_agent_env(args),
            }
        }

        task_config = {
            "id": task_id,
            "instruction": workload_task["query"],
            "question": workload_task["query"],
            "post_process": [],
        }

        env = DA_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            source_dir="",
            cache_dir="./cache",
            mnt_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        env_map[task_id] = env

        tasks.append({
            "task_id": task_id,
            "instruction": workload_task["query"],
            "mnt_dir": output_dir,
            "source_dir": "",
            "task_config": task_config,
            "env": env,
        })

        logger.info(f"Prepared task {task_id}")

    if not tasks:
        logger.info("No tasks to run")
        return

    task_results = {}
    all_trajectories = []

    for task_info in tasks:
        task_id = task_info["task_id"]
        env = env_map[task_id]
        output_dir = task_info["mnt_dir"]

        try:
            agent.set_env_and_task(env)
            success, message = agent.run()
            logger.info(f"Task {task_id}: success={success}, message={message}")

            task_results[task_id] = {"done": success, "result": message}

            traj_data = agent.get_trajectory()
            traj_entries = traj_data.get("trajectory", []) if isinstance(traj_data, dict) else []
            for entry in traj_entries:
                entry_copy = dict(entry)
                entry_copy["task_id"] = task_id
                all_trajectories.append(entry_copy)

            diff_result = env.post_process()
            env._cleanup_source_files()
            cost_metrics = _get_agent_cost_metrics(agent, task_id)
            cost_metric_files = _write_cost_metric_events(
                output_dir,
                _get_agent_cost_metric_events(agent, task_id),
            )
            trace_jsonl_name = "codex_trace.jsonl"
            trace_summary_name = "codex_trace_summary.json"
            trace_jsonl_path = os.path.join(output_dir, trace_jsonl_name)
            trace_summary_path = os.path.join(output_dir, trace_summary_name)

            trace_events_getter = getattr(agent, "get_trace_events", None)
            trace_events = trace_events_getter() if callable(trace_events_getter) else traj_entries
            with open(trace_jsonl_path, "w") as f:
                for idx, event in enumerate(trace_events):
                    event_copy = dict(event) if isinstance(event, dict) else {"type": "raw", "content": str(event)}
                    event_copy.setdefault("task_id", task_id)
                    event_copy.setdefault("sequence_index", idx)
                    f.write(json.dumps(event_copy, ensure_ascii=False) + "\n")

            trace_summary_getter = getattr(agent, "get_trace_summary", None)
            if callable(trace_summary_getter):
                trace_summary = trace_summary_getter(
                    finished=success,
                    result=message,
                    result_files=diff_result,
                )
            else:
                trace_summary = {
                    "task_id": task_id,
                    "task": task_info["instruction"],
                    "finished": success,
                    "steps": len(traj_entries),
                    "result": message,
                    "result_files": diff_result,
                    "cost_metrics": cost_metrics,
                    "actions": traj_entries,
                    "trajectory": traj_data,
                }
            with open(trace_summary_path, "w") as f:
                json.dump(trace_summary, f, indent=2, ensure_ascii=False)

            result_data = {
                "finished": success,
                "steps": len(traj_entries),
                "result": message,
                "result_files": diff_result,
                "cost_metrics": cost_metrics,
                "cost_metric_files": cost_metric_files,
                "trace_files": {
                    "jsonl": trace_jsonl_name,
                    "summary": trace_summary_name,
                },
                "Task": task_info["instruction"],
                "trajectory": traj_data,
            }
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(result_data, f, indent=2)

            logger.info(f"Finished task {task_id}")
        except Exception as e:
            logger.error(f"Post-process failed for {task_id}: {e}")
            task_results[task_id] = {"done": False, "result": ""}
        finally:
            env.close()

    agent.task_results = task_results
    agent.trajectory = all_trajectories
    _write_kramabench_cache(tasks, agent, experiment_id, domain)
    logger.info(f"All tasks completed for domain {domain}")


def main():
    args = config()
    logger.info(f"Args: {args}")

    workload_tasks = _load_workload(args)
    logger.info(f"Loaded {len(workload_tasks)} tasks for domain {args.domain}")

    if args.agent_type == "acid":
        test_acid_agent(args, workload_tasks)
    elif args.agent_type == "prompt":
        test_prompt_agent(args, workload_tasks)
    elif args.agent_type == "claude-code":
        test_claude_code_agent(args, workload_tasks)
    elif args.agent_type == "codex":
        test_codex_agent(args, workload_tasks)
    else:
        logger.error(f"Unsupported agent type: {args.agent_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
