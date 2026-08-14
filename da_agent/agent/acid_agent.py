# Main ACID Agent implementation
# Step-by-step execution with objective review and retry.

# ==================== Imports ====================

import json
import logging
import os
import re
import shutil
import shlex
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from da_agent.agent.action import Action, Bash, Python, SQL, Terminate
from da_agent.agent.prompts import (
    SYS_PROMPT_ACID,
    EXPLORATION_TASK_PROMPT,
    EXPLORATION_ACTION_PROMPT,
    EXPLORATION_SUMMARY_PROMPT,
    CURRENT_DECISION_EXTRACTION_PROMPT,
    OBSERVATION_USE_JUDGE_PROMPT,
)
from da_agent.utils.llm_utils import call_llm, call_llm_with_usage
from da_agent.utils.objective_confidence import (
    ObjectiveConfidenceEvaluator,
    ObjectiveConfidenceInput,
)
from da_agent.utils.probability_contrast import (
    HFContinuationScorer,
    ProbabilityContrastConfig,
    ProbabilityContrastEngine,
)
from da_agent.utils.evidence_surprise import (
    EvidenceSurpriseConfig,
    EvidenceSurpriseEvaluator,
)
from da_agent.utils.cost_metrics import TaskCostTracker
from da_agent.utils.observation_digest import build_observation_digest
from da_agent.review.answer_candidate import AnswerCandidateResult, AnswerCandidateTracker
from da_agent.review.exploration_evidence import ExplorationEvidenceManager
from da_agent.review.retry_policy import RetryPolicy, RetryPolicyConfig
from da_agent.review.workspace_outputs import WorkspaceOutputManager
from da_agent.envs.da_agent import DA_Agent_Env

from da_agent.utils.runtime_config import (
    ACID_RUN_STORAGE_BASE,
    PROB_CONTRAST_MODEL_PATH,
    EVIDENCE_SURPRISE_MODEL_PATH,
)

# ==================== Module Globals ====================

logger = logging.getLogger("da_agent.acid_agent")

DEFAULT_WORK_DIR = "/workspace"


class ACIDAgent:
    """
    ACID-based agent with step-by-step execution.

    Architecture:
    - Round-robin across tasks, one step per task per round
    - LLM generates actions (like PromptAgent)
    - Each task has a persistent Docker container (DA_Agent_Env with volume mount)
    - Steps execute directly in the persistent container
    - Review retry handles repair when objective checks find risky steps
    - Bash actions don't count as steps (exploratory, not task progress)
    """

    DEFAULT_MAX_STEPS_PER_TASK = 20
    DEFAULT_MAX_MEMORY_LENGTH = 15
    DEFAULT_MIN_EXPLORE_STEPS = 1  # Lower bound for adaptive exploration rounds.
    DEFAULT_MAX_EXPLORE_STEPS = 4  # Conservative cap for adaptive exploration rounds per step.

    # ==================== Configuration / Initialization ====================

    def __init__(
        self,
        model: str = "qwen3.5-35b-a3b",
        max_steps: int = None,
        max_steps_per_task: int = None,
        max_memory_length: int = None,
        max_tokens: int = 1500,
        min_explore_steps: int = None,
        max_explore_steps: int = None,
        image_name: str = "idatascience",
        run_storage_base: str = ACID_RUN_STORAGE_BASE,
        sequential_mode: bool = False,
        enable_probability_contrast: bool = True,
        enable_evidence_surprise: bool = True,
        enable_llm_current_extraction: bool = True,
        probability_contrast_model_path: str = PROB_CONTRAST_MODEL_PATH,
        probability_contrast_device: str = "auto",
        max_probability_contrast_probes: int = 4,
        evidence_surprise_model_path: str = EVIDENCE_SURPRISE_MODEL_PATH,
        evidence_surprise_device: str = "auto",
        enable_review_retry: bool = True,
        max_retry_per_step: int = 2,
    ):
        self.model = model
        self.max_steps_per_task = max_steps_per_task or max_steps or self.DEFAULT_MAX_STEPS_PER_TASK
        self.max_memory_length = max_memory_length or self.DEFAULT_MAX_MEMORY_LENGTH
        self.max_tokens = max_tokens
        self.min_explore_steps = min_explore_steps or self.DEFAULT_MIN_EXPLORE_STEPS
        self.max_explore_steps = max_explore_steps or self.DEFAULT_MAX_EXPLORE_STEPS
        self.roundnov_stop_threshold = 0.45  # Stop when a round adds little novelty.
        self.image_name = image_name
        self.run_storage_base = run_storage_base

        # Per-run storage path (set in run())
        self._run_storage_path: Optional[str] = None

        # Available action classes
        self._AVAILABLE_ACTION_CLASSES = [Bash, Python, SQL, Terminate]

        # Execution state (reset per run)
        self.trajectory: List[Dict[str, Any]] = []

        # Per-task state
        self.task_mnt_dirs: Dict[str, str] = {}
        self.task_instructions: Dict[str, str] = {}
        self.task_envs: Dict[str, DA_Agent_Env] = {}  # Persistent containers
        self.task_history_messages: Dict[str, List[Dict]] = {}
        self.task_step_counts: Dict[str, int] = {}
        self.task_done: Dict[str, bool] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}
        self.task_pending_reviews: Dict[str, List[Dict[str, Any]]] = {}
        self.task_configs: Dict[str, Dict[str, Any]] = {}
        self.task_verified_outputs: Dict[str, set] = {}
        self.task_quarantined_outputs: Dict[str, List[Dict[str, Any]]] = {}
        self._task_loggers: Dict[str, logging.Logger] = {}
        self.cost_tracker = TaskCostTracker()
        self._cost_step_context: Dict[str, Dict[str, Any]] = {}

        # Per-task exploration state (reset per run)
        self.task_exploration_summaries: Dict[str, List[str]] = {}
        self.exploration_evidence = ExplorationEvidenceManager(
            model=self.model,
            compact_text=self._compact_for_decision_prompt,
            llm_call=self._tracked_external_llm_call,
            llm_call_accepts_metadata=True,
        )
        self.workspace_outputs = WorkspaceOutputManager()

        # Sequential mode
        self.sequential_mode = sequential_mode
        
        # Round-robin state
        self._task_order: List[str] = []
        self._rr_index: int = 0

        # Parse failure retry tracking
        self._parse_failures: Dict[str, int] = {}

        self.objective_confidence_evaluator = ObjectiveConfidenceEvaluator()
        self.enable_probability_contrast = enable_probability_contrast
        self.enable_evidence_surprise = enable_evidence_surprise
        self.enable_llm_current_extraction = enable_llm_current_extraction
        self.probability_contrast_model_path = (
            os.getenv("ACID_PROB_CONTRAST_MODEL") or probability_contrast_model_path
        )
        self.probability_contrast_device = (
            os.getenv("ACID_PROB_CONTRAST_DEVICE") or probability_contrast_device
        )
        self.max_probability_contrast_probes = max_probability_contrast_probes
        self.evidence_surprise_model_path = (
            os.getenv("ACID_EVIDENCE_SURPRISE_MODEL") or evidence_surprise_model_path
        )
        self.evidence_surprise_device = (
            os.getenv("ACID_EVIDENCE_SURPRISE_DEVICE") or evidence_surprise_device
        )
        shared_scorer = None
        if (
            self.probability_contrast_model_path == self.evidence_surprise_model_path
            and self.probability_contrast_device == self.evidence_surprise_device
            and (self.enable_probability_contrast or self.enable_evidence_surprise)
        ):
            shared_scorer = HFContinuationScorer(
                model_path=self.probability_contrast_model_path,
                device=self.probability_contrast_device,
            )
        self.probability_contrast_engine = ProbabilityContrastEngine(
            ProbabilityContrastConfig(
                model_path=self.probability_contrast_model_path,
                device=self.probability_contrast_device,
                enabled=self.enable_probability_contrast,
                max_probes=self.max_probability_contrast_probes,
                context_char_limit=0,
            ),
            scorer=shared_scorer,
        )
        self.evidence_surprise_evaluator = EvidenceSurpriseEvaluator(
            EvidenceSurpriseConfig(
                model_path=self.evidence_surprise_model_path,
                device=self.evidence_surprise_device,
                enabled=self.enable_evidence_surprise,
                max_spans=10,
                context_char_limit=0,
                span_selector_model=self.model,
            ),
            scorer=shared_scorer,
            llm_call=self._tracked_external_llm_call,
            llm_call_accepts_metadata=True,
        )
        self.retry_policy = RetryPolicy(
            RetryPolicyConfig(
                enabled=enable_review_retry,
                max_retry_per_step=max_retry_per_step,
            )
        )
        self.answer_candidate_tracker = AnswerCandidateTracker(
            compact_text=self._compact_for_decision_prompt,
            llm_generate=self._generate_answer_candidate_judgment,
        )
        self.task_retry_step_counts: Dict[str, int] = {}

    # ==================== Main Run Loop ====================

    def run(self, tasks: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Main execution loop — round-robin step-by-step across tasks.

        Args:
            tasks: List of task dicts, each containing:
                - task_id: Unique identifier
                - instruction: Task instruction string
                - mnt_dir: Host directory for task output files
                - source_dir: Host directory with source files
                - task_config: (optional) Original task config
                - env: (optional) DA_Agent_Env already created

        Returns:
            Dict mapping task_id to result info {"done": bool, "result": str}
        """
        # Reset state
        self.trajectory = []
        self.task_mnt_dirs = {}
        self.task_instructions = {}
        self.task_envs = {}
        self.task_history_messages = {}
        self.task_step_counts = {}
        self.task_done = {}
        self.task_results = {}
        self.task_pending_reviews = {}
        self.task_retry_step_counts = {}
        self.task_verified_outputs = {}
        self.task_quarantined_outputs = {}
        self.task_exploration_summaries = {}
        self._task_order = []
        self._rr_index = 0
        self._parse_failures = {}
        self._cost_step_context = {}
        self.retry_policy.reset()
        self.answer_candidate_tracker.reset()
        self.cost_tracker.reset()

        # Create per-run storage
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_storage_path = os.path.join(self.run_storage_base, f"run_{run_id}")
        os.makedirs(self._run_storage_path, exist_ok=True)
        logger.info(f"ACIDAgent run started, run storage: {self._run_storage_path}")

        # Phase 1: Setup all tasks
        for task_info in tasks:
            task_id = task_info["task_id"]
            mnt_dir = task_info["mnt_dir"]
            instruction = task_info["instruction"]

            self.task_mnt_dirs[task_id] = mnt_dir
            self.task_instructions[task_id] = instruction
            self.task_configs[task_id] = task_info["task_config"]
            self._task_order.append(task_id)

            # Use the env provided by caller
            self.task_envs[task_id] = task_info["env"]

            # Build system message
            action_space = "".join(
                [cls.get_action_description() for cls in self._AVAILABLE_ACTION_CLASSES]
            )
            system_message = {
                "role": "system",
                "content": SYS_PROMPT_ACID.format(
                    work_dir=DEFAULT_WORK_DIR,
                    action_space=action_space,
                    task=instruction,
                    max_steps=self.max_steps_per_task,
                ),
            }
            self.task_history_messages[task_id] = [system_message]

            # Initialize per-task state
            self.task_step_counts[task_id] = 0
            self.task_done[task_id] = False
            self.task_results[task_id] = None
            self.task_pending_reviews[task_id] = []
            self.task_retry_step_counts[task_id] = 0
            self.task_verified_outputs[task_id] = set()
            self.task_quarantined_outputs[task_id] = []
            self.answer_candidate_tracker.setup_task(task_id)
            self._parse_failures[task_id] = 0
            self.cost_tracker.start_task(task_id)

            logger.info(f"Setup task {task_id}")

            # Per-task log file
            task_log_path = os.path.join(self._run_storage_path, f"{task_id}.log")
            task_logger = logging.getLogger(f"da_agent.task.{task_id}")
            task_logger.setLevel(logging.DEBUG)
            # Prevent propagation to root logger (avoid duplicate in main log)
            task_logger.propagate = False
            task_fh = logging.FileHandler(task_log_path, encoding="utf-8")
            task_fh.setLevel(logging.DEBUG)
            task_fh.setFormatter(logging.Formatter(
                "[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"))
            task_logger.addHandler(task_fh)
            self._task_loggers[task_id] = task_logger

        # Cleanup leftover containers from previous crashed runs.
        # Disabled by default because separate ACID runs on the same Docker image
        # cannot see each other's active task_envs and would delete live containers.
        self._cleanup_stale_containers()

        # Phase 2: Round-robin execution
        while not self._all_tasks_done():
            # Pick next task
            task_id = self._next_task()
            if task_id is None:
                break

            # Execute one step for this task (may loop internally for Bash)
            self._execute_step(task_id)

        # Phase 3: Finalize
        return self._finalize_results()

    # ==================== Task Scheduling / Lifecycle ====================

    def _cleanup_stale_containers(self) -> None:
        """Stop and remove leftover containers from previous crashed runs.

        Only removes containers that share the same image but are NOT
        part of the current run (i.e., not in self.task_envs).
        """
        if os.environ.get("ACID_CLEANUP_STALE_CONTAINERS", "false").lower() != "true":
            return

        import docker
        try:
            client = docker.from_env()

            # Get the image ID for our target image
            try:
                target_image = client.images.get(self.image_name)
                target_image_id = target_image.id
            except Exception:
                return

            # Collect container names used by current run
            active_names = set()
            for env in self.task_envs.values():
                if hasattr(env, 'container') and env.container is not None:
                    active_names.add(env.container.name)
                elif hasattr(env, 'container_name'):
                    active_names.add(env.container_name)

            for container in client.containers.list():
                if container.image.id != target_image_id:
                    continue
                if container.name in active_names:
                    continue
                logger.info(f"Cleaning up stale container: {container.name}")
                try:
                    container.stop()
                    container.remove()
                except Exception as e:
                    logger.warning(f"Failed to cleanup container {container.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list containers for cleanup: {e}")

    def _next_task(self) -> Optional[str]:
        """Pick next task in round-robin order that is not done."""
        # Sequential mode: pick the first unfinished task in order
        if self.sequential_mode:
            for tid in self._task_order:
                if not self.task_done.get(tid):
                    return tid
            return None
        
        n = len(self._task_order)
        for i in range(n):
            task_id = self._task_order[(self._rr_index + i) % n]
            if not self.task_done.get(task_id):
                self._rr_index = (self._rr_index + i + 1) % n
                return task_id
        return None

    def _all_tasks_done(self) -> bool:
        """Check if all tasks are done."""
        return all(self.task_done.values()) if self.task_done else True

    def _consume_pending_reviews(self, task_id: str) -> List[Dict[str, Any]]:
        pending_reviews = self.task_pending_reviews.pop(task_id, [])
        self.task_pending_reviews[task_id] = []
        return pending_reviews

    @staticmethod
    def _retry_chain_root_from_pending(pending_reviews: List[Dict[str, Any]]) -> str:
        for ref in pending_reviews or []:
            root = ref.get("retry_chain_root_step_id")
            if root:
                return str(root)
            retry_decision = ref.get("retry_decision") or {}
            root = retry_decision.get("retry_chain_root_step_id")
            if root:
                return str(root)
            if ref.get("retry_review") and ref.get("step_id"):
                return str(ref.get("step_id"))
        return ""

    def _make_cycle_ids(self, task_id: str, is_retry_step: bool = False) -> Tuple[int, str, str]:
        if is_retry_step:
            retry_idx = self.task_retry_step_counts.get(task_id, 0)
            return retry_idx, f"{task_id}__retry_cycle_{retry_idx}", f"{task_id}__retry_{retry_idx}"
        cycle_idx = self.task_step_counts[task_id]
        return cycle_idx, f"{task_id}__cycle_{cycle_idx}", f"{task_id}__step_{cycle_idx}"

    # ==================== Cost Metrics ====================

    def _record_llm_usage(
        self,
        task_id: str,
        usage: Any,
        *,
        count_as_turn: bool = False,
        phase: str = "unknown",
        model: str = "",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        context = self._cost_step_context.get(task_id, {})
        self.cost_tracker.record_llm_call(
            task_id,
            usage,
            count_as_turn=count_as_turn,
            phase=phase,
            model=model,
            step_id=str(context.get("step_id") or ""),
            step_index=context.get("step_index"),
            attempt_index=context.get("attempt_index"),
            start_time=start_time,
            end_time=end_time,
        )

    def _call_llm_with_usage_tracked(
        self,
        task_id: str,
        phase: str,
        payload: Dict[str, Any],
        *,
        count_as_turn: bool = False,
    ) -> Tuple[bool, str, Any, str]:
        start_time = time.time()
        status, response, usage, reasoning = call_llm_with_usage(payload)
        end_time = time.time()
        self._record_llm_usage(
            task_id,
            usage,
            count_as_turn=count_as_turn,
            phase=phase,
            model=str(payload.get("model") or self.model or ""),
            start_time=start_time,
            end_time=end_time,
        )
        return status, response, usage, reasoning

    def _call_llm_tracked(
        self,
        task_id: str,
        phase: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, str]:
        status, response, _, _ = self._call_llm_with_usage_tracked(
            task_id,
            phase,
            payload,
        )
        return status, response

    def _tracked_external_llm_call(self, payload: Dict[str, Any]) -> Tuple[bool, str, Any, str]:
        task_id = str(payload.get("task_id") or "")
        phase = str(payload.get("llm_phase") or "external")
        clean_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"task_id", "llm_phase"}
        }
        return self._call_llm_with_usage_tracked(task_id, phase, clean_payload)

    def _mark_task_finished(self, task_id: str) -> None:
        self.cost_tracker.finish_task(task_id)

    def get_cost_metrics(self, task_id: str = "") -> Dict[str, Any]:
        if task_id:
            return self.cost_tracker.get(task_id)
        return self.cost_tracker.all_metrics()

    def get_cost_metric_events(self, task_id: str = "") -> Any:
        if task_id:
            return self.cost_tracker.get_events(task_id)
        return self.cost_tracker.all_events()

    def _write_cost_metrics(self) -> None:
        run_path = getattr(self, "_run_storage_path", "")
        if not run_path:
            return
        jsonl_path = os.path.join(run_path, "cost_metrics.jsonl")
        json_path = os.path.join(run_path, "cost_metrics.json")
        events_jsonl_path = os.path.join(run_path, "cost_metrics_events.jsonl")
        try:
            self.cost_tracker.write(jsonl_path, json_path, events_jsonl_path)
        except Exception as exc:
            logger.warning(f"Failed to write cost metrics: {exc}")
            return

        for task_id, metrics in self.cost_tracker.all_metrics().items():
            payload = "[COST-METRICS] " + json.dumps(metrics, ensure_ascii=False)
            logger.info(payload)
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(payload)

    # ==================== Step Execution Bookkeeping ====================

    def _execute_action_with_details(self, task_id: str, action: Action) -> Tuple[str, str, bool, List[str], Dict[str, Any]]:
        env = self.task_envs[task_id]
        file_snapshot = self._snapshot_mnt_files(env.mnt_dir)
        execution_meta: Dict[str, Any] = {}
        action_start_time = time.time()
        self.cost_tracker.record_code_execution(task_id, type(action).__name__)

        self._set_write_protection(env.mnt_dir, readonly=True,
                                   container=env.container, work_dir=env.work_dir)

        try:
            if hasattr(env, "step_with_details"):
                raw_observation, observation, done = env.step_with_details(action)
                execution_meta = dict(getattr(env, "last_execution_meta", {}) or {})
            else:
                observation, done = env.step(action)
                raw_observation = observation
                execution_meta = dict(getattr(env, "last_execution_meta", {}) or {})
        finally:
            self._set_write_protection(env.mnt_dir, readonly=False,
                                       container=env.container, work_dir=env.work_dir)

            self._freeze_verified_outputs(task_id)
        action_end_time = time.time()
        execution_meta.setdefault("action_type", type(action).__name__)
        execution_meta.setdefault(
            "timing",
            {
                "start_time": action_start_time,
                "end_time": action_end_time,
                "duration": round(max(0.0, action_end_time - action_start_time), 3),
            },
        )
        return raw_observation, observation, done, file_snapshot, execution_meta

    def _record_step_entry(self, task_id: str, cycle_id: str, step_id: str,
                           thought: str, action: Action, raw_observation: str,
                           observation: str, review_value: float, llm_usage: Dict,
                           pending_reviews: List[Dict[str, Any]], 
                           objective_confidence: Optional[Dict[str, Any]] = None,
                           is_retry_step: bool = False,
                           exploration_summary: str = "",
                           execution_meta: Optional[Dict[str, Any]] = None,
                           observation_digest: str = "") -> None:
        self.cost_tracker.record_step(task_id, is_retry_step=is_retry_step)
        step_entry = {
            "step_id": step_id,
            "cycle_id": cycle_id,
            "task_id": task_id,
            "type": "retry_review" if is_retry_step else "task_step",
            "thought": thought,
            "action_type": type(action).__name__,
            "action": str(action),
            "code": getattr(action, "code", ""),
            "raw_observation": raw_observation,
            "observation": observation,
            "observation_digest": observation_digest or build_observation_digest(
                observation,
                execution_meta,
            ),
            "execution_meta": execution_meta or {},
            "review_value": review_value,
            "review_decision": (
                (objective_confidence or {}).get("review_decision")
                if objective_confidence is not None
                else ("pass" if review_value else "retry")
            ),
            "llm_usage": llm_usage,
            "review_feedback": pending_reviews if pending_reviews else [],
            "exploration_summary": exploration_summary or "",
            "timestamp": datetime.now().isoformat(),
        }
    
        if objective_confidence is not None:
            step_entry["objective_confidence"] = objective_confidence

        if task_id in self._task_loggers:
            tl = self._task_loggers[task_id]
            tl.info(f"Step {step_id}: {action}")
            tl.info(f"Observation: {observation}")
            if raw_observation != observation:
                tl.info(f"Raw observation length: {len(raw_observation)}")
            tl.info(
                f"Step review decision: {step_entry['review_decision']}, "
                f"review value: {review_value}, LLM usage: {llm_usage}"
            )
            if objective_confidence is not None:
                tl.info("[OBJECTIVE-CONFIDENCE] " + json.dumps(objective_confidence, ensure_ascii=False))
            if is_retry_step:
                tl.info(f"Retry review step: {step_id}")
            if pending_reviews:
                tl.info(f"Review feedback applied: {pending_reviews}")
        self.trajectory.append(step_entry)

    def _finish_counted_step(self, task_id: str, observation: str) -> None:
        self.task_step_counts[task_id] += 1
        if self.task_step_counts[task_id] >= self.max_steps_per_task:
            self.task_done[task_id] = True
            self.task_results[task_id] = {
                "done": True,
                "result": observation if observation and observation.strip() else "Max steps reached",
            }
            self._mark_task_finished(task_id)

    def _finish_retry_step(self, task_id: str) -> None:
        self.task_retry_step_counts[task_id] = self.task_retry_step_counts.get(task_id, 0) + 1

    # ==================== Action Code Extraction / Scoring Utilities ====================

    @staticmethod
    def _extract_python_file_reference_from_bash(code: str) -> str:
        match = re.search(r"(?:^|\s)python3?\s+([^\s;&|]+\.py)\b", code or "")
        if not match:
            return ""
        return match.group(1).strip("'\"")

    def _lookup_previous_code_for_file(self, task_id: str, file_ref: str) -> str:
        if not file_ref:
            return ""
        normalized = file_ref.strip().lstrip("./")
        for entry in reversed(self.trajectory):
            if entry.get("task_id") != task_id:
                continue
            action_text = entry.get("action", "")
            code = entry.get("code", "")
            if not code:
                continue
            if file_ref in action_text or normalized in action_text or action_text.endswith(normalized):
                return code
        return ""

    def _get_action_code_for_scoring(self, action: Action, task_id: Optional[str] = None) -> str:
        if isinstance(action, Bash):
            raw_code = action.code or ""
            extracted = self._extract_python_code_from_bash(raw_code)
            if extracted:
                return extracted
            if task_id:
                file_ref = self._extract_python_file_reference_from_bash(raw_code)
                previous_code = self._lookup_previous_code_for_file(task_id, file_ref)
                if previous_code:
                    return previous_code
            return raw_code
        return getattr(action, "code", "") or ""

    @staticmethod
    def _action_type_for_scoring(action: Action) -> str:
        if isinstance(action, Terminate):
            return "Terminate"
        if isinstance(action, Python):
            return "Python"
        if isinstance(action, SQL):
            return "SQL"
        if isinstance(action, Bash):
            return "Bash"
        return type(action).__name__ if action is not None else ""

    # ==================== Shared Text / Embedding Utilities ====================

    @staticmethod
    def _compact_for_decision_prompt(text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        head_len = limit // 2
        tail_len = limit - head_len
        head = text[:head_len]
        tail = text[-tail_len:]
        return head + "\n... (middle truncated) ...\n" + tail

    @staticmethod
    def _objective_execution_needs_retry(confidence: Dict[str, Any]) -> bool:
        components = confidence.get("components", {}) or {}
        execution = components.get("execution_observation", {}) or {}
        return str(execution.get("status") or "").lower() == "retry"

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            parsed = float(value)
            if parsed != parsed:
                return default
            return parsed
        except Exception:
            return default

    @staticmethod
    def _append_objective_component(
        objective_confidence: Dict[str, Any],
        component_name: str,
        *,
        status: str,
        signals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        result = dict(objective_confidence or {})
        status = "watch" if status == "warning" else status
        components = dict(result.get("components", {}) or {})
        components[component_name] = {
            "status": status,
            "signals": list(signals or []),
        }
        result["components"] = components

        retry_reasons = list(result.get("retry_reasons", []) or [])
        watchlist = list(result.get("watchlist", []) or [])
        component_record = {
            "component": component_name,
            "status": status,
            "signals": list(signals or [])[:5],
        }
        if status == "retry":
            retry_reasons.append(component_record)
        elif status == "watch":
            watchlist.append(component_record)
        result["retry_reasons"] = retry_reasons
        result["watchlist"] = watchlist

        need_retry = bool(result.get("need_retry")) or status == "retry"
        result["need_retry"] = need_retry
        result["review_decision"] = "retry" if need_retry else "pass"
        return result

    @staticmethod
    def _json_object_from_text(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
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

    def _empty_probability_contrast_result(self, reason: str) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enable_probability_contrast),
            "model": self.probability_contrast_model_path,
            "available": False,
            "risk_status": "unavailable",
            "retry_recommendation": "",
            "min_ratio": None,
            "strongest_delta": None,
            "signals": [reason],
            "probes": [],
        }

    def _empty_evidence_surprise_result(self, reason: str) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enable_evidence_surprise),
            "model": self.evidence_surprise_model_path,
            "available": False,
            "max_span_surprise_score": 0.0,
            "risk_status": "unavailable",
            "retry_recommendation": "",
            "span_selection_method": "",
            "candidate_span_count": 0,
            "selected_span_count": 0,
            "signals": [reason],
            "spans": [],
        }

    def _attach_skipped_expensive_results(
        self,
        objective_confidence: Dict[str, Any],
        *,
        evidence_reason: str,
        probability_reason: str,
    ) -> Dict[str, Any]:
        result = dict(objective_confidence or {})
        evidence = self._empty_evidence_surprise_result(evidence_reason)
        probability = self._empty_probability_contrast_result(probability_reason)
        result["evidence_surprise"] = evidence
        result["evidence_surprise_max_span_score"] = 0.0
        result["evidence_surprise_recommendation"] = ""
        result["probability_contrast"] = probability
        result["probability_contrast_recommendation"] = ""
        result["probability_contrast_min_ratio"] = None
        signals = result.setdefault("signals", {})
        signals["evidence_surprise"] = evidence["signals"]
        signals["probability_contrast"] = probability["signals"]
        return result

    def _judge_observation_use(
        self,
        task_id: str,
        thought: str,
        action: Action,
        observation: str,
        execution_meta: Optional[Dict[str, Any]],
        observation_digest: str = "",
    ) -> Dict[str, Any]:
        if self._action_type_for_scoring(action) == "Terminate":
            return {"status": "pass", "reason": "terminate actions do not need observation-use judging"}
        stdout = ""
        if isinstance(execution_meta, dict) and "stdout" in execution_meta:
            stdout = str(execution_meta.get("stdout") or "")
        visible = stdout.strip() or (observation or "").strip()
        if not visible:
            return {"status": "pass", "reason": "judge skipped because stdout/observation is empty"}
        task_text = (
            self.task_configs.get(task_id, {}).get("instruction")
            or self.task_instructions.get(task_id, "")
        )
        prompt = OBSERVATION_USE_JUDGE_PROMPT.format(
            task=task_text,
            thought=thought or "",
            action=str(action),
            observation=observation_digest
            or build_observation_digest(observation, execution_meta, max_total_chars=3000),
        )
        try:
            status, response, _, _ = self._call_llm_with_usage_tracked(
                task_id,
                "observation_use_judge",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 500,
                },
            )
        except Exception as exc:
            return {"status": "unavailable", "reason": f"observation-use judge unavailable: {exc}"}
        if not status:
            return {"status": "unavailable", "reason": "observation-use judge call failed"}
        payload = self._json_object_from_text(response)
        if not payload:
            return {"status": "unavailable", "reason": "observation-use judge returned unparsable output"}
        judge_status = str(payload.get("status") or "pass").lower()
        if judge_status not in {"pass", "warning", "retry", "unavailable", "no_signal"}:
            judge_status = "unavailable"
        payload["status"] = judge_status
        return payload

    # ==================== Objective Confidence Signals ====================

    def _extract_current_decision_anchors(
        self,
        task_id: str,
        code: str,
        observation: str,
        structured_anchor_text: str,
        observation_digest: str = "",
    ) -> str:
        if (
            not self.enable_llm_current_extraction
            or not code.strip()
            or not (structured_anchor_text or "").strip()
        ):
            return ""
        task_text = (
            self.task_configs.get(task_id, {}).get("instruction")
            or self.task_instructions.get(task_id, "")
        )
        prompt = CURRENT_DECISION_EXTRACTION_PROMPT.format(
            task=task_text,
            evidence=self._compact_for_decision_prompt(structured_anchor_text, 6000),
            code=code,
            observation=observation_digest or self._compact_for_decision_prompt(observation, 3000),
        )
        try:
            success, response = self._call_llm_tracked(
                task_id,
                "current_decision_extraction",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 2000,
                },
            )
        except Exception as exc:
            logger.warning(f"[CURRENT-ANCHORS] extraction error: {exc}")
            return ""
        if not success or not response:
            logger.warning(f"[CURRENT-ANCHORS] extraction failed: {response}")
            return ""
        if "CURRENT_DECISION_ANCHORS" not in response:
            return "CURRENT_DECISION_ANCHORS:\n" + response
        return response

    def _score_probability_contrast(self, task_id: str, code: str,
                                    observation: str,
                                    scoring_evidence_text: str,
                                    structured_anchor_text: str,
                                    current_anchor_text: str = "",
                                    observation_digest: str = "") -> Dict[str, Any]:
        task_text = (
            self.task_configs.get(task_id, {}).get("instruction")
            or self.task_instructions.get(task_id, "")
        )
        if not current_anchor_text:
            current_anchor_text = self._extract_current_decision_anchors(
                task_id=task_id,
                code=code,
                observation=observation,
                structured_anchor_text=structured_anchor_text,
                observation_digest=observation_digest,
            )
        try:
            return self.probability_contrast_engine.evaluate(
                task_text=task_text,
                evidence_text=scoring_evidence_text,
                code=code,
                observation=observation_digest or observation,
                current_anchor_text=current_anchor_text,
            ).to_dict()
        except Exception as exc:
            logger.warning(f"[PROB-CONTRAST] hook error: {exc}")
            return self._empty_probability_contrast_result(
                f"probability contrast unavailable: {exc}"
            )

    def _score_evidence_surprise(self, task_id: str, code: str,
                                 action_type: str,
                                 evidence_text: str) -> Dict[str, Any]:
        task_text = (
            self.task_configs.get(task_id, {}).get("instruction")
            or self.task_instructions.get(task_id, "")
        )
        try:
            return self.evidence_surprise_evaluator.evaluate(
                task_text=task_text,
                evidence_text=evidence_text,
                code=code,
                action_type=action_type,
                task_id=task_id,
            ).to_dict()
        except Exception as exc:
            logger.warning(f"[EVIDENCE-SURPRISE] hook error: {exc}")
            return self._empty_evidence_surprise_result(
                f"evidence surprise unavailable: {exc}"
            )

    def _merge_probability_contrast_into_confidence(
        self,
        objective_confidence: Dict[str, Any],
        probability_contrast: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(objective_confidence or {})
        scored_probes = [
            probe for probe in probability_contrast.get("probes", [])
            if probe.get("current_avg_logp") is not None
            and probe.get("alternative_avg_logp") is not None
            and not probe.get("error")
        ]
        probability_used = bool(probability_contrast.get("available") and scored_probes)

        result["schema_version"] = result.get("schema_version", 9)
        result["probability_contrast"] = probability_contrast
        result["probability_contrast_recommendation"] = probability_contrast.get("retry_recommendation", "")
        result["probability_contrast_min_ratio"] = probability_contrast.get("min_ratio")
        signals = result.setdefault("signals", {})
        signals["probability_contrast"] = probability_contrast.get("signals", [])

        if not probability_used:
            return result

        status = str(probability_contrast.get("risk_status") or "").lower()
        min_ratio = probability_contrast.get("min_ratio")
        min_ratio_value = self._float_or_default(min_ratio, None)
        if status not in {"pass", "warning", "retry"}:
            if min_ratio_value is not None and min_ratio_value < 0.25:
                status = "retry"
            elif min_ratio_value is not None and min_ratio_value < 0.75:
                status = "warning"
            else:
                status = "pass"

        component_status = "watch" if status == "warning" else status
        if status == "retry":
            signals["probability_contrast"].append(
                "objective review set to retry because probability contrast strongly favors an evidence-supported alternative"
            )
        elif status == "warning":
            signals["probability_contrast"].append(
                "probability contrast added to watchlist; watch signal alone does not trigger retry"
            )
        return self._append_objective_component(
            result,
            "probability_contrast",
            status=component_status,
            signals=signals["probability_contrast"],
        )

    def _merge_evidence_surprise_into_confidence(
        self,
        objective_confidence: Dict[str, Any],
        evidence_surprise: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(objective_confidence or {})

        result["evidence_surprise"] = evidence_surprise
        result["evidence_surprise_max_span_score"] = evidence_surprise.get(
            "max_span_surprise_score", 0.0
        )
        result["evidence_surprise_recommendation"] = evidence_surprise.get("retry_recommendation", "")
        signals = result.setdefault("signals", {})
        signals["evidence_surprise"] = evidence_surprise.get("signals", [])

        scored_spans = [
            span for span in evidence_surprise.get("spans", [])
            if not span.get("error")
            and span.get("span_surprise_score") is not None
        ]
        surprise_used = bool(evidence_surprise.get("available") and scored_spans)
        if not surprise_used:
            return result

        status = str(evidence_surprise.get("risk_status") or "").lower()
        max_span_surprise = self._float_or_default(
            evidence_surprise.get("max_span_surprise_score"),
            0.0,
        )
        if status not in {"pass", "warning", "retry"}:
            if max_span_surprise > 0.5:
                status = "retry"
            elif max_span_surprise > 0.1:
                status = "warning"
            else:
                status = "pass"

        component_status = "watch" if status == "warning" else status
        if status == "retry":
            signals["evidence_surprise"].append(
                "objective review set to retry because evidence-conditioned code surprise found a high-risk span"
            )
        elif status == "warning":
            signals["evidence_surprise"].append(
                "evidence surprise added to watchlist; watch signal alone does not trigger retry"
            )
        return self._append_objective_component(
            result,
            "evidence_surprise",
            status=component_status,
            signals=signals["evidence_surprise"],
        )

    def _compute_objective_confidence(self, task_id: str, step_id: str,
                                      thought: str, action: Action,
                                      observation: str, exploration_summary: str,
                                      execution_meta: Optional[Dict[str, Any]] = None,
                                      observation_digest: str = "") -> Dict[str, Any]:
        code = self._get_action_code_for_scoring(action, task_id=task_id)
        observation_digest = observation_digest or build_observation_digest(
            observation,
            execution_meta,
            max_total_chars=3000,
        )

        evidence_bundle = self.exploration_evidence.build_bundle(
            task_id=task_id,
            task_exploration_summaries=self.task_exploration_summaries,
            current_summary=exploration_summary,
        )
        scoring_evidence_text = evidence_bundle.scoring_evidence_text
        structured_anchor_text = evidence_bundle.structured_anchor_text

        objective_confidence = self.objective_confidence_evaluator.evaluate(
            ObjectiveConfidenceInput(
                task_id=task_id,
                step_id=step_id,
                thought=thought,
                action_type=self._action_type_for_scoring(action),
                code=code,
                observation=observation,
                evidence_text=scoring_evidence_text,
                action_output=getattr(action, "output", ""),
                execution_meta=execution_meta or {},
            )
        )

        if self._objective_execution_needs_retry(objective_confidence):
            return self._attach_skipped_expensive_results(
                objective_confidence,
                evidence_reason="evidence surprise skipped because execution channel already requires retry",
                probability_reason="probability contrast skipped because execution channel already requires retry",
            )

        observation_use_judgment = self._judge_observation_use(
            task_id=task_id,
            thought=thought,
            action=action,
            observation=observation,
            execution_meta=execution_meta or {},
            observation_digest=observation_digest,
        )
        objective_confidence = self.objective_confidence_evaluator.evaluate(
            ObjectiveConfidenceInput(
                task_id=task_id,
                step_id=step_id,
                thought=thought,
                action_type=self._action_type_for_scoring(action),
                code=code,
                observation=observation,
                evidence_text=scoring_evidence_text,
                action_output=getattr(action, "output", ""),
                execution_meta=execution_meta or {},
                observation_use_judgment=observation_use_judgment,
            )
        )

        if bool(objective_confidence.get("need_retry")):
            return self._attach_skipped_expensive_results(
                objective_confidence,
                evidence_reason="evidence surprise skipped because execution/reasoning channel already requires retry",
                probability_reason="probability contrast skipped because execution/reasoning channel already requires retry",
            )

        evidence_surprise = self._score_evidence_surprise(
            task_id=task_id,
            code=code,
            action_type=self._action_type_for_scoring(action),
            evidence_text=scoring_evidence_text,
        )

        objective_confidence = self._merge_evidence_surprise_into_confidence(
            objective_confidence,
            evidence_surprise,
        )

        current_anchor_text = self._extract_current_decision_anchors(
            task_id=task_id,
            code=code,
            observation=observation,
            structured_anchor_text=structured_anchor_text,
            observation_digest=observation_digest,
        )

        probability_contrast = self._score_probability_contrast(
            task_id=task_id,
            code=code,
            observation=observation,
            scoring_evidence_text=scoring_evidence_text,
            structured_anchor_text=structured_anchor_text,
            current_anchor_text=current_anchor_text,
            observation_digest=observation_digest,
        )
        objective_confidence = self._merge_probability_contrast_into_confidence(
            objective_confidence,
            probability_contrast,
        )

        objective_confidence["objective_evidence_context"] = {
            "trusted_evidence_chars": len(scoring_evidence_text or ""),
            "structured_anchor_chars": len(structured_anchor_text or ""),
            "authoritative_fact_count": len(evidence_bundle.authoritative_facts),
            "verified_policy_count": len(evidence_bundle.verified_policies),
            "conflict_count": len(evidence_bundle.conflicts),
        }
        return objective_confidence

    # ==================== Retry Review / Repair Scheduling ====================

    def _maybe_schedule_review_retry(self, task_id: str, step_id: str,
                                     thought: str, action: Action,
                                     observation: str,
                                     exploration_summary: str,
                                     objective_confidence: Dict[str, Any],
                                     is_retry_step: bool = False,
                                     retry_chain_root_step_id: str = "",
                                     file_snapshot: Optional[List[str]] = None,
                                     observation_digest: str = ""):
        code = self._get_action_code_for_scoring(action, task_id=task_id)
        decision = self.retry_policy.evaluate(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action_text=str(action),
            code=code,
            action_type=self._action_type_for_scoring(action),
            observation=observation_digest or observation,
            exploration_summary=exploration_summary,
            objective_confidence=objective_confidence,
            is_retry_step=is_retry_step,
            retry_chain_root_step_id=retry_chain_root_step_id or step_id,
        )

        should_log = (
            decision.should_retry
            or decision.budget_exhausted
            or decision.step_exhausted
        )
        if should_log:
            logger.info("[RETRY-REVIEW] " + json.dumps(decision.to_dict(), ensure_ascii=False))
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(
                    "[RETRY-REVIEW] " + json.dumps(decision.to_dict(), ensure_ascii=False)
                )

        if decision.should_retry:
            quarantine_info = self._quarantine_step_outputs(task_id, step_id, file_snapshot)
            pending_feedback = decision.to_pending_feedback()
            if quarantine_info:
                pending_feedback["output_quarantine"] = quarantine_info
                quarantine_prompt = self._format_quarantine_prompt(quarantine_info)
                if quarantine_prompt:
                    pending_feedback["prompt_block"] = (
                        pending_feedback.get("prompt_block", "").rstrip()
                        + "\n\n"
                        + quarantine_prompt
                    )
                self._annotate_step_quarantine(step_id, quarantine_info)
            self.task_pending_reviews.setdefault(task_id, []).append(pending_feedback)
        return decision

    # ==================== Answer Candidate Tracking ====================

    def _maybe_register_answer_candidate(
        self,
        *,
        task_id: str,
        step_id: str,
        thought: str,
        action: Action,
        observation: str,
        objective_confidence: Dict[str, Any],
        review_decision: Any,
        is_retry_step: bool,
        pending_reviews: Optional[List[Dict[str, Any]]] = None,
        observation_digest: str = "",
        exploration_summary: str = "",
    ) -> AnswerCandidateResult:
        comparison_candidate_ids, comparison_candidate_signatures = (
            self._candidate_comparison_context(pending_reviews or [])
        )
        result = self.answer_candidate_tracker.register_step(
            task_id=task_id,
            task_instruction=(
                self.task_configs.get(task_id, {}).get("instruction")
                or self.task_instructions.get(task_id, "")
            ),
            step_id=step_id,
            thought=thought,
            action_text=str(action),
            action_output=getattr(action, "output", None),
            code=self._get_action_code_for_scoring(action, task_id=task_id),
            observation=observation_digest or observation,
            exploration_summary=exploration_summary,
            objective_confidence=objective_confidence,
            should_retry=bool(
                review_decision
                and (
                    getattr(review_decision, "should_retry", False)
                    or getattr(review_decision, "budget_exhausted", False)
                    or getattr(review_decision, "step_exhausted", False)
                )
            ),
            is_retry_step=is_retry_step,
            comparison_candidate_ids=comparison_candidate_ids,
            comparison_candidate_signatures=comparison_candidate_signatures,
        )
        if result.feedback:
            self.task_pending_reviews.setdefault(task_id, []).append(result.feedback)
        if result.log_payload:
            prefix = (
                "[ANSWER-CANDIDATE-COMPARISON] "
                if result.comparison_scheduled
                else "[ANSWER-CANDIDATE] "
            )
            logger.info(prefix + json.dumps(result.log_payload, ensure_ascii=False))
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(
                    prefix + json.dumps(result.log_payload, ensure_ascii=False)
                )
        if task_id in self._task_loggers:
            candidate = result.candidate or result.merged_candidate
            if candidate:
                self._task_loggers[task_id].info(
                    "[ANSWER-CANDIDATE-DETAIL] " + json.dumps(candidate, ensure_ascii=False)
                )
        return result

    @staticmethod
    def _candidate_comparison_context(
        pending_reviews: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        candidate_ids: List[str] = []
        candidate_signatures: List[str] = []
        for ref in pending_reviews or []:
            if not ref.get("candidate_comparison"):
                continue
            candidate_ids.extend(
                str(item) for item in (ref.get("candidate_ids") or [])
                if str(item or "").strip()
            )
            candidate_signatures.extend(
                str(item) for item in (ref.get("candidate_signatures") or [])
                if str(item or "").strip()
            )
        return candidate_ids, candidate_signatures

    def _generate_answer_candidate_judgment(
        self,
        prompt: str,
        task_id: str = "",
        phase: str = "answer_candidate_judge",
    ) -> str:
        try:
            success, response = self._call_llm_tracked(
                task_id,
                phase,
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 1000,
                },
            )
        except Exception as exc:
            logger.warning(f"[ANSWER-CANDIDATE-LLM] extraction error: {exc}")
            return ""
        if not success or not response:
            logger.warning(f"[ANSWER-CANDIDATE-LLM] extraction failed: {response}")
            return ""
        return response

    # ==================== Retry Evidence / Targeted Exploration Planning ====================

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
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

    def _find_trajectory_entry(self, step_id: str) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.trajectory):
            if entry.get("step_id") == step_id:
                return entry
        return None

    @staticmethod
    def _extract_source_observation_from_prompt(prompt_block: str) -> str:
        text = prompt_block or ""
        marker = "# OBSERVATION FROM THE SOURCE STEP #"
        if marker not in text:
            return ""
        tail = text.split(marker, 1)[1]
        for stop_marker in ("\n\n# INSTRUCTIONS #", "\n\n# QUARANTINED OUTPUTS #"):
            if stop_marker in tail:
                tail = tail.split(stop_marker, 1)[0]
        observation = tail.strip()
        return "" if observation == "(empty observation)" else observation

    def _retry_source_entries(self, pending_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen = set()
        for ref in pending_reviews:
            if not ref.get("retry_review"):
                continue
            step_id = ref.get("step_id") or (ref.get("retry_decision") or {}).get("step_id", "")
            if not step_id or step_id in seen:
                continue
            seen.add(step_id)
            entry = self._find_trajectory_entry(step_id)
            prompt_observation = self._extract_source_observation_from_prompt(ref.get("prompt_block", ""))
            entry_observation = (entry or {}).get("observation", "") if entry else prompt_observation
            entry_digest = (entry or {}).get("observation_digest", "") if entry else prompt_observation
            if entry and not entry_digest:
                entry_digest = build_observation_digest(
                    entry_observation,
                    (entry or {}).get("execution_meta", {}),
                )
            entries.append({
                "step_id": step_id,
                "entry": entry,
                "observation": entry_observation,
                "observation_digest": entry_digest,
                "prompt_observation": prompt_observation,
                "exploration_summary": (entry or {}).get("exploration_summary", "") if entry else "",
                "cycle_id": (entry or {}).get("cycle_id", "") if entry else "",
            })
        return entries

    @staticmethod
    def _is_empty_observation_text(observation: str) -> bool:
        text = (observation or "").strip()
        return text == "" or text == "(empty observation)"

    def _build_retry_issue_text(self, pending_reviews: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for ref in pending_reviews:
            if ref.get("retry_review") and ref.get("prompt_block"):
                parts.append(ref["prompt_block"])
                continue
            decision = ref.get("retry_decision") or {}
            issues = decision.get("issues") or []
            if issues:
                parts.append(json.dumps(issues, ensure_ascii=False))
        return "\n\n".join(parts)

    def _build_retry_existing_evidence(
        self,
        task_id: str,
        pending_reviews: List[Dict[str, Any]],
        source_entries: List[Dict[str, Any]],
    ) -> str:
        return self.exploration_evidence.build_retry_existing_evidence(
            task_id=task_id,
            task_exploration_summaries=self.task_exploration_summaries,
            source_entries=source_entries,
        )

    def _build_retry_source_exploration_context(
        self,
        pending_reviews: List[Dict[str, Any]],
    ) -> str:
        source_entries = self._retry_source_entries(pending_reviews)
        return self.exploration_evidence.build_retry_source_context(source_entries)

    def _log_retry_exploration_plan(self, task_id: str, plan: Dict[str, Any]) -> None:
        payload = dict(plan)
        for key in ("existing_evidence", "retry_issue_text"):
            if key in payload:
                payload[key + "_chars"] = len(payload.get(key) or "")
                payload.pop(key, None)
        line = "[RETRY-EXPLORATION-PLAN] " + json.dumps(payload, ensure_ascii=False)
        logger.info(line)
        if task_id in self._task_loggers:
            self._task_loggers[task_id].info(line)

    def _fallback_targeted_retry_objective(
        self,
        pending_reviews: List[Dict[str, Any]],
    ) -> str:
        checks: List[str] = []
        for ref in pending_reviews:
            decision = ref.get("retry_decision") or {}
            for issue in decision.get("issues") or []:
                checks.extend(issue.get("required_checks") or [])
                checks.extend(issue.get("alternative_paths") or [])
                checks.extend(issue.get("snippets") or [])
        compact_checks = [c for c in checks if c][:5]
        if compact_checks:
            return "Targeted retry exploration: inspect only the missing evidence needed for these review checks: " + "; ".join(compact_checks)
        return "Targeted retry exploration: inspect the smallest concrete evidence needed to resolve the retry review issue."

    def _format_retry_exploration_skip_summary(self, plan: Dict[str, Any]) -> str:
        missing = plan.get("missing_evidence") or []
        questions = plan.get("targeted_questions") or []
        lines = [
            "Retry evidence coverage decision: skip new exploration.",
            "Existing exploration summary and structured anchors are sufficient for the retry review.",
        ]
        if plan.get("reason"):
            lines.append(f"Coverage reason: {plan['reason']}")
        if missing:
            lines.append("Missing evidence judged non-blocking: " + "; ".join(map(str, missing[:4])))
        if questions:
            lines.append("Retry should address by code verification: " + "; ".join(map(str, questions[:4])))
        lines.append("Next action should use existing anchors and run targeted verification code if needed; do not repeat broad exploration.")
        return "\n".join(lines)

    def _plan_retry_exploration(
        self,
        task_id: str,
        pending_reviews: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_entries = self._retry_source_entries(pending_reviews)
        empty_sources = [
            src.get("step_id", "")
            for src in source_entries
            if self._is_empty_observation_text(src.get("observation") or src.get("prompt_observation") or "")
        ]
        if empty_sources:
            plan = {
                "mode": "full",
                "reason": "source step observation is empty; rebuild basic execution evidence before retrying",
                "source_step_ids": [src.get("step_id", "") for src in source_entries],
                "empty_source_step_ids": empty_sources,
                "targeted_objective": "",
                "missing_evidence": ["source action produced empty observation"],
                "targeted_questions": [],
                "judge": "rule_empty_observation",
            }
            self._log_retry_exploration_plan(task_id, plan)
            return plan

        existing_evidence = self._build_retry_existing_evidence(
            task_id,
            pending_reviews,
            source_entries,
        )
        retry_issue_text = self._build_retry_issue_text(pending_reviews)
        if not existing_evidence.strip():
            plan = {
                "mode": "targeted",
                "reason": "no usable exploration summary or structured anchors were found for the retry source step",
                "source_step_ids": [src.get("step_id", "") for src in source_entries],
                "targeted_objective": self._fallback_targeted_retry_objective(pending_reviews),
                "missing_evidence": ["retry source step has no attached exploration summary"],
                "targeted_questions": [],
                "judge": "rule_missing_existing_evidence",
            }
            self._log_retry_exploration_plan(task_id, plan)
            return plan

        task = self.task_instructions.get(task_id, "")
        source_observations = []
        for src in source_entries:
            source_observations.append(
                f"{src.get('step_id', 'source step')} observation digest:\n"
                f"{self._compact_for_decision_prompt(src.get('observation_digest') or src.get('prompt_observation') or '', 3000)}"
            )
        prompt = (
            "You are deciding whether a retry step needs more exploration before writing retry code.\n"
            "Use the existing exploration summary and structured evidence anchors if they already cover the retry issue.\n"
            "Do not ask for new exploration just because extra confirmation would be nice.\n\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            '  "mode": "skip|targeted|full",\n'
            '  "can_resolve_with_existing_evidence": true or false,\n'
            '  "reason": "brief reason",\n'
            '  "missing_evidence": ["facts that are actually missing"],\n'
            '  "targeted_questions": ["small checks to run if targeted exploration is needed"],\n'
            '  "targeted_objective": "one concise targeted exploration objective, or empty string"\n'
            "}\n\n"
            "Decision rules:\n"
            "- Use mode='skip' when the existing exploration summary / structured anchors plus source observation are enough to write targeted retry verification code.\n"
            "- Use mode='targeted' only when a small missing fact must be inspected before retry code can verify or repair the issue.\n"
            "- Use mode='full' only when the source observation is empty/unusable, schema/files are unknown, or the existing evidence is too thin to scope targeted checks.\n"
            "- Targeted exploration should be one narrow read-only inspection, not a full task exploration.\n\n"
            f"# TASK #\n{task}\n\n"
            f"# RETRY ISSUE / REVIEW PROMPT #\n{self._compact_for_decision_prompt(retry_issue_text, 8000)}\n\n"
            f"# SOURCE STEP OBSERVATION(S) #\n{self._compact_for_decision_prompt(chr(10).join(source_observations), 3000)}\n\n"
            f"# EXISTING EXPLORATION SUMMARY AND STRUCTURED ANCHORS #\n{self._compact_for_decision_prompt(existing_evidence, 5000)}\n"
        )
        try:
            success, response = self._call_llm_tracked(
                task_id,
                "retry_exploration_plan",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 1000,
                },
            )
        except Exception as exc:
            logger.warning(f"[RETRY-EXPLORATION-PLAN] judge error: {exc}")
            success, response = False, ""
        parsed = self._parse_json_object(response if success else "")
        if parsed is None:
            plan = {
                "mode": "targeted",
                "reason": "coverage judge failed; run one narrow targeted exploration as fallback",
                "source_step_ids": [src.get("step_id", "") for src in source_entries],
                "targeted_objective": self._fallback_targeted_retry_objective(pending_reviews),
                "missing_evidence": ["coverage judge did not return valid JSON"],
                "targeted_questions": [],
                "judge": "fallback_invalid_json",
            }
            self._log_retry_exploration_plan(task_id, plan)
            return plan

        mode = str(parsed.get("mode") or "").strip().lower()
        if mode not in {"skip", "targeted", "full"}:
            mode = "targeted"
        if mode == "full":
            mode = "targeted"
            parsed["reason"] = (
                "coverage judge requested full exploration, but source observation is non-empty; "
                + str(parsed.get("reason") or "")
            ).strip()
        objective = str(parsed.get("targeted_objective") or "").strip()
        if mode == "targeted" and not objective:
            questions = parsed.get("targeted_questions") or parsed.get("missing_evidence") or []
            if isinstance(questions, list) and questions:
                objective = "Targeted retry exploration: " + "; ".join(str(q) for q in questions[:4])
            else:
                objective = self._fallback_targeted_retry_objective(pending_reviews)
        plan = {
            "mode": mode,
            "can_resolve_with_existing_evidence": bool(parsed.get("can_resolve_with_existing_evidence")),
            "reason": str(parsed.get("reason") or ""),
            "source_step_ids": [src.get("step_id", "") for src in source_entries],
            "targeted_objective": objective,
            "missing_evidence": parsed.get("missing_evidence") if isinstance(parsed.get("missing_evidence"), list) else [],
            "targeted_questions": parsed.get("targeted_questions") if isinstance(parsed.get("targeted_questions"), list) else [],
            "judge": "llm",
            "existing_evidence": existing_evidence,
            "retry_issue_text": retry_issue_text,
        }
        self._log_retry_exploration_plan(task_id, plan)
        return plan

    # ==================== Main Step Execution Pipeline ====================

    def _execute_step(self, task_id: str) -> bool:
        """
        Execute one step for a task using the persistent container.

        Both Bash and non-Bash actions count toward step limit.

        Returns:
            True if a step was executed (counts toward global step limit)
        """
        pending_reviews = self._consume_pending_reviews(task_id)
        is_retry_step = any(ref.get("retry_review") for ref in pending_reviews)
        retry_chain_root_step_id = (
            self._retry_chain_root_from_pending(pending_reviews)
            if is_retry_step else ""
        )
        obs = self._build_observation(task_id, pending_reviews)

        steps_remaining = self.max_steps_per_task - self.task_step_counts[task_id]
        if not is_retry_step and steps_remaining <= 1:
            obs += (
                "\n\n# STEP BUDGET WARNING #\n"
                "This is your LAST step. You MUST call Terminate(output=\"<your answer>\") now.\n"
                "Use the best answer you have computed so far from prior observations.\n"
                "If no answer was computed, output Terminate(output=\"FAIL\").\n"
            )

        if pending_reviews:
            sources = []
            for ref in pending_reviews:
                src = f"step={ref.get('step_id','')}"
                if ref.get("retry_review"):
                    review_source = "candidate_comparison" if ref.get("candidate_comparison") else "objective_review"
                    src += f", review_source={review_source}"
                sources.append(src)
            logger.info(f"Task {task_id} step uses review feedback: {sources}")

        cycle_idx, cycle_id, step_id = self._make_cycle_ids(task_id, is_retry_step=is_retry_step)
        self._cost_step_context[task_id] = {
            "step_id": step_id,
            "step_index": None if is_retry_step else cycle_idx,
            "attempt_index": cycle_idx if is_retry_step else 0,
            "is_retry_step": is_retry_step,
        }
        if not retry_chain_root_step_id:
            retry_chain_root_step_id = step_id

        retry_exploration_plan = None

        if is_retry_step:
            retry_exploration_plan = self._plan_retry_exploration(task_id, pending_reviews)
            mode = retry_exploration_plan.get("mode", "targeted")
            if mode == "skip":
                exploration_summary = self._format_retry_exploration_skip_summary(retry_exploration_plan)
            elif mode == "targeted":
                exploration_summary = self._run_exploration(
                    task_id,
                    cycle_id,
                    pending_reviews,
                    forced_objective=retry_exploration_plan.get("targeted_objective") or None,
                    max_rounds=1,
                )
            else:
                exploration_summary = self._run_exploration(task_id, cycle_id, pending_reviews)
        else:
            exploration_summary = self._run_exploration(task_id, cycle_id, pending_reviews)

        source_retry_context = ""
        if is_retry_step:
            source_retry_context = self._build_retry_source_exploration_context(pending_reviews)

        effective_exploration_summary = exploration_summary
        if source_retry_context:
            effective_exploration_summary = self.exploration_evidence.build_effective_summary(
                exploration_summary,
                source_retry_context,
            )

            obs += (
                "\n\n# SOURCE STEP ORIGINAL EXPLORATION SUMMARY #\n"
                f"{source_retry_context}\n"
                "\n# RETRY EVIDENCE USE RULE #\n"
                "Use the source step original exploration summary to preserve the original task scope, "
                "grain, and already verified counts. Use the current retry exploration summary only to "
                "repair the immediate retry issue. If a repair changes a previously verified count, "
                "population scope, or grouping grain, print the before/after counts and justify the change "
                "with evidence before trusting the result.\n"
            )

        if exploration_summary and exploration_summary.strip().upper() not in ("SKIP", "SUFFICIENT EVIDENCE; NO NEW EXPLORATION NEEDED."):
            summary_header = (
                "# CURRENT RETRY EXPLORATION SUMMARY #"
                if source_retry_context else
                "# EXPLORATION SUMMARY #"
            )
            obs += (
                f"\n\n{summary_header}\n"
                f"{exploration_summary}\n"
            )

        action, thought, llm_usage = self._predict(task_id, obs)
        if action is None:
            self._parse_failures[task_id] += 1
            logger.warning(f"Parse failure for task {task_id} "
                           f"(count={self._parse_failures[task_id]})")
            if self._parse_failures[task_id] > 3:
                self.task_done[task_id] = True
                self.task_results[task_id] = {"done": False, "result": "Parse failure"}
                self._mark_task_finished(task_id)
            return False
        self._parse_failures[task_id] = 0

        raw_observation, observation, done, file_snapshot, execution_meta = self._execute_action_with_details(task_id, action)
        observation_digest = build_observation_digest(
            observation,
            execution_meta,
            max_total_chars=3000,
        )

        logger.info(f"Task {task_id} step {step_id}: {action}")
        logger.info(f"Observation: {observation[:500]}")

        objective_confidence = self._compute_objective_confidence(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action=action,
            observation=observation,
            exploration_summary=effective_exploration_summary,
            execution_meta=execution_meta,
            observation_digest=observation_digest,
        )
        logger.info("[OBJECTIVE-CONFIDENCE] " +
                    json.dumps(objective_confidence, ensure_ascii=False))
        step_review_value = 0.0 if objective_confidence.get("need_retry") else 1.0
        
        self._record_step_entry(task_id, cycle_id, step_id, thought, action,
                                raw_observation, observation, step_review_value,
                                llm_usage, pending_reviews, 
                                objective_confidence=objective_confidence,
                                is_retry_step=is_retry_step,
                                exploration_summary=effective_exploration_summary,
                                execution_meta=execution_meta,
                                observation_digest=observation_digest)

        if done or isinstance(action, Terminate):
            review_decision = None
            review_scheduled = False
            review_blocked = False
            if bool(objective_confidence.get("need_retry")):
                review_decision = self._maybe_schedule_review_retry(
                    task_id=task_id,
                    step_id=step_id,
                    thought=thought,
                    action=action,
                    observation=observation,
                    observation_digest=observation_digest,
                    exploration_summary=effective_exploration_summary,
                    objective_confidence=objective_confidence,
                    is_retry_step=is_retry_step,
                    retry_chain_root_step_id=retry_chain_root_step_id,
                    file_snapshot=file_snapshot,
                )
                review_scheduled = bool(review_decision and review_decision.should_retry)
                review_blocked = bool(
                    review_decision
                    and (
                        review_decision.budget_exhausted
                        or review_decision.step_exhausted
                    )
                )
                if review_scheduled:
                    logger.info(f"Terminate step {step_id} deferred for review retry")
                    if is_retry_step:
                        self._finish_retry_step(task_id)
                    return False

            candidate_result = self._maybe_register_answer_candidate(
                task_id=task_id,
                step_id=step_id,
                thought=thought,
                action=action,
                observation=observation,
                observation_digest=observation_digest,
                objective_confidence=objective_confidence,
                review_decision=review_decision,
                is_retry_step=is_retry_step,
                pending_reviews=pending_reviews,
                exploration_summary=effective_exploration_summary,
            )
            candidate_comparison_scheduled = bool(
                candidate_result and candidate_result.comparison_scheduled
            )
            if candidate_comparison_scheduled:
                if is_retry_step:
                    self._finish_retry_step(task_id)
                return False
            if self._should_update_memory_for_step(
                review_decision=review_decision,
                review_scheduled=review_scheduled,
                review_blocked=review_blocked,
                objective_confidence=objective_confidence,
            ):
                self._update_history(task_id, observation_digest, thought, action)
            else:
                logger.info(f"Skip memory update for unaccepted step {step_id}")
            self.task_done[task_id] = True
            self.task_results[task_id] = {
                "done": True,
                "result": getattr(action, "output", ""),
            }
            self._mark_task_finished(task_id)
            if is_retry_step:
                self._finish_retry_step(task_id)
            return not is_retry_step

        review_decision = self._maybe_schedule_review_retry(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action=action,
            observation=observation,
            observation_digest=observation_digest,
            exploration_summary=effective_exploration_summary,
            objective_confidence=objective_confidence,
            is_retry_step=is_retry_step,
            retry_chain_root_step_id=retry_chain_root_step_id,
            file_snapshot=file_snapshot,
        )
        review_scheduled = bool(review_decision and review_decision.should_retry)
        review_blocked = bool(
            review_decision
            and (
                review_decision.budget_exhausted
                or review_decision.step_exhausted
            )
        )
        candidate_result = self._maybe_register_answer_candidate(
            task_id=task_id,
            step_id=step_id,
            thought=thought,
            action=action,
            observation=observation,
            observation_digest=observation_digest,
            objective_confidence=objective_confidence,
            review_decision=review_decision,
            is_retry_step=is_retry_step,
            pending_reviews=pending_reviews,
            exploration_summary=effective_exploration_summary,
        )
        candidate_comparison_scheduled = bool(
            candidate_result and candidate_result.comparison_scheduled
        )
        review_scheduled = review_scheduled or candidate_comparison_scheduled
        if self._should_update_memory_for_step(
            review_decision=review_decision,
            review_scheduled=review_scheduled,
            review_blocked=review_blocked,
            objective_confidence=objective_confidence,
        ):
            self._update_history(task_id, observation_digest, thought, action)
        else:
            logger.info(f"Skip memory update for unaccepted step {step_id}")

        if is_retry_step:
            if not review_scheduled and not review_blocked:
                self._mark_verified_outputs(task_id, step_id, file_snapshot)
            self._finish_retry_step(task_id)
            return False

        self._finish_counted_step(task_id, observation)

        if not review_scheduled and not review_blocked:
            self._mark_verified_outputs(task_id, step_id, file_snapshot)

        return True

    # ==================== Adaptive Exploration ====================

    def _build_known_evidence(
        self,
        task_id: str,
        current_local_summaries: Optional[List[str]] = None,
    ) -> str:
        return self.exploration_evidence.build_known_evidence(
            self.task_exploration_summaries,
            task_id,
            current_local_summaries=current_local_summaries,
        )

    def _derive_exploration_task(
        self,
        task_id: str,
        review_feedback: str,
        current_local_summaries: Optional[List[str]] = None,
    ) -> str:
        instruction = self.task_instructions.get(task_id, "")
        known_evidence = self._build_known_evidence(
            task_id,
            current_local_summaries=current_local_summaries,
        )
        prompt = EXPLORATION_TASK_PROMPT.format(
            task=instruction,
            known_evidence=known_evidence,
            review_feedback=review_feedback,
        )
        status, response, _, _ = self._call_llm_with_usage_tracked(
            task_id,
            "exploration_task",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if not status:
            return "Inspect the data files and schema."
        objective = response.strip()
        return objective.split("Exploration objective:")[-1].strip() if "Exploration objective:" in objective else objective

    def _predict_exploration_action(
        self,
        task_id: str,
        objective: str,
        current_local_summaries: Optional[List[str]] = None,
    ) -> Tuple[Optional[Action], str, str]:
        instruction = self.task_instructions.get(task_id, "")
        known_evidence = self._build_known_evidence(
            task_id,
            current_local_summaries=current_local_summaries,
        )
        action_space = "".join(
            [cls.get_action_description() for cls in self._AVAILABLE_ACTION_CLASSES]
        )
        prompt = EXPLORATION_ACTION_PROMPT.format(
            action_space=action_space,
            task=instruction,
            objective=objective,
            context=known_evidence,
        )
        status, response, _, reasoning = self._call_llm_with_usage_tracked(
            task_id,
            "exploration_action",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if not status:
            return None, "", ""
        response = response.strip()
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if not thought and reasoning:
            thought = reasoning.strip()
        action = self._parse_action(response)
        return action, thought, response

    def _summarize_exploration(self, task_id: str, objective: str, action: Action,
                               observation: str) -> str:
        instruction = self.task_instructions.get(task_id, "")
        action_str = str(action)
        prompt = EXPLORATION_SUMMARY_PROMPT.format(
            task=instruction,
            objective=objective,
            action=action_str,
            observation=observation,
        )
        status, response, _, _ = self._call_llm_with_usage_tracked(
            task_id,
            "exploration_summary",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if not status:
            return f"Explored: {objective}. Output: {observation[:500]}"
        summary = response.strip()
        return summary.split("Concise exploration summary:")[-1].strip() if "Concise exploration summary:" in summary else summary

    def _merge_exploration_summaries(
        self,
        task_id: str,
        summaries: List[str],
    ) -> Dict[str, Any]:
        return self.exploration_evidence.merge_summaries(summaries)

    def _format_exploration_merge_state(self, merge_state: Dict[str, Any]) -> str:
        return self.exploration_evidence.format_merge_state(merge_state)

    def _finalize_exploration_summary(
        self,
        task_id: str,
        merge_state: Dict[str, Any],
    ) -> str:
        instruction = self.task_instructions.get(task_id, "")
        return self.exploration_evidence.finalize_summary(
            task_id=task_id,
            task_instruction=instruction,
            merge_state=merge_state,
        )

    def _exploration_merge_needs_conflict_revision(self, merge_state: Dict[str, Any]) -> bool:
        return self.exploration_evidence.merge_needs_conflict_revision(merge_state)

    def _resolve_exploration_conflicts(
        self,
        task_id: str,
        merge_state: Dict[str, Any],
        summaries: List[str],
    ) -> Dict[str, Any]:
        instruction = self.task_instructions.get(task_id, "")
        return self.exploration_evidence.resolve_conflicts(
            task_id=task_id,
            task_instruction=instruction,
            merge_state=merge_state,
            summaries=summaries,
        )

    def _apply_conflict_revision_state(
        self,
        merge_state: Dict[str, Any],
        revision_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.exploration_evidence.apply_conflict_revision_state(
            merge_state,
            revision_state,
        )

    def _max_explore_rounds_for_current_step(
        self,
        task_id: str,
        pending_reviews: List[Dict[str, Any]],
    ) -> int:
        if any(ref.get("retry_review") for ref in pending_reviews or []):
            return max(1, int(self.max_explore_steps))

        step_index = max(0, int(self.task_step_counts.get(task_id, 0)))
        decayed_rounds = int(self.max_explore_steps) - (step_index // 2)
        return max(1, int(self.min_explore_steps), decayed_rounds)

    def _run_exploration(
        self,
        task_id: str,
        cycle_id: str,
        pending_reviews: List[Dict[str, Any]],
        forced_objective: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ) -> str:
        review_feedback_str = ""
        if pending_reviews:
            feedback_parts = []
            for ref in pending_reviews:
                if ref.get("retry_review") and ref.get("prompt_block"):
                    feedback_parts.append(ref["prompt_block"])
            review_feedback_str = "\n".join(feedback_parts)

        all_summaries = []
        prior_observations = []

        is_retry_exploration = any(ref.get("retry_review") for ref in pending_reviews or [])
        rounds = (
            max(1, int(max_rounds))
            if max_rounds is not None
            else self._max_explore_rounds_for_current_step(task_id, pending_reviews)
        )
        if task_id in self._task_loggers:
            self._task_loggers[task_id].info(
                f"Exploration round cap: {rounds} "
                f"(step_index={self.task_step_counts.get(task_id, 0)}, retry={is_retry_exploration})"
            )
        for explore_i in range(rounds):
            if forced_objective and explore_i == 0:
                objective = forced_objective
            else:
                objective = self._derive_exploration_task(
                    task_id,
                    review_feedback_str,
                    current_local_summaries=all_summaries,
                )

            if objective.strip().upper() == "SKIP":
                if not all_summaries:
                    summary = "Sufficient evidence; no new exploration needed."
                    self.task_exploration_summaries.setdefault(task_id, []).append(summary)
                break

            action, thought, raw_response = self._predict_exploration_action(
                task_id,
                objective,
                current_local_summaries=all_summaries,
            )

            if action is None:
                summary = f"Exploration objective: {objective}. No action generated."
                self.task_exploration_summaries.setdefault(task_id, []).append(summary)
                all_summaries.append(summary)
                break

            raw_observation, observation, done, file_snapshot, execution_meta = self._execute_action_with_details(task_id, action)
            observation_digest = build_observation_digest(
                observation,
                execution_meta,
                max_total_chars=3000,
            )
            logger.info(f"Task {task_id} exploration: {action}")
            logger.info(f"Exploration observation: {observation[:500]}")

            explore_step_id = f"{cycle_id}_explore_{explore_i}"
            self._cleanup_exploration_outputs(task_id, explore_step_id, file_snapshot)
            explore_entry = {
                "step_id": explore_step_id,
                "cycle_id": cycle_id,
                "task_id": task_id,
                "type": "exploration",
                "thought": thought,
                "action_type": type(action).__name__,
                "action": str(action),
                "code": getattr(action, "code", ""),
                "raw_observation": raw_observation,
                "observation": observation,
                "observation_digest": observation_digest,
                "execution_meta": execution_meta,
                "objective": objective,
                "explore_index": explore_i,
                "timestamp": datetime.now().isoformat(),
            }
            self.trajectory.append(explore_entry)
            if task_id in self._task_loggers:
                tl = self._task_loggers[task_id]
                tl.info(f"Exploration {explore_step_id}: {action}")
                tl.info(f"Exploration observation: {observation}")

            summary = self._summarize_exploration(task_id, objective, action, observation)
            explore_entry["summary"] = summary

            self.task_exploration_summaries.setdefault(task_id, []).append(summary)
            all_summaries.append(summary)
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(f"Exploration summary [{explore_i}]: {summary}")

            # ===== ROUND-NOV early-stop (替代 LLM 的 sufficiency 判断) =====
            steps_done = explore_i + 1
            if prior_observations:
                try:
                    from da_agent.utils.adaptive_exploration_metric import compute_round_novelty
                    nov = compute_round_novelty(prior_observations, observation)
                except Exception as _e:
                    nov = None
                    logger.warning(f"[ROUND-NOV] compute error: {_e}")
                if nov is not None:
                    redundant = nov["PMI_per_tok"] > self.roundnov_stop_threshold
                    if task_id in self._task_loggers:
                        self._task_loggers[task_id].info(
                            f"[ROUND-NOV] explore_{explore_i}: best-match=explore_{nov['match_round']} "
                            f"n={nov['n']} PMI={nov['PMI']} PMI/tok={nov['PMI_per_tok']} "
                            f"S_COM={nov['S_COM']} -> {'REDUNDANT' if redundant else 'NEW'}"
                        )
                    if redundant and steps_done >= self.min_explore_steps:
                        if task_id in self._task_loggers:
                            self._task_loggers[task_id].info(
                                f"Exploration stopped early (skip): redundant round, "
                                f"PMI/tok > {self.roundnov_stop_threshold}, after {steps_done} step(s)."
                            )
                        break
            prior_observations.append(observation)
            # ===== END ROUND-NOV early-stop =====


        if not all_summaries:
            return "Sufficient evidence; no new exploration needed."

        merge_state = self._merge_exploration_summaries(task_id, all_summaries)
        if task_id in self._task_loggers:
            self._task_loggers[task_id].info(
                "Exploration merge state: "
                + self._compact_for_decision_prompt(
                    self._format_exploration_merge_state(merge_state),
                    5000,
                )
            )

        if self._exploration_merge_needs_conflict_revision(merge_state):
            revision_state = self._resolve_exploration_conflicts(
                task_id=task_id,
                merge_state=merge_state,
                summaries=all_summaries,
            )
            merge_state = self._apply_conflict_revision_state(merge_state, revision_state)
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(
                    "Exploration conflict revision state: "
                    + self._compact_for_decision_prompt(
                        json.dumps(revision_state, ensure_ascii=False, indent=2),
                        5000,
                    )
                )
                self._task_loggers[task_id].info(
                    "Exploration merge state after conflict revision: "
                    + self._compact_for_decision_prompt(
                        self._format_exploration_merge_state(merge_state),
                        5000,
                    )
                )

        final_summary = self._finalize_exploration_summary(task_id, merge_state)
        self.task_exploration_summaries.setdefault(task_id, []).append(final_summary)
        if task_id in self._task_loggers:
            self._task_loggers[task_id].info(
                "Resolved exploration summary: "
                + self._compact_for_decision_prompt(final_summary, 10000)
            )
        return final_summary

    # ==================== LLM Prediction ====================

    def _predict(self, task_id: str, obs: str) -> Tuple[Optional[Action], str, Dict]:
        """Call LLM to get next action. Returns (action, thought, usage)."""
        messages = self.task_history_messages[task_id].copy()
        obs_text = (obs or "").strip()
        if obs_text:
            current_observation = f"Observation: {obs}"
            if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = (
                    messages[-1].get("content", "").rstrip()
                    + "\n\n# CURRENT CYCLE CONTEXT #\n"
                    + current_observation
                )
            else:
                messages.append({
                    "role": "user",
                    "content": current_observation,
                })
        elif not messages or messages[-1].get("role") != "user":
            messages.append({
                "role": "user",
                "content": (
                    "Observation: No new observation was produced in this cycle. "
                    "Use the latest accepted observations and evidence already in history; "
                    "do not treat this as a failed or empty execution."
                ),
            })

        # Log full LLM prompt for debugging
        logger.info(f"Task {task_id} LLM prompt ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = str(content)
            logger.info(f"  [{i}] role={role}, length={len(content)}:\n{content}")
        if task_id in self._task_loggers:
            tl = self._task_loggers[task_id]
            tl.info(f"LLM prompt ({len(messages)} messages):")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = str(content)
                tl.info(f"  [{i}] role={role}:\n{content}")

        status, response, usage, reasoning = self._call_llm_with_usage_tracked(
            task_id,
            "predict_action",
            {
                "model": self.model,
                "messages": messages,
                # "max_tokens": self.max_tokens,
            },
            count_as_turn=True,
        )

        # Log token usage
        if usage:
            logger.info(f"Task {task_id} LLM tokens: {usage}")
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(f"LLM tokens: {usage}")

        if not status:
            logger.warning(f"LLM call failed for task {task_id}")
            return None, "", usage or {}

        response = response.strip()
        logger.info(f"Task {task_id} LLM response: {response[:500]}")
        if task_id in self._task_loggers:
            self._task_loggers[task_id].info(f"LLM response: {response}")

        # Log reasoning content for debugging thinking models
        if reasoning:
            logger.info(f"Task {task_id} LLM reasoning: {reasoning[:500]}")
            if task_id in self._task_loggers:
                self._task_loggers[task_id].info(f"LLM reasoning: {reasoning}")

        # Extract thought. Confidence is intentionally not requested or parsed.
        thought_match = re.search(
            r'Thought:\s*(.*?)(?=Action:|$)', response, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # For thinking models, reasoning may contain the actual thought
        if not thought and reasoning:
            thought = reasoning.strip()

        # Tolerate stale model outputs that still include a Confidence line.
        clean_response = re.sub(
            r'(?im)^\s*Confidence:\s*(?:HIGH|MEDIUM|LOW|[0-9.]+)\s*$\n?',
            '',
            response,
        )
        action = self._parse_action(clean_response)

        if action is None:
            logger.warning(f"Failed to parse action from LLM response for task {task_id}")
            return None, thought, usage or {}

        return action, thought, usage or {}

    def _parse_action(self, output: str) -> Optional[Action]:
        """Parse action from LLM response text."""
        if output is None or len(output) == 0:
            return None

        action_string = ""
        patterns = [
            r'["\']?Action["\']?:? (.*?)Observation',
            r'["\']?Action["\']?:? (.*?)Thought',
            r'["\']?Action["\']?:? (.*?)$',
            r'^(.*?)Observation',
        ]

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break

        if action_string == "":
            action_string = output.strip()

        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                return action

        action_string = action_string.replace("\\_", "_").replace("'''", "```")
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                return action

        return None

    # ==================== History Management ====================

    @staticmethod
    def _should_update_memory_for_step(
        *,
        review_decision: Any = None,
        review_scheduled: bool = False,
        review_blocked: bool = False,
        objective_confidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if review_scheduled or review_blocked:
            return False
        if review_decision and (
            getattr(review_decision, "should_retry", False)
            or getattr(review_decision, "budget_exhausted", False)
            or getattr(review_decision, "step_exhausted", False)
        ):
            return False
        objective_confidence = objective_confidence or {}
        if "need_retry" in objective_confidence:
            return not bool(objective_confidence.get("need_retry"))
        status = str(objective_confidence.get("review_decision") or "").lower()
        return status != "retry"

    def _build_observation(
        self,
        task_id: str,
        pending_reviews: List[Dict[str, Any]],
    ) -> str:
        """Build observation prefix with review-retry feedback."""
        obs = ""

        for ref in pending_reviews:
            if ref.get("retry_review"):
                prompt_block = ref.get("prompt_block", "").strip()
                if prompt_block:
                    obs += prompt_block + "\n\n"
                else:
                    step_label = self._get_step_label(ref.get("step_id", "previous_step"))
                    obs += "# RETRY REVIEW #\n"
                    obs += f"{step_label} needs targeted review before it is trusted.\n\n"
                continue

        return obs
    
    def _update_history(self, task_id: str, observation_digest: str,
                        thought: str, action: Action) -> None:
        """Update LLM message history with compact observation context."""
        messages = self.task_history_messages[task_id]
        # Include step_id in observation so LLM can reference steps later
        step_idx = self.task_step_counts.get(task_id, 0)
        step_label = f"[step_{step_idx}] "

        messages.append({
            "role": "assistant",
            "content": f"Thought: {thought}\n\nAction: {str(action)}",
        })
        messages.append({
            "role": "user",
            "content": f"{step_label}Observation:\n{observation_digest}",
        })

        # Trim to max_memory_length (system message + pairs of user/assistant)
        if len(messages) > self.max_memory_length * 2 + 1:
            messages[:] = [messages[0]] + messages[-self.max_memory_length * 2:]

    def _get_step_label(self, step_id: str) -> str:
        """Extract step label from step_id, consistent with history format."""
        # step_id format: "{task_id}__step_{N}" -> "[step_N]"
        # Same format as _update_history uses
        parts = step_id.rsplit("__step_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"[step_{parts[1]}]"
        return step_id

    # ==================== Bash Python Extraction ====================

    @staticmethod
    def _extract_python_code_from_bash(command: str, mnt_dir: str = None) -> Optional[str]:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            tokens = []

        for i, token in enumerate(tokens):
            if os.path.basename(token) in ("python", "python3"):
                if i + 2 < len(tokens) and tokens[i + 1] == "-c":
                    return tokens[i + 2]
                for arg in tokens[i + 1:]:
                    if arg.endswith(".py"):
                        if mnt_dir:
                            script_path = arg
                            if script_path.startswith(DEFAULT_WORK_DIR + "/"):
                                script_path = os.path.join(mnt_dir, script_path[len(DEFAULT_WORK_DIR) + 1:])
                            elif not os.path.isabs(script_path):
                                script_path = os.path.join(mnt_dir, script_path)
                            if os.path.exists(script_path):
                                with open(script_path, "r", encoding="utf-8", errors="replace") as f:
                                    return f.read()
                        return f"# Python script executed from Bash: {arg}"

        heredoc_match = re.search(
            r'python3?\s+<<[\'\"]?(\w+)[\'\"]?\n([\s\S]*?)\n\1',
            command,
        )
        if heredoc_match:
            return heredoc_match.group(2)

        return None

    # ==================== Workspace Outputs / Quarantine ====================

    def _compute_outputs_from_diff(self, mnt_dir: str, file_snapshot: List[str]) -> List[str]:
        return self.workspace_outputs.compute_outputs_from_diff(mnt_dir, file_snapshot)

    def _quarantine_step_outputs(
        self,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
    ) -> Dict[str, Any]:
        return self.workspace_outputs.quarantine_step_outputs(
            task_id=task_id,
            step_id=step_id,
            file_snapshot=file_snapshot,
            task_envs=self.task_envs,
            task_quarantined_outputs=self.task_quarantined_outputs,
            task_loggers=self._task_loggers,
        )

    @staticmethod
    def _format_quarantine_prompt(quarantine_info: Dict[str, Any]) -> str:
        return WorkspaceOutputManager.format_quarantine_prompt(quarantine_info)

    def _annotate_step_quarantine(self, step_id: str, quarantine_info: Dict[str, Any]) -> None:
        self.workspace_outputs.annotate_step_quarantine(
            self.trajectory,
            step_id,
            quarantine_info,
        )

    def _mark_verified_outputs(
        self,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
    ) -> List[str]:
        return self.workspace_outputs.mark_verified_outputs(
            task_id=task_id,
            step_id=step_id,
            file_snapshot=file_snapshot,
            task_envs=self.task_envs,
            task_verified_outputs=self.task_verified_outputs,
            task_loggers=self._task_loggers,
        )

    def _cleanup_exploration_outputs(
        self,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
    ) -> Dict[str, Any]:
        return self.workspace_outputs.cleanup_exploration_outputs(
            task_id=task_id,
            step_id=step_id,
            file_snapshot=file_snapshot,
            task_envs=self.task_envs,
            task_loggers=self._task_loggers,
        )

    def _freeze_verified_outputs(
        self,
        task_id: str,
        rel_paths: Optional[List[str]] = None,
    ) -> None:
        self.workspace_outputs.freeze_verified_outputs(
            task_id=task_id,
            task_envs=self.task_envs,
            task_verified_outputs=self.task_verified_outputs,
            rel_paths=rel_paths,
        )

    # ==================== Docker Workspace / Write Protection ====================

    def _snapshot_mnt_files(self, mnt_dir: str) -> List[str]:
        return self.workspace_outputs.snapshot_mnt_files(mnt_dir)

    def _set_write_protection(self, mnt_dir: str, readonly: bool = True,
                               container=None, work_dir: str = '/workspace') -> None:
        self.workspace_outputs.set_write_protection(
            mnt_dir,
            readonly=readonly,
            container=container,
            work_dir=work_dir,
        )

    # ==================== Task Setup ====================

    def _setup_task_files(self, task_id: str, source_dir: Optional[str],
                          mnt_dir: str) -> None:
        """Copy source files to task's mnt_dir/source/ directory."""
        import shutil

        source_dest = os.path.join(mnt_dir, "source")
        os.makedirs(source_dest, exist_ok=True)

        if source_dir and os.path.isdir(source_dir):
            task_source = os.path.join(source_dir, task_id)
            if os.path.isdir(task_source):
                shutil.copytree(task_source, source_dest, dirs_exist_ok=True)
                logger.info(f"Copied source files from {task_source} to {source_dest}")
            else:
                logger.warning(f"Task source dir {task_source} not found")
        else:
            logger.info(f"No source_dir provided for task {task_id}")

    # ==================== Finalization ====================

    def _finalize_results(self) -> Dict[str, Dict[str, Any]]:
        """Finalize task results after execution loop."""
        for task_id in self.task_mnt_dirs:
            if task_id not in self.task_results or self.task_results[task_id] is None:
                if self.task_done.get(task_id):
                    self.task_results[task_id] = {
                        "done": False,
                        "result": "Task incomplete",
                    }
                else:
                    self.task_done[task_id] = True
                    self.task_results[task_id] = {
                        "done": False,
                        "result": "Global max steps reached",
                    }
            self._mark_task_finished(task_id)

        # Cleanup intermediate files from each task's mnt_dir
        for task_id, mnt_dir in self.task_mnt_dirs.items():
            task_config = self.task_configs.get(task_id, {})
            self._cleanup_intermediate_files(mnt_dir, task_id, task_config)

        self._write_cost_metrics()
        return self.task_results

    def _cleanup_intermediate_files(self, mnt_dir: str, task_id: str,
                                      task_config: Dict[str, Any]) -> None:
        """Remove intermediate files from mnt_dir after task completion.

        Keeps: source directory, files listed in task_config['file'] (expected
        outputs), and small files. Removes everything else.
        """
        keep_files = set()

        # Keep expected output files from task config
        expected_files = task_config.get("file", [])
        keep_files.update(expected_files)

        removed = 0
        removed_size = 0
        for root, dirs, files in os.walk(mnt_dir, topdown=True):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, mnt_dir)

                # Keep expected output files
                if rel in keep_files:
                    continue

                # Keep small files (< 1MB) — they're unlikely to be problematic
                try:
                    size = os.path.getsize(fpath)
                    if size < 1_000_000:
                        continue

                    os.remove(fpath)
                    removed += 1
                    removed_size += size
                except Exception as e:
                    logger.warning(f"Failed to remove {rel}: {e}")

        if removed > 0:
            logger.info(f"Cleanup for task {task_id}: removed {removed} "
                       f"intermediate files ({removed_size/1_000_000:.1f}MB)")

    # ==================== Backward Compatibility ====================

    def set_env_and_task(self, env) -> None:
        """Backward-compatible single-task initialization."""
        task_id = env.task_config.get("id", str(uuid.uuid4())[:8])
        self.run([{
            "task_id": task_id,
            "instruction": env.task_config["instruction"],
            "task_config": env.task_config,
            "mnt_dir": env.mnt_dir,
            "source_dir": env.source_dir,
            "env": env,
        }])

    def get_single_result(self) -> Tuple[bool, str]:
        """Get result for single-task mode (backward compat)."""
        if not self.task_results:
            return False, "No tasks executed"
        first_result = next(iter(self.task_results.values()))
        return first_result.get("done", False), first_result.get("result", "")

    def get_trajectory(self) -> Dict[str, Any]:
        """Return execution trajectory."""
        trajectory_entries = []
        for entry in self.trajectory:
            action = entry.get("action", "")
            step_id = entry.get("step_id") or entry.get("type", "entry")
            trajectory_entries.append({
                "observation": entry.get("observation", ""),
                "observation_digest": entry.get("observation_digest", ""),
                "raw_observation": entry.get("raw_observation"),
                "thought": entry.get("thought", ""),
                "action": action,
                "code": entry.get("code"),
                "response": f"Step {step_id}: {action}" if action else f"Entry {step_id}",
                "task_id": entry.get("task_id"),
                "cycle_id": entry.get("cycle_id"),
                "type": entry.get("type"),
            })

        return {
            "Task": "; ".join(self.task_instructions.values()),
            "trajectory": trajectory_entries,
        }
