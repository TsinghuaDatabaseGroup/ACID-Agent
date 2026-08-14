import os
import sys
import re
sys.path.append("./")
from benchmark.benchmark_utils import print_error, print_warning

import json
import pandas as pd
import subprocess
from typeguard import typechecked
from typing import Dict, List, Tuple

from benchmark.benchmark_api import System
import argparse
import threading
import ast
import time
import random

from dotenv import load_dotenv
from .text_inspector_tool import TextInspectorTool
from .tools import list_input_filepaths, write_file, get_csv_metadata, summarize_dataframe
from .smolagents_prompts import PDT_TASK_PROMPT_TEMPLATE, DECOMPOSER_PROMPT_TEMPLATE, DECOMPOSER_DESCRIPTION, EXECUTOR_DESCRIPTION, EXECUTOR_PROMPT_TEMPLATE, EXECUTOR_PROMPT_TEMPLATE_LAST_STEP
from .smolagents_system import Smolagents
from .smolagents_utils import parse_token_counts

from .smolagents_lib import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
    AgentLogger
)
from .smolagents_lib.monitoring import LogLevel
load_dotenv()

class SmolagentsPDT(Smolagents):
    def __init__(self, model: str, name="example", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.name = name
        self.dataset_directory = None  # TODO(SUT engineer): Update me
        self.model = model # claude-3-7-sonnet-latest
        custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
        model_params = {
            "model_id": model, #"claude-3-7-sonnet-latest",
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        self.llm_code = LiteLLMModel(**model_params)
        if "claude" in model:
            reason_model_params = {
                #TODO double check
                "model_id": "o3", #"claude-3-7-sonnet-latest",
                "custom_role_conversions": custom_role_conversions,
                "max_completion_tokens": 8192,
            }
            self.llm_reason = LiteLLMModel(**reason_model_params) # Reasoning model can be set to a different one
        else:
            self.llm_reason = LiteLLMModel(**model_params)  # same as code model
        self.verbose = kwargs.get("verbose", False)
        self.verbosity_level = 2

        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))
        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_steps = kwargs.get("max_steps", 10)
        self.text_limit = kwargs.get("text_limit", 100000)
        self.planning_interval = kwargs.get("planning_interval", 4)

        self.question_output_dir = ""  # to be set in run()
        self.question_intermediate_dir = ""  # to be set in run()

    def _init_output_dir(self, query_id: str) -> None:
        """
        Initialize the output directory for the question.
        :param question: Question object
        """
        question_output_dir = os.path.join(self.output_dir, query_id)
        if not os.path.exists(question_output_dir):
            os.makedirs(question_output_dir)
        self.question_output_dir = question_output_dir
        self.question_intermediate_dir = os.path.join(
            self.question_output_dir, "_intermediate"
        )
        if not os.path.exists(self.question_intermediate_dir):
            os.makedirs(self.question_intermediate_dir)

    def _get_output(self, answer_path: str, pipeline_code_path: str) -> Tuple[str, str]:
        """
        Get the output from the answer and pipeline code files.
        :param answer_path: str
        :param pipeline_code_path: str
        :return: (answer, pipeline_code)
        """
        try:
            with open(answer_path, "r") as f:
                answer = f.read().strip()
        except Exception as e:
            print_error(f"Failed to read answer file: {e}")
            answer = "Failed to read answer."

        try:
            with open(pipeline_code_path, "r") as f:
                pipeline_code = f.read().strip()
        except Exception as e:
            print_error(f"Failed to read pipeline code file: {e}")
            pipeline_code = "Failed to read pipeline code."

        return answer, pipeline_code

    def _get_token_counts(self, query_id: str) -> Dict[str, int]:
        """
        Get the token counts from the log file.
        :param query_id: str
        :return: Dict[str, int]
        """
        logger_path = os.path.join(self.question_intermediate_dir, f"{query_id}.txt")
        with open(logger_path, "r") as f:
            trace = f.read()
        inp, out = parse_token_counts(trace)
        return {"input_tokens": inp, "output_tokens": out}

    def create_pdt_agents(self, task_id: str):

        logger_path = os.path.join(self.question_intermediate_dir, f"{task_id}.txt")
        logger = AgentLogger(level=LogLevel(self.verbosity_level), log_file=logger_path)

        # === Subtask Decomposer Agent ===
        decomposer_agent = ToolCallingAgent(
            model=self.llm_reason,
            tools=[list_input_filepaths, summarize_dataframe],
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            logger=logger,
            planning_interval=self.planning_interval,
            name="decomposer_agent",
            description= DECOMPOSER_DESCRIPTION,
            provide_run_summary=True,
        )

        # === Tool/Code Executor Agent ===
        executor_agent = CodeAgent(
            model=self.llm_code,
            planner_model=self.llm_reason,
            tools=[
                list_input_filepaths,
                TextInspectorTool(self.llm_reason, self.text_limit),
                write_file,
            ],
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=self.planning_interval,
            logger=logger,
            name="executor_agent",
            description= EXECUTOR_DESCRIPTION,
        )

        return decomposer_agent, executor_agent

    def run_pdt_pipeline(self,decomposer_agent, executor_agent, task_prompt, workload_name, task_id: str) -> Dict:
        """
        Orchestrates the Planner -> Decomposer -> Executor flow for a single question.
        Returns a dict with the plan, subtasks, subtask_outputs, and final_answer.
        """
        # ---- 1) Decomposer: JSON subtasks ----
        decomposer_prompt = DECOMPOSER_PROMPT_TEMPLATE.format(
            workload_name=workload_name,
            task_prompt=task_prompt,
        )
        raw_subtasks = decomposer_agent.run(decomposer_prompt)
        token_counts = self._get_token_counts(task_id)
        # Try to parse JSON (you can add more robust cleaning if needed)
        try:
            subtasks = json.loads(str(raw_subtasks))
        except json.JSONDecodeError:
            # Fallback: wrap in list if model forgot brackets etc.
            try:
                cleaned = raw_subtasks.strip()
                if not cleaned.startswith("["):
                    cleaned = "[" + cleaned
                if not cleaned.endswith("]"):
                    cleaned = cleaned + "]"
                subtasks = json.loads(cleaned)
            except Exception:
                # If this still fails, just fall back to a single-subtask wrapper
                subtasks = [
                    {
                        "id": 1,
                        "description": f"Execute the entire task directly: {task_prompt}",
                        "depends_on": [],
                    }
                ]

        # ---- 2) Executor: run each subtask sequentially ----
        answer_path = os.path.join(self.question_intermediate_dir, f"answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, f"pipeline_code.py")
        subtask_outputs = {}
        last_id = sorted(subtask_outputs.keys())[-1] if subtask_outputs else 1
        for subtask in subtasks:
            sid = subtask.get("id")
            desc = subtask.get("description", "")
            deps = subtask.get("depends_on", [])

            deps_summaries = [
                f"(Subtask {d}): {subtask_outputs.get(d, '')}" for d in deps if d in subtask_outputs
            ]
            deps_text = "\n".join(deps_summaries) if deps_summaries else "None yet."

            executor_prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                workload_name=workload_name,
                task_prompt=task_prompt,
                sid=sid,
                desc=desc,
                deps_text=deps_text,
            )

            if sid == last_id:
                executor_prompt += EXECUTOR_PROMPT_TEMPLATE_LAST_STEP.format(
                    answer_path=answer_path,
                    pipeline_code_path=pipeline_code_path,
                )
            subtask_output = executor_agent.run(executor_prompt)
            subtask_outputs[sid] = subtask_output
            agent_token_counts = self._get_token_counts(task_id)
            token_counts["input_tokens"] += agent_token_counts["input_tokens"]
            token_counts["output_tokens"] += agent_token_counts["output_tokens"]

        # Define final answer as the output of the last subtask (by id order)
        final_answer = subtask_outputs[last_id]

        return {
            "subtasks": subtasks,
            "subtask_outputs": subtask_outputs,
            "final_answer": final_answer,
            "token_usage": token_counts["input_tokens"] + token_counts["output_tokens"],
            "token_usage_input": token_counts["input_tokens"],
            "token_usage_output": token_counts["output_tokens"],
        }

    def robust_pdt_pipeline(
        self,
        decomposer_agent,
        executor_agent,
        task_prompt: str,
        workload_name: str,
        task_id: str,
        max_retry: int = 10,
    ) -> Dict:
        """
        Robust wrapper around run_pdt_pipeline that retries the whole
        Planner -> Decomposer -> Executor pipeline when Anthropic / LiteLLM
        throws an InternalServerError (5xx).

        On success: returns the same dict as run_pdt_pipeline.
        On total failure: returns a structured SYSTEM_ERROR dict with zero token usage.
        """
        base_backoff = 1.0  # seconds
        last_error = None

        for attempt in range(1, max_retry + 1):
            try:
                # Your original logic lives here
                return self.run_pdt_pipeline(
                    decomposer_agent=decomposer_agent,
                    executor_agent=executor_agent,
                    task_prompt=task_prompt,
                    workload_name=workload_name,
                    task_id=task_id,
                )

            except Exception as e:
                last_error = e
                # Use your existing logging helper if available
                try:
                    print_error(
                        f"[{self.name}] InternalServerError on "
                        f"attempt {attempt}/{max_retry}: {e}"
                    )
                except NameError:
                    print(
                        f"[{self.name}] InternalServerError on "
                        f"attempt {attempt}/{max_retry}: {e}"
                    )

                if attempt == max_retry:
                    break

                # Exponential backoff with jitter: 1, 2, 4, 8, ... (capped at 30s)
                sleep_secs = base_backoff * (2 ** (attempt - 1))
                sleep_secs = min(sleep_secs, 30.0)
                sleep_secs += random.uniform(0, 0.5)
                time.sleep(sleep_secs)

        # If we get here, all retries for InternalServerError failed
        err_msg = (
            "SYSTEM_ERROR: Anthropic internal server error after "
            f"{max_retry} retries in PDT pipeline."
        )
        try:
            print_error(
                f"[{self.name}] {err_msg} Last error for task_id={task_id}: {last_error}"
            )
        except NameError:
            print(f"[{self.name}] {err_msg} Last error for task_id={task_id}: {last_error}")

        return {
            "subtasks": [],
            "subtask_outputs": {},
            "final_answer": err_msg,
            "error": str(last_error) if last_error else "Unknown Anthropic internal error.",
            "token_usage": 0,
            "token_usage_input": 0,
            "token_usage_output": 0,
        }

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        # read files based on the dataset_directory, could be pdf, csv, etc.
        # store the data in a dictionary
        # store the dictionary in self.dataset_directory
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".csv"):
                try:
                    self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))
                except Exception as e:
                    print_warning(f"Failed to read {file}: {e}")

    @typechecked
    def serve_query(self, query: str, query_id: str = "default_name-0", subset_files: List[str] = []) -> Dict:
        """
        Serve a query using the LLM with a hard 15-minute timeout.

        If the PDT pipeline exceeds 15 minutes (900 seconds),
        the process is terminated and a timeout response is returned.
        """
        TIME_LIMIT = 1200  # 20 minutes

        # Prepare output directory
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        dataset_directory = os.path.join(os.getcwd(), "data", dataset_name, "input")

        # -------------------------------------------------------
        # Worker function that runs the PDT pipeline
        # -------------------------------------------------------
        result_container: Dict = {"result": None, "error": None}

        def worker():
            try:
                decomposer_agent, executor_agent = self.create_pdt_agents(query_id)

                task_prompt = PDT_TASK_PROMPT_TEMPLATE.format(
                    dataset_name=dataset_name,
                    dataset_directory=dataset_directory,
                    query=query,
                    subset_files=subset_files,
                )

                pdt_result = self.robust_pdt_pipeline(
                    decomposer_agent,
                    executor_agent,
                    task_prompt,
                    dataset_name,
                    query_id,
                    max_retry=5,
                )

                result_container["result"] = pdt_result

            except Exception as e:
                result_container["error"] = str(e)

        # -------------------------------------------------------
        # Run the worker thread with a hard timeout
        # -------------------------------------------------------
        start_time = time.time()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=TIME_LIMIT)
        runtime = time.time() - start_time

        # -------------------------------------------------------
        # Check for timeout
        # -------------------------------------------------------
        if thread.is_alive():
            # Thread timed out
            return {
                "id": query_id,
                "runtime": runtime,
                "timeout": True,
                "message": f"Query exceeded the 20-minute time limit (1200 seconds).",
                "subtasks": None,
                "subtask_outputs": None,
                "explanation": {"id": "main-task", "answer": None},
                "token_usage": 0,
                "token_usage_input": 0,
                "token_usage_output": 0,
                "pipeline_code": "N/A",
            }

        # -------------------------------------------------------
        # Pipeline finished normally
        # -------------------------------------------------------
        if result_container["error"] is not None or result_container["result"] is None:
            return {
                "id": query_id,
                "runtime": runtime,
                "timeout": False,
                "error": result_container["error"],
                "subtasks": None,
                "subtask_outputs": None,
                "explanation": {"id": "main-task", "answer": None},
                "token_usage": 0,
                "token_usage_input": 0,
                "token_usage_output": 0,
                "pipeline_code": "N/A",
            }

        pdt_result = result_container["result"]
        answer_path = os.path.join(self.question_intermediate_dir, f"answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, f"pipeline_code.py")
        answer, pipeline_code = self._get_output(answer_path, pipeline_code_path)

        results = {
            "id": query_id,
            "runtime": runtime,
            "timeout": False,
            "subtasks": pdt_result["subtasks"],
            "subtask_outputs": pdt_result["subtask_outputs"],
            "explanation": {"id": "main-task", "answer": pdt_result["final_answer"]},
            "token_usage": pdt_result["token_usage"],
            "token_usage_input": pdt_result["token_usage_input"],
            "token_usage_output": pdt_result["token_usage_output"],
            "pipeline_code": pipeline_code,
        }

        # print(results)
        return results


def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions = [
        {
            "id": "environment-easy-1",
            "query": "In 2018, how many bacterial exceedances were observed in freshwater beaches?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-medium-1",
            "query": "For the freshwater beaches, what is the difference between the percentage exceedance rate in 2023 and the historic average from 2002 to 2022?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-hard-1",
            "query": "For the marine beaches, what is the Pearson product-moment correlation from 2002 to 2023 between the rainfall amount in inches during the months June, July, August, and September and the percentage exceedance rate?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
    ]

    # Process each question
    out_dir = os.path.join(current_dir, "../../testresults/pdt")
    smolagents_dr = SmolagentsPDT(
        model="claude-3-7-sonnet-latest", output_dir=out_dir, verbose=True
    )
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        smolagents_dr.serve_query(question["query"], question["id"])
        break

if __name__ == "__main__":
    main()
