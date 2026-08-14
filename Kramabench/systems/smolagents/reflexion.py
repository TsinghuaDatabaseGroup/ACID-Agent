import os
import sys
import re

from .smolagents_lib.monitoring import LogLevel
sys.path.append("./")
from benchmark.benchmark_utils import print_error, print_warning

import json
import pandas as pd
import ast
import subprocess
import random
from typeguard import typechecked
from typing import Dict, List, Tuple

from benchmark.benchmark_api import System
import argparse
import threading
import time

from dotenv import load_dotenv
from .text_inspector_tool import TextInspectorTool
from .answer_inspector_tool import AnswerInspectorTool
from .tools import write_file, list_input_filepaths, get_csv_metadata, summarize_dataframe
from .smolagents_utils import parse_token_counts
from .smolagents_prompts import CRITIQUE_AGENT_PROMPT_TEMPLATE, CRITIQUE_AGENT_SYSTEM_PROMPT, REFLEXION_TASK_PROMPT_TEMPLATE
from .smolagents_system import Smolagents

from .smolagents_lib import (
    CodeAgent,
    LiteLLMModel,
    AgentLogger,
    ToolCallingAgent,
)
load_dotenv()

class SmolagentsReflexion(Smolagents):
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
        self.llm_reason = LiteLLMModel(**model_params) # Reasoning model can be set to a different one
        self.verbose = kwargs.get("verbose", False)      
        self.verbosity_level = 2

        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))
        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_steps = kwargs.get("max_steps", 20)
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
    
    def _parse_output(self, output: str, query_id: str) -> Tuple[str, str]:
        """
        Parse the output from the agent to extract the answer and pipeline code.
        Write intermediate logs to file.
        :param output: str
        :param query_id: str
        :return: (answer, pipeline_code)
        """
        # Parse the output to extract the final answer and pipeline code
        try:
            # Case 1: agent already returned a dict-like object
            if isinstance(output, dict):
                output_dict = output
            else:
                # Coerce to string in case it's something like a custom object
                output_str = str(output)

                # Try to find the first {...} block (non-greedy) in the text
                match = re.search(r"\{.*?\}", output_str, re.DOTALL)
                if not match:
                    raise ValueError("No JSON-like object found in agent output.")

                obj_str = match.group(0)

                # First try strict JSON
                try:
                    output_dict = json.loads(obj_str)
                except json.JSONDecodeError:
                    # Fallback: many LLMs emit Python dict reprs with single quotes
                    try:
                        output_dict = ast.literal_eval(obj_str)
                        if not isinstance(output_dict, dict):
                            raise ValueError("Parsed object is not a dict.")
                    except Exception as inner_e:
                        raise ValueError(
                            f"Failed to parse as JSON or Python dict: {inner_e}"
                        ) from inner_e

            # Extract fields with sensible fallbacks / alternative keys
            answer = (
                output_dict.get("explanation")
                or output_dict.get("answer")
                or "No explanation found."
            )
            pipeline_code = (
                output_dict.get("pipeline_code")
                or output_dict.get("code")
                or "No pipeline code found."
            )

        except Exception as e:
            print_error(f"Failed to parse output JSON: {e}\nRaw output:\n{output}")
            answer = "Failed to parse output."
            pipeline_code = "N/A"
        
        # Write results to file
        with open(os.path.join(self.question_intermediate_dir, f"answer.txt"), "w") as f:
            f.write(answer)
        with open(os.path.join(self.question_intermediate_dir, f"pipeline_code.py"), "w") as f:
            f.write(pipeline_code)

        return answer, pipeline_code

    def create_agent(self, task_id:str) -> CodeAgent:

        logger_path = os.path.join(self.question_intermediate_dir, f"{task_id}.txt")
        logger = AgentLogger(level=LogLevel(self.verbosity_level), log_file=logger_path)
        #############################################
        
        critique_agent = ToolCallingAgent(
            model=self.llm_reason,
            tools=[TextInspectorTool(self.llm_reason, self.text_limit), AnswerInspectorTool(self.llm_reason), list_input_filepaths, get_csv_metadata, summarize_dataframe],
            max_steps=10,
            verbosity_level=self.verbosity_level,
            logger=logger,
            planning_interval=self.planning_interval,
            name="critique_agent",
            description=CRITIQUE_AGENT_SYSTEM_PROMPT,
            provide_run_summary=True,
        )
        critique_agent.prompt_templates["managed_agent"]["task"] += CRITIQUE_AGENT_PROMPT_TEMPLATE

        manager_agent = CodeAgent(
            model=self.llm_code,
            planner_model=self.llm_reason,
            tools=[TextInspectorTool(self.llm_reason, self.text_limit), list_input_filepaths, write_file], #process_beach_data
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=self.planning_interval,
            logger=logger,
            managed_agents=[critique_agent],
        )

        return manager_agent

    def robust_run(self, agent, task: str, max_retry: int = 10):
        """
        Run agent.run(task) with retries on litellm.InternalServerError
        (e.g., Anthropic 5xx). Returns the agent output or None if all
        retries fail.

        Other exceptions are treated as non-retryable and re-raised.
        """
        base_backoff = 1.0  # seconds
        last_error = None

        for attempt in range(1, max_retry + 1):
            try:
                return agent.run(task)

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

                # Exponential backoff with jitter
                sleep_secs = base_backoff * (2 ** (attempt - 1))  # 1, 2, 4, 8, ...
                sleep_secs = min(sleep_secs, 30.0)                # cap at 30s
                sleep_secs += random.uniform(0, 0.5)
                time.sleep(sleep_secs)

        # All retries exhausted
        try:
            print_error(f"[{self.name}] Failed after {max_retry} retries: {last_error}")
        except NameError:
            print(f"[{self.name}] Failed after {max_retry} retries: {last_error}")

        return None
    
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
    def serve_query(self, query: str, query_id: str = "default_name-0", subset_files:List[str]=[]) -> Dict:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        :param query: str
        :param query_id: str
        :param subset_files: list of strings with filenames to include in the experiments
        :return: output_dict {"explanation": answer, "pipeline_code": pipeline_code}
        """
        # TODO: 
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        answer_path = os.path.join(self.question_intermediate_dir, f"answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, f"pipeline_code.py")
        task = REFLEXION_TASK_PROMPT_TEMPLATE.format(
            dataset_name=dataset_name,
            dataset_directory=os.path.join(os.getcwd(), 'data', dataset_name, 'input'),
            query=query,
            subset_files=subset_files,
            answer_path=answer_path,
            pipeline_code_path=pipeline_code_path,
        )
        start_time = time.time()
        agent = self.create_agent(task_id=query_id)
        _ = self.robust_run(agent, task, max_retry=5)
        runtime = time.time() - start_time

        answer, pipeline_code = self._get_output(answer_path, pipeline_code_path)
        token_counts = self._get_token_counts(query_id)
        results = {
            "id": query_id,
            "runtime": runtime,
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": pipeline_code,
            "token_usage": token_counts["input_tokens"] + token_counts["output_tokens"],
            "token_usage_input": token_counts["input_tokens"],
            "token_usage_output": token_counts["output_tokens"],
        }
        #print(results)
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
    out_dir = os.path.join(current_dir, "../../testresults/reflexion")
    smolagents_dr = SmolagentsReflexion(
        model="claude-3-7-sonnet-latest", output_dir=out_dir, verbose=True
    )
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        smolagents_dr.serve_query(question["query"], question["id"])
        break

if __name__ == "__main__":
    main()