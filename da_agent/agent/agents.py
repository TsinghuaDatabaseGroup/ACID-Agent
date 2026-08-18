"""PromptAgent Module - DA-Agent-style baseline agent wrapper.

This module implements a prompt-driven data analysis agent:
- Builds task prompts and action-space instructions
- Calls an LLM to produce one action per step
- Executes actions in the Docker-backed data-analysis environment
- Records trajectory, final answer, and cost metrics for benchmark runs

References:
- https://github.com/yiyihum/da-code/tree/main/da_agent/agent/agents.py
"""

import base64
import json
import logging
import os
import re
import time
import uuid
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List
from da_agent.agent.prompts import SYS_PROMPT_IN_OUR_CODE
from da_agent.agent.action import Bash, Action, Terminate, Python, SQL
from da_agent.envs.da_agent import DA_Agent_Env
from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict

from da_agent.utils.cost_metrics import TaskCostTracker, _jsonable
from da_agent.utils.llm_utils import call_llm_with_usage



logger = logging.getLogger("da_agent")


class PromptAgent:
    def __init__(
        self,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
    ):
        
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps
        
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.system_message = ""
        self.history_messages = []
        self.env = None
        self.codes = []
        self._AVAILABLE_ACTION_CLASSES = [Bash, Python, SQL, Terminate]
        # self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate]
        self.work_dir = "/workspace"
        self.task_id = ""
        self.task_start_time = None
        self.task_end_time = None
        self.cost_tracker = TaskCostTracker()
        self.cost_metrics = {}
        self.cost_metric_events = {}
        self.trace_events = []
        self._current_step_idx = 0
        
    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self.trace_events = []
        self.task_id = str(self.env.task_config.get("id") or "")
        self.task_start_time = None
        self.task_end_time = None
        self.instruction = self.env.task_config['instruction']
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        self.system_message = SYS_PROMPT_IN_OUR_CODE.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
        if self.task_id:
            self.cost_tracker.start_task(self.task_id)
        
    def predict(self, obs: Dict=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """    
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        status = False
        while not status:
            messages = self.history_messages.copy()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })  
            llm_start = time.time()
            status, response, usage, reasoning = call_llm_with_usage({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
            llm_end = time.time()
            self.cost_tracker.record_llm_call(
                self.task_id,
                usage,
                count_as_turn=True,
                phase="da_agent_turn",
                model=self.model,
                step_id=f"step_{self._current_step_idx}",
                step_index=self._current_step_idx,
                start_time=llm_start,
                end_time=llm_end,
            )
            response = response.strip()
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            

        try:
            action = self.parse_action(response)
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            if thought:
                thought = thought.group(1).strip()
            else:
                thought = response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
        
        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)

        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)
        if action is not None:
            self.codes.append(action.code)
        else:
            self.codes.append(None)

        action_type = action.__class__.__name__ if action is not None else "ParseFailure"
        self.trace_events.append({
            "type": "da_agent_turn",
            "task_id": self.task_id,
            "step_id": f"step_{self._current_step_idx}",
            "step_index": self._current_step_idx,
            "timestamp": time.time(),
            "observation": obs,
            "thought": thought,
            "action": str(action),
            "action_type": action_type,
            "code": action.code if action is not None else None,
            "response": response,
            "reasoning": reasoning,
            "llm_usage": _jsonable(usage),
            "timing": {
                "start_time": llm_start,
                "end_time": llm_end,
                "duration": round(max(0.0, llm_end - llm_start), 3),
            },
        })

        return response, action
    
    
    
    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output: str) -> Action:
        """ Parse action from text """
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
        
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    
    
    def run(self):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        self.task_start_time = time.time()
        while not done and step_idx < self.max_steps:

            self._current_step_idx = step_idx
            self.cost_tracker.record_step(self.task_id)
            _, action = self.predict(
                obs
            )
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                if last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, please provide a different action."
                        repeat_action = True
                else:
                    self.cost_tracker.record_code_execution(
                        self.task_id,
                        action.__class__.__name__,
                    )
                    exec_start = time.time()
                    obs, done = self.env.step(action)
                    exec_end = time.time()
                    if self.trace_events:
                        self.trace_events[-1]["execution_timing"] = {
                            "start_time": exec_start,
                            "end_time": exec_end,
                            "duration": round(max(0.0, exec_end - exec_start), 3),
                        }
                        self.trace_events[-1]["post_action_observation"] = obs
                        self.trace_events[-1]["done"] = done
                    last_action = action
                    repeat_action = False

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        self.task_end_time = time.time()
        self.cost_tracker.finish_task(self.task_id)
        self.cost_metrics[self.task_id] = self.cost_tracker.get(self.task_id)
        self.cost_metric_events[self.task_id] = self.cost_tracker.get_events(self.task_id)
        return done, result

    def get_trajectory(self):
        trajectory = [_jsonable(event) for event in self.trace_events]
        trajectory_log = {
            "Task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }
        return trajectory_log

    def get_cost_metrics(self, task_id: str = None) -> Dict[str, Any]:
        target_task_id = task_id or self.task_id
        return self.cost_tracker.get(target_task_id)

    def get_cost_metric_events(self, task_id: str = None) -> List[Dict[str, Any]]:
        target_task_id = task_id or self.task_id
        return self.cost_tracker.get_events(target_task_id)

    def get_trace_events(self) -> List[Dict[str, Any]]:
        return [_jsonable(event) for event in self.trace_events]

    def get_trace_summary(self, finished: bool, result: str, result_files: dict = None) -> Dict[str, Any]:
        cost_metrics = self.get_cost_metrics(self.task_id)
        return {
            "task_id": self.task_id,
            "task": self.instruction,
            "model": self.model,
            "finished": finished,
            "steps": len(self.trace_events),
            "result": result,
            "result_files": result_files or {},
            "total_token_usage": {
                "input_tokens": cost_metrics.get("input_tokens", 0),
                "output_tokens": cost_metrics.get("output_tokens", 0),
                "reasoning_tokens": cost_metrics.get("reasoning_tokens", 0),
                "cached_input_tokens": cost_metrics.get("cached_input_tokens", 0),
                "total_tokens": cost_metrics.get("total_tokens", 0),
            },
            "cost_metrics": cost_metrics,
            "actions": [_jsonable(event) for event in self.trace_events],
            "trajectory": self.get_trajectory(),
        }


if __name__ == "__main__":
    agent = PromptAgent()
    response = """Bash(code=\"\"ls -a\"):\n\n(Note: I am using the 'ls -a' command to list all files, including hidden ones, in the working directory. This will help me ensure that I am in the correct directory and provide a reference for the file paths.\")"""
    import pdb; pdb.set_trace()
    action = agent.parse_action(response)
    print(action)
