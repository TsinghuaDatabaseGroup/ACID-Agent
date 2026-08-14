import os
import sys
from benchmark.benchmark_utils import print_error, print_warning
from typing import List
from .smolagents_lib.models import LiteLLMModel

sys.path.append("./")

from benchmark.benchmark_api import System

class Smolagents(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.model = model

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        self.dataset_directory = dataset_directory

    def serve_query(self, query: str, query_id: str, subset_files: List[str]) -> dict:
        return { "answer": "This is a placeholder answer."}

    def __getstate__(self):
        """
        Custom serialization to handle non-picklable LiteLLMModel objects.
        Excludes llm_code and llm_reason, which will be recreated on unpickling.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable LLM model objects
        state.pop('llm_code', None)
        state.pop('llm_reason', None)
        return state

    def __setstate__(self, state):
        """
        Custom deserialization to recreate the LiteLLMModel objects.
        """
        self.__dict__.update(state)
        # Recreate the LLM models using the stored model name
        custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
        if "claude" in self.model and "PDT" in self.name:
            # TODO double check
            model_id = "o3"  #"claude-3-7-sonnet-latest" 
        else:
            model_id = self.model

        model_params = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        self.llm_code = LiteLLMModel(**model_params)
        self.llm_reason = LiteLLMModel(**model_params)
