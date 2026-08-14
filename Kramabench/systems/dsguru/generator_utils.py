"""
This file contains the Generator classes and generator factory.
"""
from __future__ import annotations
import logging

import os
import time
from benchmark.benchmark_utils import print_error, print_info
import ollama
import anthropic
from openai import OpenAI

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import PyPDF2

OpenAIModelList = ["gpt-4o", "gpt-4o-mini", "gpt-4o-v", "gpt-4o-mini-v", "o3-2025-04-16"]
TogetherModelList = ["google/gemma-2b-it", "meta-llama/Llama-2-13b-chat-hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B","google/gemma-3-27b-it", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-Coder-32B-Instruct", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]
ClaudeModelList = ["claude-sonnet-4-5-20250929", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]


import re

class Generator:

    def __init__(self, model: str, verbose: bool = False):
        self.model = model
        if model in OpenAIModelList:
            self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        elif model in TogetherModelList:
            self.client = Together(api_key=get_api_key("TOGETHER_API_KEY"))
        elif model in ClaudeModelList:
            self.client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported model: {model}")
        self.verbose = verbose

    def __getstate__(self):
        """Exclude non-picklable client from serialization."""
        state = self.__dict__.copy()
        # Remove the unpicklable client object
        state.pop('client', None)
        return state

    def __setstate__(self, state):
        """Restore state and re-initialize the client after deserialization."""
        self.__dict__.update(state)
        # Re-initialize the client
        if self.model in OpenAIModelList:
            self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        elif self.model in TogetherModelList:
            self.client = Together(api_key=get_api_key("TOGETHER_API_KEY"))
        elif self.model in ClaudeModelList:
            self.client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    # GV: This function is the call_gpt function from the baseline_utils.py file. But for Ollama we substitute it
    def __call__(self, messages):
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        max_retries = 5
        retry_count = 0
        fatal = False
        fatal_reason = None
        while retry_count < max_retries:
            try:
                if self.model in ClaudeModelList:
                    # skip system messages
                    claude_messages = [m for m in messages if m['role'] in ['user', 'assistant']]
                    result = self.client.messages.create(
                        model=self.model,
                        messages=claude_messages,
                        max_tokens=4000,
                    )
                    input_tokens = result.usage.input_tokens
                    output_tokens = result.usage.output_tokens
                    self.total_tokens += input_tokens + output_tokens
                    self.input_tokens += input_tokens
                    self.output_tokens += output_tokens
                else:
                    result = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                    self.total_tokens += result.usage.total_tokens
                    self.input_tokens += result.usage.prompt_tokens
                    self.output_tokens += result.usage.completion_tokens
                break # break out of while loop if no error
            
            except Exception as e:
                print_error(f"An error occurred: {e}.")
                if "context_length_exceeded" in f"{e}" or "too long" in f"{e}":
                    fatal = True
                    fatal_reason = f"{e}"
                    break
                else:
                    print_info("Retrying...")
                    retry_count += 1
                    time.sleep(10 * retry_count)  # Wait 
        
        if fatal:
            print(f"Fatal error occured. ERROR: {fatal_reason}")
            res = f"Fatal error occured. ERROR: {fatal_reason}"
        elif retry_count == max_retries:
            print("Max retries reached. Skipping...")
            res = "Max retries reached. Skipping..."
        else:
            try:
                if self.model in ClaudeModelList:
                    res = result.content[0].text
                else: 
                    res = result.choices[0].message.content
            except Exception as e:
                print("Error:", e)
                res = ""
            #print(res)
        return res

def pdf_to_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

class OllamaGenerator(Generator):
    def __init__(self, model: str, 
                verbose=False,
                server_url="http://localhost:11434",
                *args, 
                **kwargs
                ):
        super().__init__(model, verbose)
        self.model = model
        self.server_url = server_url
        self.client = ollama.Client(host=server_url)

    def __getstate__(self):
        """Exclude non-picklable client from serialization."""
        state = self.__dict__.copy()
        # Remove the unpicklable client object
        state.pop('client', None)
        return state

    def __setstate__(self, state):
        """Restore state and re-initialize the client after deserialization."""
        self.__dict__.update(state)
        # Re-initialize the client with the stored server_url
        self.client = ollama.Client(host=self.server_url)

    def __call__(self, messages):

        max_retries = 5
        retry_count = 0
        fatal = False
        fatal_reason = None
        self.total_tokens = 0
        while retry_count < max_retries:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format="json",
                    stream=False,
                    options= {"num_ctx": 640000}
                )
                self.total_tokens += response["usage"]["total_tokens"]
                if not response["done"]:
                    logging.warning("WARNING - Conversation kept going! Maybe output is truncated.")
                break

            except Exception as e:
                print(f"An error occurred: {e}.")
                if "context_length_exceeded" in f"{e}":
                    fatal = True
                    fatal_reason = f"{e}"
                    break
                else:
                    print("Retrying...")
                    retry_count += 1
                    time.sleep(10 * retry_count)  # Wait 
        
        if fatal:
            print(f"Fatal error occured. ERROR: {fatal_reason}")
            res = f"Fatal error occured. ERROR: {fatal_reason}"
        elif retry_count == max_retries:
            print("Max retries reached. Skipping...")
            res = "Max retries reached. Skipping..."
        else:
            try:
                res = response["message"]["content"]
            except Exception as e:
                print("Error:", e)
                res = ""
            #print(res)
        return res
