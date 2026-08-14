"""LLM API utilities for ACID-Agent.

This module wraps OpenAI-compatible chat/completion calls, model-specific
request options, retry handling, and usage extraction. All higher-level agent
components should route LLM calls through this layer so token and cost metrics
can be collected consistently.
"""

import time
import json
import logging
import os
import requests
import numpy as np
from openai import OpenAI
import re

logger = logging.getLogger("da_agent.llm")

# API configuration - using Dashscope (Alibaba Cloud)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
OPENAI_API_BASE = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or os.getenv("DASHSCOPE_API_BASE")
    or (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if os.getenv("DASHSCOPE_API_KEY")
        else "https://api.openai.com/v1"
    )
)


class OpenAIClient:
    """
    OpenAI-compatible client using Dashscope (Alibaba Cloud) API.

    Supports models like deepseek-v3, deepseek-r1, qwen, etc.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        resolved_api_key = api_key or OPENAI_API_KEY
        if not resolved_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY or DASHSCOPE_API_KEY for LLM calls.")
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url or OPENAI_API_BASE,
        )

    def generate(self, messages, model: str, enable_thinking: bool = False, **kwargs):
        """
        Generate response from LLM with streaming.

        Args:
            messages: List of message dictionaries
            model: Model name (e.g., 'deepseek-v3', 'qwen-max')
            enable_thinking: Enable thinking mode for deepseek models
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (reasoning_content, answer_content, usage)
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"enable_thinking": enable_thinking},
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )

            reasoning_content = ""
            answer_content = ""
            usage = None

            for chunk in completion:
                if not chunk.choices:
                    usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    answer_content += delta.content

            return reasoning_content, answer_content, usage

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "", str(e), None


def _should_enable_thinking(model: str) -> bool:
    """Auto-detect whether thinking mode should be enabled for a model."""
    return False
    # return model in ['qwen3.5-397b-a17b']


def _build_llm_kwargs(payload: dict) -> dict:
    """Extract common kwargs from payload for client.generate()."""
    model = payload["model"]
    enable_thinking = payload.get("enable_thinking", _should_enable_thinking(model))

    kwargs = {}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    if "max_tokens" in payload:
        kwargs["max_tokens"] = payload["max_tokens"]
    if "temperature" in payload:
        kwargs["temperature"] = payload["temperature"]
    if "top_p" in payload:
        kwargs["top_p"] = payload["top_p"]

    return kwargs


def call_llm_dashscope(payload: dict) -> tuple:
    """
    Call LLM using Dashscope API.

    Args:
        payload: Dictionary with model, messages, and other parameters

    Returns:
        Tuple of (success, response)
    """
    client = OpenAIClient()
    model = payload.get("model", "deepseek-v3")
    messages = payload.get("messages", [])

    kwargs = _build_llm_kwargs(payload)
    reasoning, answer, usage = client.generate(messages, model, **kwargs)

    if answer and not answer.startswith("Error"):
        return True, answer
    return False, answer or reasoning


def call_llm_with_usage(payload: dict) -> tuple:
    """Call LLM and return (success, response, usage, reasoning).

    reasoning_content is returned separately so callers can extract
    confidence etc. without polluting the main response (which goes
    into LLM memory).
    """
    client = OpenAIClient()
    model = payload.get("model", "deepseek-v3")
    messages = payload.get("messages", [])

    kwargs = _build_llm_kwargs(payload)
    reasoning, answer, usage = client.generate(messages, model, **kwargs)

    if answer and not answer.startswith("Error"):
        return True, answer, usage, reasoning
    return False, answer or reasoning, usage, reasoning


# Alias for compatibility
call_llm = call_llm_dashscope
