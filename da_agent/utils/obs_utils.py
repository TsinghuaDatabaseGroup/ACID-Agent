"""Observation processing utilities for ACID-Agent.

This module provides shared helpers for normalizing, truncating, and preparing
environment observations before they are used in prompts, summaries, reviews,
or trace records.
"""

# Observation classification utilities

OBS_CLASS_ERROR = "error"
OBS_CLASS_PARSE_FAIL = "parse_failure"
OBS_CLASS_WARNING = "warning"
OBS_CLASS_SUCCESS_NO_OUTPUT = "success_no_output"
OBS_CLASS_STANDARD = "standard_output"

# Confidence caps per observation class (used by acid_agent)
OBS_CONFIDENCE_CAPS = {
    OBS_CLASS_ERROR: 0.1,
    OBS_CLASS_PARSE_FAIL: 0.1,
    OBS_CLASS_WARNING: 0.5,
    OBS_CLASS_SUCCESS_NO_OUTPUT: 1.0,
    OBS_CLASS_STANDARD: 1.0,
}


def classify_observation(observation: str) -> str:
    """Classify observation into a category.

    Returns one of: error, parse_failure, warning, success_no_output, standard_output
    """
    if observation.startswith("Failed to parse action from your response,"):
        return OBS_CLASS_PARSE_FAIL
    if observation.startswith("ERROR:") or "Traceback (" in observation \
            or observation.startswith("bash: -c: line ") or "Error: " in observation:
        return OBS_CLASS_ERROR
    if "Warning:" in observation:
        return OBS_CLASS_WARNING
    if observation.strip() == "executed successfully. No output." \
            or "executed successfully. No output" in observation:
        return OBS_CLASS_SUCCESS_NO_OUTPUT
    return OBS_CLASS_STANDARD


def get_confidence_cap(observation: str) -> float:
    """Get confidence upper bound based on observation classification."""
    return OBS_CONFIDENCE_CAPS[classify_observation(observation)]