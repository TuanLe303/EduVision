"""Public API for the report_generator src package."""

from .aggregator import ClassStats, SessionSummary, StudentSummary, aggregate_jsonl
from .llm_client import GPTClient, GeminiClient, LLMClient, build_llm_client
from .prompt_builder import build_prompt

__all__ = [
    "SessionSummary",
    "StudentSummary",
    "ClassStats",
    "aggregate_jsonl",
    "LLMClient",
    "GeminiClient",
    "GPTClient",
    "build_llm_client",
    "build_prompt",
]
