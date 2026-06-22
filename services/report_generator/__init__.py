"""Report Generator service — aggregate vision-AI session data and generate LLM reports."""

from .src import (
    ClassStats,
    GPTClient,
    GeminiClient,
    LLMClient,
    SessionSummary,
    StudentSummary,
    aggregate_jsonl,
    build_llm_client,
    build_prompt,
)

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
