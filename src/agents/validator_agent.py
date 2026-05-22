"""Validator Agent — checks a proposed ODE solution."""

from __future__ import annotations

from crewai import Agent, LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_LIGHT
from tools.crewai_tools import validate_solution_json_tool


def get_llm() -> LLM:
    return LLM(
        model=f"openai/{LLM_MODEL_LIGHT}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_tokens=2048,
    )


def create_validator_agent() -> Agent:
    return Agent(
        role="ODE Solution Validator",
        goal="Validate proposed ODE solutions by substituting them back into the equation.",
        backstory=(
            "You are a strict mathematical reviewer. "
            "Always call validate_solution_json and return only the JSON object produced by the tool."
        ),
        tools=[validate_solution_json_tool],
        llm=get_llm(),
        verbose=True,
    )
