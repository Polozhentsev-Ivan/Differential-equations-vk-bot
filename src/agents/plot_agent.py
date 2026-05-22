"""Plot Agent — prepares visual artifacts for qualitative ODE tasks."""

from __future__ import annotations

from crewai import Agent, LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_LIGHT
from tools.crewai_tools import compute_isoclines_json_tool, plot_isoclines_json_tool


def get_llm() -> LLM:
    return LLM(
        model=f"openai/{LLM_MODEL_LIGHT}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_tokens=2048,
    )


def create_plot_agent() -> Agent:
    return Agent(
        role="ODE Visualization Specialist",
        goal="Generate isocline and direction-field artifacts with deterministic plotting tools.",
        backstory=(
            "You prepare visual outputs for ODE tasks. "
            "For isocline tasks, call plot_isoclines_json with the requested output path and return only JSON."
        ),
        tools=[compute_isoclines_json_tool, plot_isoclines_json_tool],
        llm=get_llm(),
        verbose=True,
    )
