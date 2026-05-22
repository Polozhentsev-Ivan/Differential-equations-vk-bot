"""Classifier Agent — determines applicable SymPy methods for an ODE."""

from __future__ import annotations

from crewai import Agent, LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_LIGHT
from tools.crewai_tools import classify_ode_json_tool


def get_llm() -> LLM:
    return LLM(
        model=f"openai/{LLM_MODEL_LIGHT}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_tokens=2048,
    )


def create_classifier_agent() -> Agent:
    return Agent(
        role="ODE Classification Specialist",
        goal="Classify differential equations using deterministic SymPy tools.",
        backstory=(
            "You are responsible for identifying applicable solution methods for ODEs. "
            "Always call classify_ode_json and return only the JSON object produced by the tool."
        ),
        tools=[classify_ode_json_tool],
        llm=get_llm(),
        verbose=True,
    )
