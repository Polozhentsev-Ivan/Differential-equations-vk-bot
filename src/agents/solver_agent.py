"""Solver Agent — решает ОДУ с помощью SymPy tools."""

from __future__ import annotations

from crewai import Agent, LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_LIGHT
from tools.crewai_tools import (
    classify_ode_tool,
    classify_ode_json_tool,
    compute_isoclines_tool,
    compute_isoclines_json_tool,
    plot_isoclines_tool,
    plot_isoclines_json_tool,
    solve_ode_tool,
    solve_ode_json_tool,
    validate_solution_tool,
    validate_solution_json_tool,
)


def get_llm() -> LLM:
    return LLM(
        model=f"openai/{LLM_MODEL_LIGHT}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_tokens=4096,
    )


def create_solver_agent() -> Agent:
    return Agent(
        role="Математический решатель ОДУ",
        goal=(
            "Точно решить обыкновенное дифференциальное уравнение, "
            "провалидировать решение и вернуть результат."
        ),
        backstory=(
            "Ты — эксперт-математик, специализирующийся на дифференциальных уравнениях. "
            "Ты используешь инструменты SymPy для классификации, решения и валидации ОДУ. "
            "Всегда сначала классифицируй уравнение, потом реши его, потом проверь решение."
        ),
        tools=[
            classify_ode_tool,
            classify_ode_json_tool,
            solve_ode_tool,
            solve_ode_json_tool,
            validate_solution_tool,
            validate_solution_json_tool,
            compute_isoclines_tool,
            compute_isoclines_json_tool,
            plot_isoclines_tool,
            plot_isoclines_json_tool,
        ],
        llm=get_llm(),
        verbose=True,
    )
