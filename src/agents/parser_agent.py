"""Parser Agent — извлекает уравнение, тип задачи и метод из .md текста."""

from __future__ import annotations

from crewai import Agent, LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_LIGHT


def get_llm() -> LLM:
    return LLM(
        model=f"openai/{LLM_MODEL_LIGHT}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_tokens=4096,
    )


def create_parser_agent() -> Agent:
    return Agent(
        role="Mathematical Task Parser",
        goal=(
            "Extract the differential equation, task type, and solution method "
            "from the user's markdown text. Return structured JSON."
        ),
        backstory=(
            "You parse mathematical tasks written in Russian or English. "
            "You extract: the ODE equation in SymPy-compatible format (using y', y'', x, y), "
            "the task type (solve, isoclines, phase_portrait, direction_field, classify), "
            "and the solution method if specified (auto if not). "
            "You also extract initial conditions if present.\n\n"
            "IMPORTANT format rules:\n"
            "- Equation must use: y' for first derivative, y'' for second, x and y as variables\n"
            "- Replace LaTeX notation: \\frac{dy}{dx} -> y', x^2 -> x**2\n"
            "- task_type must be one of: solve, isoclines, phase_portrait, direction_field, classify\n"
            "- solve_method must be one of: auto, separable, 1st_linear, "
            "1st_homogeneous_coeff_best, 1st_exact, Bernoulli, Riccati_special_minus2, "
            "nth_linear_constant_coeff_variation_of_parameters, "
            "nth_linear_constant_coeff_undetermined_coefficients\n"
            "- If no specific method is mentioned, use 'auto'"
        ),
        tools=[],
        llm=get_llm(),
        verbose=True,
    )
