"""CrewAI tool-обёртки над SymPy функциями и визуализацией."""

from __future__ import annotations

from crewai.tools import tool

from tools.sympy_tools import (
    classify_ode_type,
    compute_isoclines,
    solve_ode,
    validate_solution,
)
from tools.plot_tools import plot_isoclines_full
from models.schemas import SolveMethod


@tool("classify_ode")
def classify_ode_tool(equation: str) -> str:
    """Classify an ODE and return a list of applicable solution methods (SymPy hints).

    Args:
        equation: The ODE as a string, e.g. "y' = x*y" or "y'' + y = 0".
    """
    try:
        hints = classify_ode_type(equation)
        return f"Classification for '{equation}':\n" + "\n".join(f"  - {h}" for h in hints)
    except Exception as exc:
        return f"Error classifying ODE: {exc}"


@tool("solve_ode")
def solve_ode_tool(equation: str, method: str = "auto") -> str:
    """Solve an ODE using SymPy. Returns the solution or an error message.

    Args:
        equation: The ODE as a string, e.g. "y' = x*y".
        method: Solution method hint. Use "auto" for automatic selection,
                or a specific SymPy hint like "separable", "1st_linear", "Bernoulli", etc.
    """
    solve_method = SolveMethod.AUTO
    if method != "auto":
        try:
            solve_method = SolveMethod(method)
        except ValueError:
            solve_method = SolveMethod.AUTO

    result = solve_ode(equation, method=solve_method)

    if not result.success:
        return f"Failed to solve: {result.error}"

    lines = [
        f"Equation: {equation}",
        f"Solution: {result.solution}",
        f"ODE type: {result.ode_type}",
        f"Method used: {result.method_used}",
    ]
    return "\n".join(lines)


@tool("validate_solution")
def validate_solution_tool(equation: str, solution: str) -> str:
    """Validate an ODE solution by substituting it back into the original equation.

    Args:
        equation: The original ODE, e.g. "y' = x*y".
        solution: The proposed solution as returned by solve_ode, e.g. "Eq(y(x), C1*exp(x**2/2))".
    """
    result = validate_solution(equation, solution)
    if result.is_valid:
        return f"VALID: {result.details}"
    return f"INVALID: {result.details}"


@tool("compute_isoclines")
def compute_isoclines_tool(equation: str, c_values: str = "-2,-1,0,1,2") -> str:
    """Compute isoclines for a first-order ODE y' = f(x, y).

    An isocline for slope C is the curve where f(x, y) = C.

    Args:
        equation: The ODE, e.g. "y' = x**2 + y".
        c_values: Comma-separated slope values, e.g. "-2,-1,0,1,2".
    """
    try:
        c_list = [float(c.strip()) for c in c_values.split(",")]
    except ValueError:
        c_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

    try:
        result = compute_isoclines(equation, c_values=c_list)
    except Exception as exc:
        return f"Error computing isoclines: {exc}"

    lines = [f"Isoclines for '{equation}':"]
    for entry in result.isoclines:
        lines.append(f"  C = {entry.c_value}: {entry.curve_equation}")
    return "\n".join(lines)


@tool("plot_isoclines")
def plot_isoclines_tool(equation: str, c_values: str = "-3,-2,-1,0,1,2,3", output_path: str = "output/isoclines.png") -> str:
    """Plot direction field, isoclines and solution curves for a first-order ODE.

    Generates a PNG image with:
    - Direction field (gray arrows)
    - Colored isocline curves for each slope value C
    - Blue integral curves (numerical solutions)

    Args:
        equation: The ODE, e.g. "y' = x**2 + y".
        c_values: Comma-separated slope values for isoclines.
        output_path: Path where to save the PNG image.
    """
    try:
        c_list = [float(c.strip()) for c in c_values.split(",")]
    except ValueError:
        c_list = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    try:
        saved = plot_isoclines_full(equation, c_values=c_list, output_path=output_path)
        return f"Plot saved to: {saved}"
    except Exception as exc:
        return f"Error plotting: {exc}"
