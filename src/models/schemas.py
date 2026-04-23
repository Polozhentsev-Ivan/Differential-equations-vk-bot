from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskType(str, Enum):
    SOLVE = "solve"
    ISOCLINES = "isoclines"
    PHASE_PORTRAIT = "phase_portrait"
    DIRECTION_FIELD = "direction_field"
    CLASSIFY = "classify"


class SolveMethod(str, Enum):
    """Метод решения ОДУ. Значения совпадают с hint-ами SymPy dsolve."""

    AUTO = "auto"
    SEPARABLE = "separable"
    LINEAR_FIRST_ORDER = "1st_linear"
    HOMOGENEOUS = "1st_homogeneous_coeff_best"
    EXACT = "1st_exact"
    BERNOULLI = "Bernoulli"
    RICCATI = "Riccati_special_minus2"
    VARIATION_OF_PARAMS = "nth_linear_constant_coeff_variation_of_parameters"
    UNDETERMINED_COEFFS = "nth_linear_constant_coeff_undetermined_coefficients"


# Маппинг человеко-читаемых названий → enum (для Parser Agent)
SOLVE_METHOD_ALIASES: dict[str, SolveMethod] = {
    "разделение переменных": SolveMethod.SEPARABLE,
    "разделением переменных": SolveMethod.SEPARABLE,
    "separable": SolveMethod.SEPARABLE,
    "линейное": SolveMethod.LINEAR_FIRST_ORDER,
    "линейным": SolveMethod.LINEAR_FIRST_ORDER,
    "1st_linear": SolveMethod.LINEAR_FIRST_ORDER,
    "однородное": SolveMethod.HOMOGENEOUS,
    "однородным": SolveMethod.HOMOGENEOUS,
    "homogeneous": SolveMethod.HOMOGENEOUS,
    "точное": SolveMethod.EXACT,
    "точным": SolveMethod.EXACT,
    "exact": SolveMethod.EXACT,
    "бернулли": SolveMethod.BERNOULLI,
    "bernoulli": SolveMethod.BERNOULLI,
    "рикатти": SolveMethod.RICCATI,
    "riccati": SolveMethod.RICCATI,
    "вариация постоянных": SolveMethod.VARIATION_OF_PARAMS,
    "вариацией постоянных": SolveMethod.VARIATION_OF_PARAMS,
    "метод вариации постоянных": SolveMethod.VARIATION_OF_PARAMS,
    "variation of parameters": SolveMethod.VARIATION_OF_PARAMS,
    "неопределённые коэффициенты": SolveMethod.UNDETERMINED_COEFFS,
    "неопределёнными коэффициентами": SolveMethod.UNDETERMINED_COEFFS,
    "метод неопределённых коэффициентов": SolveMethod.UNDETERMINED_COEFFS,
    "undetermined coefficients": SolveMethod.UNDETERMINED_COEFFS,
}


class ParsedTask(BaseModel):
    equation: str
    task_type: TaskType
    solve_method: SolveMethod = SolveMethod.AUTO
    initial_conditions: Optional[dict[str, float]] = None
    parameters: Optional[dict[str, str]] = None
    original_text: str = ""


class SolverResult(BaseModel):
    success: bool
    solution: Optional[str] = None
    ode_type: Optional[str] = None
    method_used: Optional[str] = None
    classification: Optional[list[str]] = None
    error: Optional[str] = None


class ValidationResult(BaseModel):
    is_valid: bool
    details: Optional[str] = None


class IsoclineEntry(BaseModel):
    c_value: float
    curve_equation: str


class IsoclinesResult(BaseModel):
    original_equation: str
    isoclines: list[IsoclineEntry]
    plot_path: Optional[str] = None


class PipelineResult(BaseModel):
    """Результат работы всего пайплайна."""
    success: bool
    input_file: str
    output_file: Optional[str] = None
    pdf_file: Optional[str] = None
    plot_file: Optional[str] = None
    parsed_task: Optional[ParsedTask] = None
    solver_result: Optional[SolverResult] = None
    error: Optional[str] = None
    stage_failed: Optional[str] = None
