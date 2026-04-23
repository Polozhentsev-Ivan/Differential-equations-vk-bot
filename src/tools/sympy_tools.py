"""SymPy tools для решения, классификации и валидации ОДУ."""

from __future__ import annotations

import re
from typing import Optional

from sympy import (
    Derivative,
    Eq,
    Function,
    Symbol,
    classify_ode as _classify_ode,
    checkodesol as _checkodesol,
    dsolve as _dsolve,
    solve,
    symbols,
)
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from models.schemas import (
    IsoclineEntry,
    IsoclinesResult,
    SolveMethod,
    SolverResult,
    ValidationResult,
)

x = Symbol("x")
y = Function("y")
_YX = Symbol("_YX")

_PARSE_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _parse_math_expr(expr_str: str):
    """Преобразует строку математического выражения в SymPy-выражение.

    Автоматически заменяет ``y(x)`` и ``y`` на плейсхолдер перед парсингом,
    чтобы ``implicit_multiplication_application`` не искажала результат.
    """
    s = expr_str.strip()
    s = s.replace("^", "**")

    s = re.sub(r"\by\s*\(\s*x\s*\)", "_YX", s)
    s = re.sub(r"\by\b", "_YX", s)

    local: dict = {"x": x, "_YX": _YX}
    result = parse_expr(s, local_dict=local, transformations=_PARSE_TRANSFORMATIONS)
    return result.subs(_YX, y(x))


def _replace_derivatives(s: str) -> str:
    """Заменяет y''', y'', y', dy/dx, d2y/dx2 на плейсхолдеры _DN_.

    Возвращает (строка с плейсхолдерами, dict плейсхолдер → порядок).
    """
    mapping: dict[str, int] = {}

    def _tick_repl(m: re.Match) -> str:
        order = len(m.group(1))
        placeholder = f"_D{order}_"
        mapping[placeholder] = order
        return placeholder

    s = re.sub(r"y('+)(?:\(x\))?", _tick_repl, s)

    def _dn_repl(m: re.Match) -> str:
        order = int(m.group(1))
        placeholder = f"_D{order}_"
        mapping[placeholder] = order
        return placeholder

    s = re.sub(r"d(\d+)y/dx\d+", _dn_repl, s)
    s = re.sub(r"dy/dx", lambda _: (mapping.update({"_D1_": 1}), "_D1_")[1], s)

    return s, mapping


def parse_ode(equation_str: str) -> Eq:
    """Парсит строковое ОДУ в SymPy Eq, пригодный для dsolve.

    Поддерживает деривативы в обеих частях и смешанные выражения вида::

        y' + y = x
        y'' + 3y' + 2y = sin(x)
        dy/dx = x*y
    """
    eq_str = equation_str.strip()

    if "=" not in eq_str:
        raise ValueError(f"Уравнение должно содержать '=': {eq_str}")

    lhs_str, rhs_str = eq_str.split("=", 1)
    full_expr = f"({lhs_str}) - ({rhs_str})"

    full_expr, deriv_map = _replace_derivatives(full_expr)

    placeholder_symbols = {name: Symbol(name) for name in deriv_map}

    full_expr = full_expr.replace("^", "**")
    full_expr = re.sub(r"\by\s*\(\s*x\s*\)", "_YX", full_expr)
    full_expr = re.sub(r"\by\b", "_YX", full_expr)

    local: dict = {"x": x, "_YX": _YX}
    local.update(placeholder_symbols)

    parsed = parse_expr(full_expr, local_dict=local, transformations=_PARSE_TRANSFORMATIONS)
    parsed = parsed.subs(_YX, y(x))

    for name, order in deriv_map.items():
        parsed = parsed.subs(placeholder_symbols[name], y(x).diff(x, order))

    return Eq(parsed, 0)


def solve_ode(
    equation_str: str,
    method: SolveMethod = SolveMethod.AUTO,
    initial_conditions: Optional[dict[str, float]] = None,
) -> SolverResult:
    """Решает ОДУ с помощью SymPy dsolve.

    Args:
        equation_str: уравнение в строковом виде, напр. ``"y' = x*y"``
        method: метод решения (AUTO = автовыбор)
        initial_conditions: начальные условия вида ``{"y(0)": 1}``
    """
    try:
        eq = parse_ode(equation_str)
    except Exception as exc:
        return SolverResult(success=False, error=f"Ошибка парсинга: {exc}")

    classification = None
    try:
        classification = list(_classify_ode(eq, y(x)))
    except Exception:
        pass

    hint = method.value if method != SolveMethod.AUTO else None

    ics_sympy = None
    if initial_conditions:
        ics_sympy = {}
        for k, v in initial_conditions.items():
            ics_sympy[_parse_math_expr(k)] = v

    try:
        if hint:
            sol = _dsolve(eq, y(x), hint=hint, ics=ics_sympy)
        else:
            sol = _dsolve(eq, y(x), ics=ics_sympy)
    except Exception as exc:
        return SolverResult(
            success=False,
            classification=classification,
            error=f"SymPy dsolve не смог решить: {exc}",
        )

    method_used = hint or (classification[0] if classification else "default")

    return SolverResult(
        success=True,
        solution=str(sol),
        ode_type=classification[0] if classification else None,
        method_used=method_used,
        classification=classification,
    )


def classify_ode_type(equation_str: str) -> list[str]:
    """Возвращает список подходящих методов решения (SymPy hints)."""
    eq = parse_ode(equation_str)
    return list(_classify_ode(eq, y(x)))


def _parse_solution(solution_str: str) -> Eq:
    """Парсит строковое решение в SymPy Eq.

    Обрабатывает формат вывода dsolve: ``Eq(y(x), C1*exp(x**2/2))``
    а также простые выражения: ``C1*exp(x**2/2)``.
    """
    s = solution_str.strip()
    s = re.sub(r"\by\s*\(\s*x\s*\)", "_YX", s)
    s = re.sub(r"\by\b", "_YX", s)

    local: dict = {"x": x, "_YX": _YX, "Eq": Eq, "C1": Symbol("C1"), "C2": Symbol("C2"), "C3": Symbol("C3")}

    expr = parse_expr(s, local_dict=local, transformations=_PARSE_TRANSFORMATIONS)
    expr = expr.subs(_YX, y(x))
    if isinstance(expr, Eq):
        return expr
    return Eq(y(x), expr)


def validate_solution(equation_str: str, solution_str: str) -> ValidationResult:
    """Проверяет решение подстановкой в исходное уравнение (checkodesol)."""
    try:
        eq = parse_ode(equation_str)
    except Exception as exc:
        return ValidationResult(is_valid=False, details=f"Ошибка парсинга уравнения: {exc}")

    try:
        sol_eq = _parse_solution(solution_str)
    except Exception as exc:
        return ValidationResult(is_valid=False, details=f"Ошибка парсинга решения: {exc}")

    try:
        result = _checkodesol(eq, sol_eq)
        is_valid = result[0]
        return ValidationResult(
            is_valid=bool(is_valid),
            details=f"Остаток: {result[1]}" if not is_valid else "Решение корректно",
        )
    except Exception as exc:
        return ValidationResult(is_valid=False, details=f"Ошибка валидации: {exc}")


def compute_isoclines(
    equation_str: str,
    c_values: Optional[list[float]] = None,
) -> IsoclinesResult:
    """Вычисляет изоклины для ОДУ первого порядка y' = f(x, y).

    Изоклина для наклона C — кривая f(x, y) = C,
    т.е. y выражается через x и C (если возможно).
    """
    if c_values is None:
        c_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    eq = parse_ode(equation_str)

    dy = y(x).diff(x)
    f_solutions = solve(eq, dy)
    if not f_solutions:
        raise ValueError(f"Не удалось выделить y' из уравнения: {equation_str}")
    rhs = f_solutions[0]

    y_sym = Symbol("_y")
    rhs_with_y_sym = rhs.subs(y(x), y_sym)

    entries: list[IsoclineEntry] = []
    for c_val in c_values:
        isocline_eq = Eq(rhs_with_y_sym, c_val)
        try:
            solutions = solve(isocline_eq, y_sym)
            for sol in solutions:
                entries.append(
                    IsoclineEntry(
                        c_value=c_val,
                        curve_equation=f"y = {sol.subs(y_sym, Symbol('y'))}",
                    )
                )
        except Exception:
            entries.append(
                IsoclineEntry(
                    c_value=c_val,
                    curve_equation=str(isocline_eq),
                )
            )

    return IsoclinesResult(
        original_equation=equation_str,
        isoclines=entries,
    )
