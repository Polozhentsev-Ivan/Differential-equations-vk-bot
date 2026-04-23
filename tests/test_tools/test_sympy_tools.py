"""Тесты SymPy tools — вызываем напрямую, без агентов."""

import pytest
from sympy import Eq, Function, Symbol

from models.schemas import SolveMethod
from tools.sympy_tools import (
    classify_ode_type,
    compute_isoclines,
    parse_ode,
    solve_ode,
    validate_solution,
)

x = Symbol("x")
y = Function("y")


# ───────────────────────── parse_ode ─────────────────────────


class TestParseOde:
    def test_first_order_tick(self):
        eq = parse_ode("y' = x*y")
        assert isinstance(eq, Eq)
        assert eq.has(y(x).diff(x))

    def test_first_order_dydx(self):
        eq = parse_ode("dy/dx = x + y")
        assert eq.has(y(x).diff(x))

    def test_second_order_ticks(self):
        eq = parse_ode("y'' = -y")
        assert eq.has(y(x).diff(x, 2))

    def test_second_order_dn(self):
        eq = parse_ode("d2y/dx2 = -y")
        assert eq.has(y(x).diff(x, 2))

    def test_caret_as_power(self):
        eq = parse_ode("y' = x^2 + y")
        assert eq.has(y(x).diff(x))

    def test_no_equals_raises(self):
        with pytest.raises(ValueError, match="="):
            parse_ode("y' + x*y")

    def test_mixed_lhs(self):
        """y' + y = x — деривативы смешаны с другими членами."""
        eq = parse_ode("y' + y = x")
        assert eq.has(y(x).diff(x))
        assert eq.has(y(x))

    def test_second_order_mixed(self):
        """y'' + 3y' + 2y = 0"""
        eq = parse_ode("y'' + 3*y' + 2*y = 0")
        assert eq.has(y(x).diff(x, 2))
        assert eq.has(y(x).diff(x))


# ───────────────────────── solve_ode ─────────────────────────


class TestSolveOde:
    def test_separable_auto(self):
        result = solve_ode("y' = x*y")
        assert result.success
        assert result.solution is not None
        assert "exp" in result.solution

    def test_separable_explicit_method(self):
        result = solve_ode("y' = x*y", method=SolveMethod.SEPARABLE)
        assert result.success
        assert result.solution is not None

    def test_linear_first_order(self):
        result = solve_ode("y' + y = x")
        assert result.success
        assert result.solution is not None

    def test_second_order(self):
        result = solve_ode("y'' + y = 0")
        assert result.success
        assert "sin" in result.solution or "cos" in result.solution

    def test_bad_equation_returns_error(self):
        result = solve_ode("это не уравнение")
        assert not result.success
        assert result.error is not None

    def test_wrong_method_returns_error(self):
        result = solve_ode("y'' + y = 0", method=SolveMethod.SEPARABLE)
        assert not result.success


# ──────────────────────── classify_ode ───────────────────────


class TestClassifyOde:
    def test_separable_classified(self):
        hints = classify_ode_type("y' = x*y")
        assert any("separable" in h for h in hints)

    def test_linear_classified(self):
        hints = classify_ode_type("y' + y = x")
        assert any("linear" in h for h in hints)

    def test_returns_list(self):
        hints = classify_ode_type("y' = x + y")
        assert isinstance(hints, list)
        assert len(hints) > 0


# ─────────────────────── validate_solution ───────────────────


class TestValidateSolution:
    def test_correct_solution(self):
        result = solve_ode("y' = x*y")
        assert result.success

        val = validate_solution("y' = x*y", result.solution)
        assert val.is_valid

    def test_wrong_solution(self):
        val = validate_solution("y' = x*y", "Eq(y(x), x)")
        assert not val.is_valid


# ─────────────────────── compute_isoclines ───────────────────


class TestComputeIsoclines:
    def test_basic_isoclines(self):
        result = compute_isoclines("y' = x**2 + y")
        assert len(result.isoclines) > 0
        assert result.original_equation == "y' = x**2 + y"

    def test_custom_c_values(self):
        result = compute_isoclines("y' = x + y", c_values=[0.0, 1.0])
        assert len(result.isoclines) == 2

    def test_isocline_equations_contain_x(self):
        result = compute_isoclines("y' = x**2 + y", c_values=[0.0])
        assert "x" in result.isoclines[0].curve_equation
